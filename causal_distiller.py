import logging
import os
import random
import pickle
import time
import psutil
import wandb

import numpy as np
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from tqdm import tqdm, trange
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from BERT.pytorch_pretrained_bert.modeling import BertConfig
from BERT.pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from BERT.pytorch_pretrained_bert.tokenization import BertTokenizer
from iit_modelings.diito_transformer import InterventionableEncoder

from src.argument_parser import default_parser, get_predefine_argv, complete_argument
from src.nli_data_processing import processors, output_modes
from src.data_processing import init_model, get_task_dataloader
from src.modeling import BertForSequenceClassificationEncoder, FCClassifierForSequenceClassification, FullFCClassifierForSequenceClassification
from src.utils import load_model, count_parameters, eval_model_dataloader_nli, eval_model_dataloader
from src.KD_loss import distillation_loss, patience_loss, diito_distillation_loss
from envs import HOME_DATA_FOLDER

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskSpecificCausalDistiller:
    def __init__(
        self, params, 
        train_dataset, eval_dataset, 
        eval_label_ids, num_labels, output_mode,
        student_encoder: nn.Module, student_classifier: nn.Module,
        teacher_encoder: nn.Module, teacher_classifier: nn.Module,
    ):
        if params.is_wandb:
            run = wandb.init(
                project=params.wandb_metadata.split(":")[-1], 
                entity=params.wandb_metadata.split(":")[0],
                name=params.run_name,
            )
            wandb.config.update(params)
        self.is_wandb = params.is_wandb
        logger.info("Initializing Normal Distiller (Task Specific)")
        
        self.params = params
        
        self.output_model_file = '{}_nlayer.{}_lr.{}_T.{}.alpha.{}_beta.{}_bs.{}_diito.{}_nm.{}_intprop.{}_intmax.{}_intconsec.{}_dtaug.{}_dtpair.{}_maxex.{}'.format(
            self.params.task_name, 
            self.params.student_hidden_layers,
            self.params.learning_rate,
            self.params.T, 
            self.params.alpha, 
            self.params.beta,
            self.params.train_batch_size * self.params.gradient_accumulation_steps,
            self.params.is_diito,
            self.params.neuron_mapping,
            self.params.interchange_prop,
            self.params.interchange_max_token,
            self.params.interchange_consecutive_only,
            self.params.data_augment,
            self.params.data_pair,
            self.params.max_training_examples,
        )
        
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.eval_label_ids = eval_label_ids
        self.num_labels = num_labels
        self.output_mode = output_mode
        
        # common used vars
        self.local_rank = -1
        self.fp16 = params.fp16
        self.T = params.T
        self.alpha = params.alpha
        self.beta = params.beta
        self.normalize_patience = params.normalize_patience
        self.learning_rate = params.learning_rate
        self.train_batch_size = params.train_batch_size
        self.output_dir = params.output_dir
        self.warmup_proportion = params.warmup_proportion
        self.num_train_optimization_steps = params.num_train_optimization_steps
        self.task_name = params.task_name
        self.kd_model = params.kd_model 
        self.weights = params.weights
        self.fc_layer_idx = params.fc_layer_idx
        self.n_gpu = params.n_gpu
        self.device = params.device
        self.num_train_epochs = params.num_train_epochs
        self.gradient_accumulation_steps = params.gradient_accumulation_steps
        self.loss_scale = params.loss_scale
        
        # DIITO params
        self.is_diito = params.is_diito
        self.diito_type = params.diito_type
        self.interchange_prop = params.interchange_prop
        self.interchange_max_token = params.interchange_max_token
        self.interchange_masked_token_only = False # this is not supported.
        self.interchange_consecutive_only = params.interchange_consecutive_only
        self.data_augment = params.data_augment
        self.data_pair = params.data_pair
        
        # models
        self.student_encoder = student_encoder
        self.student_classifier = student_classifier
        self.teacher_encoder = teacher_encoder
        self.teacher_classifier = teacher_classifier
        
        # getting params to optimize early
        param_optimizer = list(self.student_encoder.named_parameters())
        param_optimizer += list(
            self.student_classifier.named_parameters()
        )
        
        # parallel models
        self.student_encoder.to(self.device)
        self.student_classifier.to(self.device)
        self.teacher_encoder.to(self.device)
        self.teacher_classifier.to(self.device)
        if self.local_rank != -1:
            raise NotImplementedError('not implemented for local_rank != 1')
        elif self.n_gpu > 1:
            logger.info('data parallel because more than one gpu for all models')
            self.student_encoder = torch.nn.DataParallel(self.student_encoder)
            self.student_classifier = torch.nn.DataParallel(self.student_classifier)
            self.teacher_encoder = torch.nn.DataParallel(self.teacher_encoder)
            self.teacher_classifier = torch.nn.DataParallel(self.teacher_classifier)

        # make the model interventionable if diito is enabled.
        if self.is_diito:
            self.student_encoder = InterventionableEncoder(self.student_encoder)
            self.teacher_encoder = InterventionableEncoder(self.teacher_encoder)
            try:
                self.teacher_hidden_layers = self.teacher_encoder.model.config.num_hidden_layers
            except:
                self.teacher_hidden_layers = self.teacher_encoder.model.module.config.num_hidden_layers
        else:
            try:
                self.teacher_hidden_layers = self.teacher_encoder.config.num_hidden_layers
            except:
                self.teacher_hidden_layers = self.teacher_encoder.module.config.num_hidden_layers
            
        self.student_hidden_layers = params.student_hidden_layers
        self.neuron_mapping = params.neuron_mapping
        self.layer_mapping = {}
        if self.neuron_mapping == "full":
            layer_count = self.teacher_hidden_layers // self.student_hidden_layers
            for i in range(0, self.student_hidden_layers):
                self.layer_mapping[i] = []
                for j in range(0, layer_count):
                    self.layer_mapping[i] += [(i*layer_count+j)]
        elif self.neuron_mapping == "late":
            assert False # Not Implemented
        elif self.neuron_mapping == "mid":
            assert False # Not Implemented
        logger.info(f'neuron mapping: {self.neuron_mapping}')
        logger.info(f'corresponding layer mapping:')
        logger.info(self.layer_mapping)

        # log to a local file
        log_train = open(os.path.join(self.output_dir, 'train_log.txt'), 'w', buffering=1)
        log_eval = open(os.path.join(self.output_dir, 'eval_log.txt'), 'w', buffering=1)
        print('epoch,global_steps,step,acc,loss,kd_loss,ce_loss,AT_loss', file=log_train)
        print('epoch,acc,loss', file=log_eval)
        log_train.close()
        log_eval.close()
            
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if self.fp16:
            logger.info('FP16 activate, use apex FusedAdam')
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            self.optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=self.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if self.loss_scale == 0:
                self.optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                self.optimizer = FP16_Optimizer(optimizer, static_loss_scale=self.loss_scale)
        else:
            logger.info('FP16 is not activated, use BertAdam')
            self.optimizer = BertAdam(
                optimizer_grouped_parameters,
                lr=self.learning_rate,
                warmup=self.warmup_proportion,
                t_total=self.num_train_optimization_steps
            )
        
        # other params that report to tensorboard
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sequences_epoch = 0
        self.total_loss_epoch = 0
        self.last_loss = 0
        self.last_loss_dl = 0
        self.last_kd_loss = 0
        self.last_ce_loss = 0 
        self.last_pt_loss = 0
        self.lr_this_step = 0
        self.last_log = 0
        
        self.last_loss_diito = 0
        self.last_cf_pt_loss = 0
        
        self.acc_tr_loss = 0
        self.acc_tr_kd_loss = 0
        self.acc_tr_ce_loss = 0
        self.acc_tr_pt_loss = 0
        self.acc_tr_acc = 0
        
        self.acc_tr_loss_diito = 0
        self.acc_tr_cf_pt_loss = 0
        self.acc_tr_cf_acc = 0
        
        self.tr_loss = 0
        self.tr_kd_loss = 0
        self.tr_ce_loss = 0
        self.tr_pt_loss = 0
        self.tr_acc = 0
        
        self.tr_loss_diito = 0
        self.tr_cf_pt_loss = 0
        self.tr_cf_acc = 0
        
        # DIITO related params that report to tensorboard
    
    def prepare_batch(self, input_ids, input_mask, segment_ids, label_ids, ):
        if self.is_diito:
            dual_input_ids = input_ids.clone()
            dual_input_mask = input_mask.clone()
            dual_segment_ids = segment_ids.clone()
            dual_label_ids = label_ids.clone()
            causal_sort_index = [i for i in range(dual_input_ids.shape[0])]
            random.shuffle(causal_sort_index)
            dual_input_ids = dual_input_ids[causal_sort_index]
            dual_input_mask = dual_input_mask[causal_sort_index]
            dual_segment_ids = dual_segment_ids[causal_sort_index]
            dual_label_ids = dual_label_ids[causal_sort_index]
            return dual_input_ids, dual_input_mask, dual_segment_ids, dual_label_ids, \
                input_ids, input_mask, segment_ids, label_ids
        else:
            return input_ids, input_mask, segment_ids, label_ids
    
    def train(self):
        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0
        
        self.student_encoder.train()
        self.teacher_encoder.eval()
        self.student_classifier.train()
        self.teacher_classifier.eval()
        
        self.last_log = time.time()
        
        for epoch in trange(int(self.num_train_epochs), desc="Epoch"):
            tr_loss, tr_ce_loss, tr_kd_loss, tr_acc = 0, 0, 0, 0
            
            nb_tr_examples, nb_tr_steps = 0, 0
            
            iter_bar = tqdm(self.train_dataset, desc="-Iter", disable=False)
            for batch in iter_bar:
                batch = tuple(t.to(self.device) for t in batch)
                # teascher patient is on-the-fly, we can skip the logic for different batch format.
                prepared_batch = self.prepare_batch(
                    *batch,
                )
                if self.is_diito:
                    self.step_diito(
                        *prepared_batch
                    )
                elif self.is_diito:
                    pass
                elif self.is_diito:
                    pass
                else:
                    self.step(
                        *prepared_batch
                    )
                iter_bar.update()
                iter_bar.set_postfix(
                    {
                        "Last_loss": f"{self.last_loss:.2f}", 
                        "Avg_cum_loss": f"{self.total_loss_epoch/self.n_iter:.2f}", 
                    }
                )
            iter_bar.close()

            logger.info(f"--- Ending epoch {self.epoch}/{self.num_train_epochs-1}")
            self.end_epoch()

        logger.info("Save very last checkpoint as `pytorch_model.bin`.")
        self.save_checkpoint(checkpoint_name="pytorch_model.bin")
        logger.info("Training is finished")
    
    def prepare_interchange_mask(
        self,
        lengths, dual_lengths,
        pred_mask, dual_pred_mask,
    ):        
        # params
        interchange_prop = self.interchange_prop
        interchange_max_token = self.interchange_max_token # if -1 then we don't restrict on this.
        interchange_masked_token_only = self.interchange_masked_token_only
        interchange_consecutive_only = self.interchange_consecutive_only
        
        interchange_mask = torch.zeros_like(pred_mask, dtype=torch.bool)
        dual_interchange_mask = torch.zeros_like(dual_pred_mask, dtype=torch.bool)

        batch_size, max_seq_len = pred_mask.shape[0], pred_mask.shape[1]
        _, dual_max_seq_len = dual_pred_mask.shape[0], dual_pred_mask.shape[1]
        interchange_position = []
        for i in range(0, batch_size):
            min_len = min(lengths[i].tolist(), dual_lengths[i].tolist())
            if interchange_consecutive_only:
                if interchange_max_token != -1:
                    interchange_count = min(interchange_max_token, int(min_len*interchange_prop))
                else:
                    interchange_count = int(min_len*interchange_prop)
                start_index = random.randint(0, lengths[i].tolist()-interchange_count)
                end_index = start_index + interchange_count
                dual_start_index = random.randint(0, dual_lengths[i].tolist()-interchange_count)
                dual_end_index = dual_start_index + interchange_count
                interchange_mask[i][start_index:end_index] = 1
                dual_interchange_mask[i][dual_start_index:dual_end_index] = 1
            else:
                # we follow these steps to sample the position:
                # 1. sample positions in the main example
                # 2. get the actual sampled positions
                # 3. sample accordingly from the dual example
                if interchange_masked_token_only:
                    # a corner case we need to consider is that the masked token
                    # numbers may differ across two examples.
                    interchange_count = pred_mask[i].sum()
                    if interchange_count > dual_lengths[i]:
                        # not likely, but we need to handle this.
                        interchange_count = dual_lengths[i]
                    interchange_position = pred_mask[i].nonzero().view(-1).tolist()
                    interchange_position = random.sample(interchange_position, interchange_count)
                    interchange_mask[i][interchange_position] = 1
                    dual_interchange_position = random.sample(range(dual_max_seq_len), interchange_count)
                    dual_interchange_mask[i][dual_interchange_position] = 1
                else:
                    if interchange_max_token != -1:
                        interchange_count = min(interchange_max_token, int(min_len*interchange_prop))
                    else:
                        interchange_count = int(min_len*interchange_prop)
                    interchange_position = random.sample(range(max_seq_len), interchange_count)
                    interchange_mask[i][interchange_position] = 1
                    dual_interchange_position = random.sample(range(dual_max_seq_len), interchange_count)
                    dual_interchange_mask[i][dual_interchange_position] = 1

        # sanity checks
        assert interchange_mask.long().sum(dim=-1).tolist() == \
                dual_interchange_mask.long().sum(dim=-1).tolist()

        return interchange_mask, dual_interchange_mask
    
    def step_data_pair(
        self,
        source_input_ids,
        source_input_mask,
        source_segment_ids,
        source_label_ids,
        base_input_ids,
        base_input_mask,
        base_segment_ids,
        base_label_ids,
    ):
        self.step(
            base_input_ids,
            base_input_mask,
            base_segment_ids,
            base_label_ids,
            skip_update_iter=True,
        )
        self.step(
            source_input_ids,
            source_input_mask,
            source_segment_ids,
            source_label_ids,
        )
        # two examples but only update once.
    
    def step_data_augment(
        self,
        source_input_ids,
        source_input_mask,
        source_segment_ids,
        source_label_ids,
        base_input_ids,
        base_input_mask,
        base_segment_ids,
        base_label_ids,
    ):
        source_intervention_mask, base_intervention_mask = self.prepare_interchange_mask(
            source_input_mask.sum(dim=-1), base_input_mask.sum(dim=-1),
            source_input_mask, base_input_mask,
        )
        student_invention_layer = random.choice(list(self.layer_mapping.keys()))
        teacher_invention_layer = self.layer_mapping[student_invention_layer]

        ##########
        # teacher
        with torch.no_grad():
            if self.alpha == 0:
                teacher_pred, teacher_patience = None, None
            else:
                _, teacher_base_outputs, teacher_counterfactual_outputs = \
                    self.teacher_encoder.forward_data_augment(
                        source=[
                            source_input_ids, source_segment_ids, source_input_mask, 
                        ],
                        base=[
                            base_input_ids, base_segment_ids, base_input_mask,
                        ],
                        source_mask=source_intervention_mask, 
                        base_mask=base_intervention_mask,
                    )
                full_output_teacher, pooled_output_teacher = teacher_base_outputs
                cf_full_output_teacher, cf_pooled_output_teacher = teacher_counterfactual_outputs
                
                if self.kd_model.lower() in['kd', 'kd.cls']:
                    
                    teacher_pred = self.teacher_classifier(pooled_output_teacher)
                    cf_teacher_pred = self.teacher_classifier(cf_pooled_output_teacher)
                    
                    if self.kd_model.lower() == 'kd.cls':
                        
                        teacher_patience = torch.stack(full_output_teacher[:-1]).transpose(0, 1)
                        cf_teacher_patience = torch.stack(cf_full_output_teacher[:-1]).transpose(0, 1)
                        if self.fp16:
                            teacher_patience = teacher_patience.half()
                            cf_teacher_patience = cf_teacher_patience.half()
                        layer_index = [int(i) for i in self.fc_layer_idx.split(',')]
                        teacher_patience = torch.stack(
                            [torch.FloatTensor(teacher_patience[:,int(i)]) for i in layer_index]
                        ).transpose(0, 1)
                        cf_teacher_patience = torch.stack(
                            [torch.FloatTensor(cf_teacher_patience[:,int(i)]) for i in layer_index]
                        ).transpose(0, 1)
                        
                    else:
                        teacher_patience = None
                        cf_teacher_patience = None
                else:
                    raise ValueError(f'{self.kd_model} not implemented yet')
                if self.fp16:
                    teacher_pred = teacher_pred.half()
                    cf_teacher_pred = cf_teacher_pred.half()
        
        
        ##########
        # student
        _, student_base_outputs, student_counterfactual_outputs = \
            self.student_encoder.forward_data_augment(
                source=[
                    source_input_ids, source_segment_ids, source_input_mask, 
                ],
                base=[
                    base_input_ids, base_segment_ids, base_input_mask,
                ],
                source_mask=source_intervention_mask, 
                base_mask=base_intervention_mask,
            )
        full_output_student, pooled_output_student = student_base_outputs
        cf_full_output_student, cf_pooled_output_student = student_counterfactual_outputs
        
        if self.kd_model.lower() in['kd', 'kd.cls']:
            logits_pred_student = self.student_classifier(
                pooled_output_student
            )
            cf_logits_pred_student = self.student_classifier(
                cf_pooled_output_student
            )
            if self.kd_model.lower() == 'kd.cls':
                student_patience = torch.stack(full_output_student[:-1]).transpose(0, 1)
                cf_student_patience = torch.stack(cf_full_output_student[:-1]).transpose(0, 1)
            else:
                student_patience = None
                cf_student_patience = None
        else:
            raise ValueError(f'{self.kd_model} not implemented yet')
        
        # calculate loss, along with counterfactual loss
        loss_dl, kd_loss, ce_loss = distillation_loss(
            logits_pred_student, source_label_ids, teacher_pred, T=self.T, alpha=self.alpha
        )
        loss_diito = diito_distillation_loss(
            logits_pred_student, teacher_pred, T=self.T, alpha=self.alpha
        )
        
        if self.beta > 0:
            if student_patience.shape[0] != input_ids.shape[0]:
                # For RACE
                n_layer = student_patience.shape[1]
                student_patience = student_patience.transpose(0, 1).contiguous().view(
                    n_layer, input_ids.shape[0], -1
                ).transpose(0,1)
                cf_student_patience = cf_student_patience.transpose(0, 1).contiguous().view(
                    n_layer, input_ids.shape[0], -1
                ).transpose(0,1)
            pt_loss = self.beta * patience_loss(
                teacher_patience, student_patience, 
                self.normalize_patience
            )
            cf_pt_loss = self.beta * patience_loss(
                cf_teacher_patience, cf_student_patience, 
                self.normalize_patience
            )
            loss = loss_dl + pt_loss + cf_pt_loss + loss_diito
        else:
            pt_loss = torch.tensor(0.0)
            cf_pt_loss = torch.tensor(0.0)
            loss = loss_dl + loss_diito
        if self.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        
        # last standard loss
        self.total_loss_epoch += loss.item()
        self.last_loss = loss.item()
        self.last_loss_dl = loss_dl.mean().item() if self.n_gpu > 0 else loss_dl.item()
        self.last_kd_loss = kd_loss.mean().item() if self.n_gpu > 0 else kd_loss.item()
        self.last_ce_loss = ce_loss.mean().item() if self.n_gpu > 0 else ce_loss.item()
        self.last_pt_loss = pt_loss.mean().item() if self.n_gpu > 0 else pt_loss.item()
        # last cf loss
        self.last_loss_diito = loss_diito.mean().item() if self.n_gpu > 0 else loss_diito.item()
        self.last_cf_pt_loss = cf_pt_loss.mean().item() if self.n_gpu > 0 else cf_pt_loss.item()
        
        # epoch standard loss
        n_sample = source_input_ids.shape[0]
        self.acc_tr_loss += self.last_loss * n_sample
        self.acc_tr_kd_loss += self.last_kd_loss * n_sample
        self.acc_tr_ce_loss += self.last_ce_loss * n_sample
        self.acc_tr_pt_loss = self.last_pt_loss * n_sample
        # epoch cf loss
        self.acc_tr_loss_diito += self.last_loss_diito * n_sample
        self.acc_tr_cf_pt_loss = self.last_cf_pt_loss * n_sample
        
        # pred acc
        pred_cls = logits_pred_student.data.max(1)[1]
        self.acc_tr_acc += pred_cls.eq(source_label_ids).sum().cpu().item()
        # cf pred acc
        cf_pred_cls_student = cf_logits_pred_student.data.max(1)[1]
        cf_pred_cls_teacher = cf_teacher_pred.data.max(1)[1]
        self.acc_tr_cf_acc += cf_pred_cls_student.eq(cf_pred_cls_teacher).sum().cpu().item()
        
        self.n_sequences_epoch += n_sample
        
        # standard acc metrics
        self.tr_loss = self.acc_tr_loss / self.n_sequences_epoch
        self.tr_kd_loss = self.acc_tr_kd_loss / self.n_sequences_epoch
        self.tr_ce_loss = self.acc_tr_ce_loss / self.n_sequences_epoch
        self.tr_pt_loss = self.acc_tr_pt_loss / self.n_sequences_epoch
        self.tr_acc = self.acc_tr_acc / self.n_sequences_epoch
        
        # cf acc metrics
        self.tr_loss_diito = self.acc_tr_loss_diito / self.n_sequences_epoch
        self.tr_cf_pt_loss = self.acc_tr_cf_pt_loss / self.n_sequences_epoch
        self.tr_cf_acc = self.acc_tr_cf_acc / self.n_sequences_epoch
              
        self.optimize(loss)
    
    def step_diito(
        self,
        source_input_ids,
        source_input_mask,
        source_segment_ids,
        source_label_ids,
        base_input_ids,
        base_input_mask,
        base_segment_ids,
        base_label_ids,
    ):
        source_intervention_mask, base_intervention_mask = self.prepare_interchange_mask(
            source_input_mask.sum(dim=-1), base_input_mask.sum(dim=-1),
            source_input_mask, base_input_mask,
        )
        student_invention_layer = random.choice(list(self.layer_mapping.keys()))
        teacher_invention_layer = self.layer_mapping[student_invention_layer]
        ##########
        # teacher
        with torch.no_grad():
            if self.alpha == 0:
                teacher_pred, teacher_patience = None, None
            else:
                _, teacher_base_outputs, teacher_counterfactual_outputs = \
                    self.teacher_encoder.forward(
                        source=[
                            source_input_ids, source_segment_ids, source_input_mask, 
                        ],
                        base=[
                            base_input_ids, base_segment_ids, base_input_mask,
                        ],
                        source_mask=source_intervention_mask, 
                        base_mask=base_intervention_mask,
                        coords=teacher_invention_layer,
                    )
                full_output_teacher, pooled_output_teacher = teacher_base_outputs
                cf_full_output_teacher, cf_pooled_output_teacher = teacher_counterfactual_outputs
                
                if self.kd_model.lower() in['kd', 'kd.cls']:
                    
                    teacher_pred = self.teacher_classifier(pooled_output_teacher)
                    cf_teacher_pred = self.teacher_classifier(cf_pooled_output_teacher)
                    
                    if self.kd_model.lower() == 'kd.cls':
                        
                        teacher_patience = torch.stack(full_output_teacher[:-1]).transpose(0, 1)
                        cf_teacher_patience = torch.stack(cf_full_output_teacher[:-1]).transpose(0, 1)
                        if self.fp16:
                            teacher_patience = teacher_patience.half()
                            cf_teacher_patience = cf_teacher_patience.half()
                        layer_index = [int(i) for i in self.fc_layer_idx.split(',')]
                        teacher_patience = torch.stack(
                            [torch.FloatTensor(teacher_patience[:,int(i)]) for i in layer_index]
                        ).transpose(0, 1)
                        cf_teacher_patience = torch.stack(
                            [torch.FloatTensor(cf_teacher_patience[:,int(i)]) for i in layer_index]
                        ).transpose(0, 1)
                        
                    else:
                        teacher_patience = None
                        cf_teacher_patience = None
                else:
                    raise ValueError(f'{self.kd_model} not implemented yet')
                if self.fp16:
                    teacher_pred = teacher_pred.half()
                    cf_teacher_pred = cf_teacher_pred.half()
        
        ##########
        # student
        _, student_base_outputs, student_counterfactual_outputs = \
            self.student_encoder.forward(
                source=[
                    source_input_ids, source_segment_ids, source_input_mask, 
                ],
                base=[
                    base_input_ids, base_segment_ids, base_input_mask,
                ],
                source_mask=source_intervention_mask, 
                base_mask=base_intervention_mask,
                coords=[student_invention_layer],
            )
        full_output_student, pooled_output_student = student_base_outputs
        cf_full_output_student, cf_pooled_output_student = student_counterfactual_outputs
        
        if self.kd_model.lower() in['kd', 'kd.cls']:
            logits_pred_student = self.student_classifier(
                pooled_output_student
            )
            cf_logits_pred_student = self.student_classifier(
                cf_pooled_output_student
            )
            if self.kd_model.lower() == 'kd.cls':
                student_patience = torch.stack(full_output_student[:-1]).transpose(0, 1)
                cf_student_patience = torch.stack(cf_full_output_student[:-1]).transpose(0, 1)
            else:
                student_patience = None
                cf_student_patience = None
        else:
            raise ValueError(f'{self.kd_model} not implemented yet')
        
        # calculate loss, along with counterfactual loss
        loss_dl, kd_loss, ce_loss = distillation_loss(
            logits_pred_student, source_label_ids, teacher_pred, T=self.T, alpha=self.alpha
        )
        loss_diito = diito_distillation_loss(
            logits_pred_student, teacher_pred, T=self.T, alpha=self.alpha
        )
        
        if self.beta > 0:
            if student_patience.shape[0] != input_ids.shape[0]:
                # For RACE
                n_layer = student_patience.shape[1]
                student_patience = student_patience.transpose(0, 1).contiguous().view(
                    n_layer, input_ids.shape[0], -1
                ).transpose(0,1)
                cf_student_patience = cf_student_patience.transpose(0, 1).contiguous().view(
                    n_layer, input_ids.shape[0], -1
                ).transpose(0,1)
            pt_loss = self.beta * patience_loss(
                teacher_patience, student_patience, 
                self.normalize_patience
            )
            cf_pt_loss = self.beta * patience_loss(
                cf_teacher_patience, cf_student_patience, 
                self.normalize_patience
            )
            loss = loss_dl + pt_loss + cf_pt_loss + loss_diito
        else:
            pt_loss = torch.tensor(0.0)
            cf_pt_loss = torch.tensor(0.0)
            loss = loss_dl + loss_diito
        if self.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        
        # last standard loss
        self.total_loss_epoch += loss.item()
        self.last_loss = loss.item()
        self.last_loss_dl = loss_dl.mean().item() if self.n_gpu > 0 else loss_dl.item()
        self.last_kd_loss = kd_loss.mean().item() if self.n_gpu > 0 else kd_loss.item()
        self.last_ce_loss = ce_loss.mean().item() if self.n_gpu > 0 else ce_loss.item()
        self.last_pt_loss = pt_loss.mean().item() if self.n_gpu > 0 else pt_loss.item()
        # last cf loss
        self.last_loss_diito = loss_diito.mean().item() if self.n_gpu > 0 else loss_diito.item()
        self.last_cf_pt_loss = cf_pt_loss.mean().item() if self.n_gpu > 0 else cf_pt_loss.item()
        
        # epoch standard loss
        n_sample = source_input_ids.shape[0]
        self.acc_tr_loss += self.last_loss * n_sample
        self.acc_tr_kd_loss += self.last_kd_loss * n_sample
        self.acc_tr_ce_loss += self.last_ce_loss * n_sample
        self.acc_tr_pt_loss = self.last_pt_loss * n_sample
        # epoch cf loss
        self.acc_tr_loss_diito += self.last_loss_diito * n_sample
        self.acc_tr_cf_pt_loss = self.last_cf_pt_loss * n_sample
        
        # pred acc
        pred_cls = logits_pred_student.data.max(1)[1]
        self.acc_tr_acc += pred_cls.eq(source_label_ids).sum().cpu().item()
        # cf pred acc
        cf_pred_cls_student = cf_logits_pred_student.data.max(1)[1]
        cf_pred_cls_teacher = cf_teacher_pred.data.max(1)[1]
        self.acc_tr_cf_acc += cf_pred_cls_student.eq(cf_pred_cls_teacher).sum().cpu().item()
        
        self.n_sequences_epoch += n_sample
        
        # standard acc metrics
        self.tr_loss = self.acc_tr_loss / self.n_sequences_epoch
        self.tr_kd_loss = self.acc_tr_kd_loss / self.n_sequences_epoch
        self.tr_ce_loss = self.acc_tr_ce_loss / self.n_sequences_epoch
        self.tr_pt_loss = self.acc_tr_pt_loss / self.n_sequences_epoch
        self.tr_acc = self.acc_tr_acc / self.n_sequences_epoch
        
        # cf acc metrics
        self.tr_loss_diito = self.acc_tr_loss_diito / self.n_sequences_epoch
        self.tr_cf_pt_loss = self.acc_tr_cf_pt_loss / self.n_sequences_epoch
        self.tr_cf_acc = self.acc_tr_cf_acc / self.n_sequences_epoch
              
        self.optimize(loss)
    
    def step(
        self,
        input_ids,
        input_mask,
        segment_ids,
        label_ids,
        skip_update_iter=False,
    ):
        # teacher no_grad() forward pass.
        with torch.no_grad():
            if self.alpha == 0:
                teacher_pred, teacher_patience = None, None
            else:
                # define a new function to compute loss values for both output_modes
                (full_output_teacher, pooled_output_teacher), _ = self.teacher_encoder(
                    input_ids, segment_ids, input_mask
                )
                if self.kd_model.lower() in['kd', 'kd.cls']:
                    teacher_pred = self.teacher_classifier(pooled_output_teacher)
                    if self.kd_model.lower() == 'kd.cls':
                        teacher_patience = torch.stack(full_output_teacher[:-1]).transpose(0, 1)
                        if self.fp16:
                            teacher_patience = teacher_patience.half()
                        layer_index = [int(i) for i in self.fc_layer_idx.split(',')]
                        teacher_patience = torch.stack(
                            [torch.FloatTensor(teacher_patience[:,int(i)]) for i in layer_index]
                        ).transpose(0, 1)
                    else:
                        teacher_patience = None
                else:
                    raise ValueError(f'{self.kd_model} not implemented yet')
                if self.fp16:
                    teacher_pred = teacher_pred.half()
            
        # student with_grad() forward pass.
        (full_output_student, pooled_output_student), _ = self.student_encoder(
            input_ids, segment_ids, input_mask
        )
        if self.kd_model.lower() in['kd', 'kd.cls']:
            logits_pred_student = self.student_classifier(
                pooled_output_student
            )
            if self.kd_model.lower() == 'kd.cls':
                student_patience = torch.stack(full_output_student[:-1]).transpose(0, 1)
            else:
                student_patience = None
        else:
            raise ValueError(f'{self.kd_model} not implemented yet')

        # calculate loss
        loss_dl, kd_loss, ce_loss = distillation_loss(
            logits_pred_student, label_ids, teacher_pred, T=self.T, alpha=self.alpha
        )
        if self.beta > 0:
            if student_patience.shape[0] != input_ids.shape[0]:
                # For RACE
                n_layer = student_patience.shape[1]
                student_patience = student_patience.transpose(0, 1).contiguous().view(
                    n_layer, input_ids.shape[0], -1
                ).transpose(0,1)
            pt_loss = self.beta * patience_loss(
                teacher_patience, student_patience, 
                self.normalize_patience
            )
            loss = loss_dl + pt_loss
        else:
            pt_loss = torch.tensor(0.0)
            loss = loss_dl
        if self.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        
        # bookkeeping
        self.total_loss_epoch += loss.item()
        self.last_loss = loss.item()
        self.last_loss_dl = loss_dl.mean().item() if self.n_gpu > 0 else loss_dl.item()
        self.last_kd_loss = kd_loss.mean().item() if self.n_gpu > 0 else kd_loss.item()
        self.last_ce_loss = ce_loss.mean().item() if self.n_gpu > 0 else ce_loss.item()
        self.last_pt_loss = pt_loss.mean().item() if self.n_gpu > 0 else pt_loss.item()
        
        n_sample = input_ids.shape[0]
        self.acc_tr_loss += self.last_loss * n_sample
        self.acc_tr_kd_loss += self.last_kd_loss * n_sample
        self.acc_tr_ce_loss += self.last_ce_loss * n_sample
        self.acc_tr_pt_loss = self.last_pt_loss * n_sample
        pred_cls = logits_pred_student.data.max(1)[1]
        self.acc_tr_acc += pred_cls.eq(label_ids).sum().cpu().item()
        self.n_sequences_epoch += n_sample
        
        self.tr_loss = self.acc_tr_loss / self.n_sequences_epoch
        self.tr_kd_loss = self.acc_tr_kd_loss / self.n_sequences_epoch
        self.tr_ce_loss = self.acc_tr_ce_loss / self.n_sequences_epoch
        self.tr_pt_loss = self.acc_tr_pt_loss / self.n_sequences_epoch
        self.tr_acc = self.acc_tr_acc / self.n_sequences_epoch
              
        self.optimize(loss, skip_update_iter=skip_update_iter)
            
    def optimize(self, loss, skip_update_iter=False):
        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps
        
        # backward()
        if self.fp16:
            self.optimizer.backward(loss)
        else:
            loss.backward()
        
        if not skip_update_iter:
            self.iter()

        if self.n_iter % self.gradient_accumulation_steps == 0:
            if self.fp16:
                self.lr_this_step = self.learning_rate * warmup_linear(
                    self.n_total_iter / self.num_train_optimization_steps,
                    self.warmup_proportion
                )
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr_this_step
            else:
                self.lr_this_step = self.optimizer.get_lr()
                    
            self.optimizer.step()
            self.optimizer.zero_grad()
    
    def iter(self):
        """
        Update global counts, write to tensorboard and save checkpoint.
        """
        
        self.n_iter += 1
        self.n_total_iter += 1
        if self.n_total_iter % self.params.checkpoint_interval == 0:
            pass
            # you can uncomment this line, if you really have checkpoints.
            # self.save_checkpoint()
        
        """
        Logging is not affected by the flag skip_update_iter.
        We want to log crossway effects, and losses should be
        in the same magnitude.
        """
        if self.n_total_iter % self.params.log_interval == 0:
            self.log_tensorboard()
            self.last_log = time.time()

    def log_tensorboard(self):
        """
        Log into tensorboard. Only by the master process.
        """

        log_train = open(os.path.join(self.output_dir, 'train_log.txt'), 'a', buffering=1)
        print('{},{},{},{},{},{},{},{}'.format(
                self.epoch+1, self.n_total_iter, self.n_iter, 
                self.tr_acc,
                self.tr_loss, 
                self.tr_kd_loss,
                self.tr_ce_loss, 
                self.tr_pt_loss
            ),
            file=log_train
        )
        log_train.close()
        
        if not self.is_wandb:
            pass # log to local logging file?
        else:    
            wandb.log(
                {
                    "train/loss": self.last_loss, 
                    "train/loss_dl": self.last_loss_dl, 
                    "train/kd_loss": self.last_kd_loss, 
                    "train/ce_loss": self.last_ce_loss, 
                    "train/pt_loss": self.last_pt_loss, 
                    
                    "train/epoch_loss": self.tr_loss, 
                    "train/epoch_kd_loss": self.tr_kd_loss, 
                    "train/epoch_ce_loss": self.tr_ce_loss, 
                    "train/epoch_pt_loss": self.tr_pt_loss, 
                    "train/epoch_tr_acc": self.tr_acc, 
                    
                    "train/epoch_loss_diito": self.tr_loss_diito, 
                    "train/epoch_cf_pt_loss": self.tr_cf_pt_loss, 
                    "train/epoch_tr_cf_loss": self.tr_cf_acc, 
                }, 
                step=self.n_total_iter
            )

            wandb.log(
                {
                    "train/learning_rate": self.lr_this_step,
                    "train/speed": time.time() - self.last_log,
                }, 
                step=self.n_total_iter
            )

    def end_epoch(self):
        """
        Finally arrived at the end of epoch (full pass on dataset).
        Do some tensorboard logging and checkpoint saving.
        """
        logger.info(f"{self.n_sequences_epoch} sequences have been trained during this epoch.")

        # let us do evaluation on the eval just for bookkeeping.
        # make sure this is not intervening your training in anyway
        # otherwise, data is leaking!
        if self.is_diito:
            if 'race' in self.task_name:
                result = eval_model_dataloader(
                    self.student_encoder.model, self.student_classifier, 
                    self.eval_dataset, self.device, False
                )
            else:
                result = eval_model_dataloader_nli(
                    self.task_name.lower(), self.eval_label_ids, 
                    self.student_encoder.model, self.student_classifier, self.eval_dataset,
                    self.kd_model, self.num_labels, self.device, 
                    self.weights, self.fc_layer_idx, self.output_mode
                )
        else:
            if 'race' in self.task_name:
                result = eval_model_dataloader(
                    self.student_encoder, self.student_classifier, 
                    self.eval_dataset, self.device, False
                )
            else:
                result = eval_model_dataloader_nli(
                    self.task_name.lower(), self.eval_label_ids, 
                    self.student_encoder, self.student_classifier, self.eval_dataset,
                    self.kd_model, self.num_labels, self.device, 
                    self.weights, self.fc_layer_idx, self.output_mode
                )
        log_eval = open(os.path.join(self.output_dir, 'eval_log.txt'), 'a', buffering=1)
        if self.task_name in ['CoLA']:
            print('{},{},{}'.format(self.epoch+1, result['mcc'], result['eval_loss']), file=log_eval)
        else:
            if 'race' in self.task_name:
                print('{},{},{}'.format(self.epoch+1, result['acc'], result['loss']), file=log_eval)
            else:
                print('{},{},{}'.format(self.epoch+1, result['acc'], result['eval_loss']), file=log_eval)
        log_eval.close()
        
        self.save_checkpoint()
        if self.is_wandb:
            wandb.log(
                {
                    "epoch/loss": self.total_loss_epoch / self.n_iter, 
                    'epoch': self.epoch
                }
            )
            if self.task_name in ['CoLA']:
                wandb.log(
                    {
                        "epoch/eval_mcc": result['mcc'], 
                        "epoch/eval_loss": result['eval_loss'], 
                        'epoch': self.epoch
                    }
                )
            else:
                if 'race' in self.task_name:
                    wandb.log(
                        {
                            "epoch/eval_acc": result['acc'], 
                            "epoch/eval_loss": result['loss'], 
                            'epoch': self.epoch
                        }
                    )
                else:
                    wandb.log(
                        {
                            "epoch/eval_acc": result['acc'], 
                            "epoch/eval_loss": result['eval_loss'], 
                            'epoch': self.epoch
                        }
                    ) 

        self.epoch += 1
        self.n_sequences_epoch = 0
        self.n_iter = 0
        self.total_loss_epoch = 0
        
        self.acc_tr_loss = 0
        self.acc_tr_kd_loss = 0
        self.acc_tr_ce_loss = 0
        self.acc_tr_acc = 0
    
        self.acc_tr_loss_diito = 0
        self.acc_tr_cf_pt_loss = 0
        self.acc_tr_cf_acc = 0
    
    def save_checkpoint(self, checkpoint_name=None):
        if checkpoint_name == None:
            if self.n_gpu > 1:
                torch.save(
                    self.student_encoder.module.state_dict(), 
                    os.path.join(self.output_dir, self.output_model_file + f'_e.{self.epoch}.encoder.pkl')
                )
                torch.save(
                    self.student_classifier.module.state_dict(), 
                    os.path.join(self.output_dir, self.output_model_file + f'_e.{self.epoch}.cls.pkl')
                )
            else:
                torch.save(
                    self.student_encoder.state_dict(), 
                    os.path.join(self.output_dir, self.output_model_file + f'_e.{self.epoch}.encoder.pkl')
                )
                torch.save(
                    self.student_classifier.state_dict(), 
                    os.path.join(self.output_dir, self.output_model_file + f'_e.{self.epoch}.cls.pkl')
                )
        else:
            if self.n_gpu > 1:
                torch.save(
                    self.student_encoder.module.state_dict(), 
                    os.path.join(self.output_dir, "encoder."+checkpoint_name)
                )
                torch.save(
                    self.student_classifier.module.state_dict(), 
                    os.path.join(self.output_dir, "cls."+checkpoint_name)
                )
            else:
                torch.save(
                    self.student_encoder.state_dict(), 
                    os.path.join(self.output_dir, "encoder."+checkpoint_name)
                )
                torch.save(
                    self.student_classifier.state_dict(), 
                    os.path.join(self.output_dir, "cls."+checkpoint_name)
                )
