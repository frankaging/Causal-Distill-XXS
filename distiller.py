import logging
import os
import random
import pickle
import time
import psutil

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

from src.argument_parser import default_parser, get_predefine_argv, complete_argument
from src.nli_data_processing import processors, output_modes
from src.data_processing import init_model, get_task_dataloader
from src.modeling import BertForSequenceClassificationEncoder, FCClassifierForSequenceClassification, FullFCClassifierForSequenceClassification
from src.utils import load_model, count_parameters, eval_model_dataloader_nli, eval_model_dataloader
from src.KD_loss import distillation_loss, patience_loss
from envs import HOME_DATA_FOLDER

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskSpecificDistiller:
    def __init__(
        self, params, dataset,
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
        
        self.output_model_file = '{}_nlayer.{}_lr.{}_T.{}.alpha.{}_beta.{}_bs.{}'.format(
            self.params.task_name, 
            self.params.student_hidden_layers,
            self.params.learning_rate,
            self.params.T, 
            self.params.alpha, 
            self.params.beta,
            self.params.train_batch_size * self.params.gradient_accumulation_steps
        )
        
        self.dataset = dataset

        self.student_encoder = student_encoder
        self.student_classifier = student_classifier
        self.teacher_encoder = teacher_encoder
        self.teacher_classifier = teacher_classifier
        self.student_config = student_encoder.config
        self.teacher_config = teacher_encoder.config
        self.vocab_size = student_encoder.config.vocab_size
        
        # common used vars
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
        
        # log to a local file
        log_train = open(os.path.join(self.output_dir, 'train_log.txt'), 'w', buffering=1)
        log_eval = open(os.path.join(self.output_dir, 'eval_log.txt'), 'w', buffering=1)
        print('epoch,global_steps,step,acc,loss,kd_loss,ce_loss,AT_loss', file=log_train)
        print('epoch,acc,loss', file=log_eval)
        log_train.close()
        log_eval.close()
    
        param_optimizer = list(
            self.student_encoder.named_parameters()
        ) + list(
            self.student_classifier.named_parameters()
        )
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
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                self.optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                self.optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
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
        
        self.acc_tr_loss = 0
        self.acc_tr_kd_loss = 0
        self.acc_tr_ce_loss = 0
        self.acc_tr_pt_loss = 0
        self.acc_tr_acc = 0
        
        self.tr_loss = 0
        self.tr_kd_loss = 0
        self.tr_ce_loss = 0
        self.tr_pt_loss = 0
        self.tr_acc = 0
    
    def prepare_batch(self, input_ids, input_mask, segment_ids, label_ids, is_DIITO=False):
        if is_DIITO:
            pass
        else:
            return input_ids, input_mask, segment_ids, label_ids
    
    def train(self):
        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0
        self.student_encoder.train()
        self.student_classifier.train()
        self.teacher_encoder.eval()
        self.teacher_classifier.eval()
        self.last_log = time.time()
        
        for epoch in trange(int(self.num_train_epochs), desc="Epoch"):
            tr_loss, tr_ce_loss, tr_kd_loss, tr_acc = 0, 0, 0, 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            iter_bar = tqdm(self.dataset, desc="-Iter", disable=False)
            for batch in iter_bar:
                batch = tuple(t.to(self.device) for t in batch)
                # teascher patient is on-the-fly, we can skip the logic for different batch format.
                input_ids, input_mask, segment_ids, label_ids = self.prepare_batch(
                    *batch
                )
                self.step(
                    input_ids,
                    input_mask,
                    segment_ids,
                    label_ids,
                )
                iter_bar.update()
                iter_bar.set_postfix(
                    {
                        "Last_loss": f"{self.last_loss:.2f}", 
                        "Avg_cum_loss": f"{self.total_loss_epoch/self.n_iter:.2f}", 
                    }
                )
            iter_bar.close()

            logger.info(f"--- Ending epoch {self.epoch}/{self.params.n_epoch-1}")
            self.end_epoch()

        logger.info("Save very last checkpoint as `pytorch_model.bin`.")
        self.save_checkpoint(checkpoint_name="pytorch_model.bin")
        logger.info("Training is finished")
        
    def step(
        self,
        input_ids,
        input_mask,
        segment_ids,
        label_ids,
    ):
        # teacher no_grad() forward pass.
        with torch.no_grad():
            if self.alpha == 0:
                teacher_pred, teacher_patience = None, None
            else:
                # define a new function to compute loss values for both output_modes
                full_output_teacher, pooled_output_teacher = self.teacher_encoder(
                    input_ids, segment_ids, input_mask
                )
                if self.kd_model.lower() in['kd', 'kd.cls']:
                    teacher_pred = self.teacher_classifier(pooled_output_teacher)
                    if self.kd_model.lower() == 'kd.cls':
                        teacher_patience = torch.stack(full_output_teacher[:-1]).transpose(0, 1)
                        if args.fp16:
                            teacher_patience = teacher_patience.half()
                    else:
                        teacher_patience = None
                else:
                    raise ValueError(f'{args.kd_model} not implemented yet')
                if args.fp16:
                    teacher_pred = teacher_pred.half()
            
        # student with_grad() forward pass.
        full_output_student, pooled_output_student = self.student_encoder(
            input_ids, segment_ids, input_mask
        )
        if self.kd_model.lower() in['kd', 'kd.cls']:
            logits_pred_student = self.student_classifier(
                pooled_output_student
            )
            if args.kd_model.lower() == 'kd.cls':
                student_patience = torch.stack(full_output_student[:-1]).transpose(0, 1)
            else:
                student_patience = None
        else:
            raise ValueError(f'{self.kd_model} not implemented yet')

        # only extracting those interested layers.
        layer_index = [int(i) for i in self.fc_layer_idx.split(',')]
        teacher_patience = torch.stack(
            [torch.FloatTensor(teacher_patience[:,int(i)]) for i in layer_index]
        ).transpose(0, 1)

        # calculate loss
        loss_dl, kd_loss, ce_loss = distillation_loss(
            logits_pred_student, label_ids, teacher_pred, T=self.T, alpha=self.alpha
        )
        if args.beta > 0:
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
        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        
        # bookkeeping?
        self.last_loss_dl = 0
        self.last_kd_loss = 0
        self.last_ce_loss = 0 
        self.last_pt_loss = 0
        
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
        self.acc_tr_pt_loss += self.last_pt_loss * n_sample
        pred_cls = logits_pred_student.data.max(1)[1]
        self.acc_tr_acc += pred_cls.eq(label_ids).sum().cpu().item()
        self.n_sequences_epoch += n_sample
        
        self.tr_loss = self.acc_tr_loss / self.n_sequences_epoch
        self.tr_kd_loss = self.acc_tr_kd_loss / self.n_sequences_epoch
        self.tr_ce_loss = self.acc_tr_ce_loss / self.n_sequences_epoch
        self.tr_pt_loss = self.acc_tr_pt_loss / self.n_sequences_epoch
        self.tr_acc = self.acc_tr_acc / self.n_sequences_epoch
              
        self.optimize(loss)

            
    def optimize(self, loss):
        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps
        
        # backward()
        if self.fp16:
            self.optimizer.backward(loss)
        else:
            loss.backward()

        self.iter()

        if self.n_iter % self.gradient_accumulation_steps == 0:
            if self.fp16:
                self.lr_this_step = self.learning_rate * warmup_linear(
                    self.n_total_iter / self.num_train_optimization_steps,
                    self.warmup_proportion
                )
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr_this_step
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
        if not self.is_master:
            return
        
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
                    
                    "train/acc_loss": self.tr_loss, 
                    "train/acc_kd_loss": self.tr_kd_loss, 
                    "train/acc_ce_loss": self.tr_ce_loss, 
                    "train/acc_pt_loss": self.tr_pt_loss, 
                    "train/acc_tr_acc": self.tr_acc, 
                }, 
                step=self.n_total_iter
            )

            wandb.log(
                {
                    "train/learning_rate": self.lr_this_step,
                    "train/memory_usage": psutil.virtual_memory()._asdict()["used"] / 1_000_000,
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

        self.save_checkpoint()
        if self.is_wandb:
            wandb.log(
                {
                    "epoch/loss": self.total_loss_epoch / self.n_iter, 
                    'epoch': self.epoch
                }
            )

        self.epoch += 1
        self.n_sequences_epoch = 0
        self.n_iter = 0
        self.total_loss_epoch = 0
    
    def save_checkpoint(self, checkpoint_name=None):
        if checkpoint_name == None:
            if args.n_gpu > 1:
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
            if args.n_gpu > 1:
                torch.save(
                    self.student_encoder.module.state_dict(), 
                    os.path.join(self.output_dir, checkpoint_name)
                )
                torch.save(
                    self.student_classifier.module.state_dict(), 
                    os.path.join(self.output_dir, checkpoint_name)
                )
            else:
                torch.save(
                    self.student_encoder.state_dict(), 
                    os.path.join(self.output_dir, checkpoint_name)
                )
                torch.save(
                    self.student_classifier.state_dict(), 
                    os.path.join(self.output_dir, checkpoint_name)
                )
        