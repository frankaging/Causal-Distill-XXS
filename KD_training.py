#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import logging
import os
import random
import pickle

import numpy as np
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from tqdm import tqdm, trange

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


# In[ ]:


#########################################################################
# Prepare Parser
##########################################################################
parser = default_parser()
DEBUG = True
if DEBUG:
    logger.info("IN DEBUG MODE")
    # run simple fune-tuning *teacher* by uncommenting below cmd
    # argv = get_predefine_argv('glue', 'RTE', 'finetune_teacher')

    # run simple fune-tuning *student* by uncommenting below cmd
    # argv = get_predefine_argv('glue', 'RTE', 'finetune_student')

    # run vanilla KD by uncommenting below cmd
    # argv = get_predefine_argv('glue', 'RTE', 'kd')

    # run Patient Teacher by uncommenting below cmd
    argv = get_predefine_argv('glue', 'SST-2', 'kd.cls')
    try:
        args = parser.parse_args(argv)
    except NameError:
        raise ValueError('please uncomment one of option above to start training')
else:
    logger.info("IN CMD MODE")
    args = parser.parse_args()
args = complete_argument(args, is_debug=DEBUG)


# In[ ]:


args.raw_data_dir = os.path.join(HOME_DATA_FOLDER, 'data_raw', args.task_name)
args.feat_data_dir = os.path.join(HOME_DATA_FOLDER, 'data_feat', args.task_name)

args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
logger.info('actual batch size on all GPU = %d' % args.train_batch_size)
device, n_gpu = args.device, args.n_gpu

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

logger.info('Input Argument Information')
args_dict = vars(args)
for a in args_dict:
    logger.info('%-28s  %s' % (a, args_dict[a]))

#########################################################################
# Prepare  Data
##########################################################################
task_name = args.task_name.lower()

if task_name not in processors and 'race' not in task_name:
    raise ValueError("Task not found: %s" % (task_name))

if 'race' in task_name:
    pass
else:
    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list)

tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)


# In[ ]:


if args.do_train:
    train_sampler = SequentialSampler if DEBUG else RandomSampler
    read_set = 'train'
    logger.info('skipping loading teacher\'s predictoin, we calculate this on-the-fly')
    train_examples, train_dataloader, _ = get_task_dataloader(task_name, read_set, tokenizer, args, SequentialSampler,
                                                              batch_size=args.train_batch_size)
    num_train_optimization_steps = int(len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    # Run prediction for full data
    eval_examples, eval_dataloader, eval_label_ids = get_task_dataloader(task_name, 'dev', tokenizer, args, SequentialSampler, batch_size=args.eval_batch_size)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)


# In[ ]:


#########################################################################
# Prepare model
#########################################################################
student_config = BertConfig(os.path.join(args.bert_model, 'bert_config.json'))
teacher_config = BertConfig(os.path.join(args.bert_model, 'bert_config.json'))
if args.kd_model.lower() in ['kd', 'kd.cls']:
    logger.info('using normal Knowledge Distillation')
    output_all_layers = args.kd_model.lower() == 'kd.cls'
    logger.info('*' * 77)
    logger.info("Loading the student model...")
    logger.info('*' * 77)
    student_encoder, student_classifier = init_model(
        task_name, output_all_layers, 
        args.student_hidden_layers, student_config,
    )

    n_student_layer = len(student_encoder.bert.encoder.layer)
    student_encoder = load_model(
        student_encoder, args.encoder_checkpoint_student, args, 'student', 
        verbose=True, DEBUG=False,
    )
    logger.info('*' * 77)
    student_classifier = load_model(
        student_classifier, args.cls_checkpoint_student, args, 'classifier', 
        verbose=True, DEBUG=False,
    )
    
    logger.info('*' * 77)
    logger.info("Loading the teacher model...")
    logger.info('*' * 77)
    # since we also calculate teacher's output on-fly, we need to load the teacher model as well.
    # note that, we assume teacher model is pre-trained already.
    teacher_encoder, teacher_classifier = init_model(
        task_name, output_all_layers, 
        teacher_config.num_hidden_layers, teacher_config,
    )
    
    n_teacher_layer = len(teacher_encoder.bert.encoder.layer)
    teacher_encoder = load_model(
        teacher_encoder, args.encoder_checkpoint_teacher, args, 'student', 
        verbose=True, DEBUG=False,
    )
    logger.info('*' * 77)
    teacher_classifier = load_model(
        teacher_classifier, args.cls_checkpoint_teacher, args, 'classifier', 
        verbose=True, DEBUG=False,
    )

else:
    # originally, the codebase supports kd.full, but that is never used.
    raise ValueError('%s KD not found, please use kd or kd.cls' % args.kd)

n_param_student = count_parameters(student_encoder) + count_parameters(student_classifier)
logger.info('number of layers in student model = %d' % n_student_layer)
logger.info('num parameters in student model are %d and %d' % (count_parameters(student_encoder), count_parameters(student_classifier)))


# In[ ]:


#########################################################################
# Prepare optimizer
#########################################################################
if args.do_train:
    param_optimizer = list(student_encoder.named_parameters()) + list(student_classifier.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        logger.info('FP16 activate, use apex FusedAdam')
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        logger.info('FP16 is not activated, use BertAdam')
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)


# In[ ]:


#########################################################################
# Model Training
#########################################################################
output_model_file = '{}_nlayer.{}_lr.{}_T.{}.alpha.{}_beta.{}_bs.{}'.format(args.task_name, args.student_hidden_layers,
                                                                            args.learning_rate,
                                                                            args.T, args.alpha, args.beta,
                                                                            args.train_batch_size * args.gradient_accumulation_steps)
if args.do_train:
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    student_encoder.train()
    student_classifier.train()
    teacher_encoder.eval()
    teacher_classifier.eval()
    
    log_train = open(os.path.join(args.output_dir, 'train_log.txt'), 'w', buffering=1)
    log_eval = open(os.path.join(args.output_dir, 'eval_log.txt'), 'w', buffering=1)
    print('epoch,global_steps,step,acc,loss,kd_loss,ce_loss,AT_loss', file=log_train)
    print('epoch,acc,loss', file=log_eval)
    
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss, tr_ce_loss, tr_kd_loss, tr_acc = 0, 0, 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            # teascher patient is on-the-fly, we can skip the logic for different batch format.
            input_ids, input_mask, segment_ids, label_ids = batch
            teacher_pred, teacher_patience = None, None
            
            # teacher no_grad() forward pass.
            with torch.no_grad():
                if args.alpha == 0:
                    teacher_pred, teacher_patience = None, None
                else:
                    # define a new function to compute loss values for both output_modes
                    full_output_teacher, pooled_output_teacher = teacher_encoder(
                        input_ids, segment_ids, input_mask
                    )
                    if args.kd_model.lower() in['kd', 'kd.cls']:
                        teacher_pred = teacher_classifier(pooled_output_teacher)
                        if args.kd_model.lower() == 'kd.cls':
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
            full_output_student, pooled_output_student = student_encoder(input_ids, segment_ids, input_mask)
            if args.kd_model.lower() in['kd', 'kd.cls']:
                logits_pred_student = student_classifier(pooled_output_student)
                if args.kd_model.lower() == 'kd.cls':
                    student_patience = torch.stack(full_output_student[:-1]).transpose(0, 1)
                else:
                    student_patience = None
            else:
                raise ValueError(f'{args.kd_model} not implemented yet')
            
            # only extracting those interested layers.
            layer_index = [int(i) for i in args.fc_layer_idx.split(',')]
            teacher_patience = torch.stack([torch.FloatTensor(teacher_patience[:,int(i)]) for i in layer_index]).transpose(0, 1)
            
            # calculate loss
            loss_dl, kd_loss, ce_loss = distillation_loss(logits_pred_student, label_ids, teacher_pred, T=args.T, alpha=args.alpha)
            if args.beta > 0:
                if student_patience.shape[0] != input_ids.shape[0]:
                    # For RACE
                    n_layer = student_patience.shape[1]
                    student_patience = student_patience.transpose(0, 1).contiguous().view(n_layer, input_ids.shape[0], -1).transpose(0,1)
                pt_loss = args.beta * patience_loss(teacher_patience, student_patience, args.normalize_patience)
                loss = loss_dl + pt_loss
            else:
                pt_loss = torch.tensor(0.0)
                loss = loss_dl
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            
            # backward()
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            
            # bookkeepings
            n_sample = input_ids.shape[0]
            tr_loss += loss.item() * n_sample
            if isinstance(kd_loss, float):
                tr_kd_loss += kd_loss * n_sample
            else:
                tr_kd_loss += kd_loss.item() * n_sample
            tr_ce_loss += ce_loss.item() * n_sample
            tr_loss_pt = pt_loss.item() * n_sample

            pred_cls = logits_pred_student.data.max(1)[1]
            tr_acc += pred_cls.eq(label_ids).sum().cpu().item()
            nb_tr_examples += n_sample
            nb_tr_steps += 1

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                      args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if global_step % args.log_every_step == 0:
                print('{},{},{},{},{},{},{},{}'.format(epoch+1, global_step, step, tr_acc / nb_tr_examples,
                                                       tr_loss / nb_tr_examples, tr_kd_loss / nb_tr_examples,
                                                       tr_ce_loss / nb_tr_examples, tr_loss_pt / nb_tr_examples),
                      file=log_train)
            
            
            
            
    


# In[ ]:




