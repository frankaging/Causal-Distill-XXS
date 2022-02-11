#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

from distiller import TaskSpecificDistiller
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
DEBUG = False
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
    args.num_train_optimization_steps = num_train_optimization_steps

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


distiller = TaskSpecificDistiller(
    args, train_dataloader, 
    student_encoder, student_classifier,
    teacher_encoder, teacher_classifier,
)


# In[ ]:


logger.info("Hey Zen: Let's go get some drinks.")
distiller.train()


# In[ ]:


if args.do_eval:
    # Save a trained model and the associated configuration
    if 'race' in task_name:
        result = eval_model_dataloader(student_encoder, student_classifier, eval_dataloader, device, False)
    else:
        result = eval_model_dataloader_nli(args.task_name.lower(), eval_label_ids, student_encoder, student_classifier, eval_dataloader,
                                           args.kd_model, num_labels, device, args.weights, args.fc_layer_idx, output_mode)

    output_test_file = os.path.join(args.output_dir, "eval_results_" + output_model_file + '.txt')
    with open(output_test_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key]))) 
            

