# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team and Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" The distiller to distil the student.
    Adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM)
"""
import math
import os
import time

import psutil
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from grouped_batch_sampler import GroupedBatchSampler, create_lengths_groups
from lm_seqs_dataset import LmSeqsDataset
from transformers import get_linear_schedule_with_warmup
from utils import logger

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

import argparse
import json
import os
import pickle
import shutil
import random

import numpy as np
import torch

from distiller import Distiller
from lm_seqs_dataset import LmSeqsDataset
from transformers import (
    AutoConfig,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    DistilBertConfig,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    AutoTokenizer,
    AutoModelForMaskedLM,
)
from utils import git_log, init_gpu_params, logger, set_seed
from datasets import load_dataset
from counterfactual_utils import *
import wandb
from models.modeling_distilbert import DistilBertForMaskedLM

# Examples of interchange.
# activations_counterfactual_teacher = get_activation_at(
#     teacher_bert,
#     batch["input_ids"],
#     batch["attention_mask"],
#     variable_names=["$L:1$H:1$[0:32]"]
# )
# interchange_with_activation_at(
#     teacher_bert,
#     batch["input_ids"],
#     batch["attention_mask"],
#     interchanged_variables=[torch.zeros(32, 512, 32)],
#     variable_names=["$L:1$H:1$[0:32]"]
# )

class CausalXXSDistiller:
    def __init__(
        self, params: dict, token_probs: torch.tensor, student: nn.Module, teacher: nn.Module
    ):
        pass
    
    
    
    
    
    
    