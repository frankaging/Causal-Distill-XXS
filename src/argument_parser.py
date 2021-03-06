import argparse
import logging
import os
import random
import torch

from envs import HOME_DATA_FOLDER

logger = logging.getLogger(__name__)


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def is_folder_empty(folder_name):
    if len([f for f in os.listdir(folder_name) if not f.startswith('.')]) == 0:
        return True
    else:
        return False


def default_parser():
    parser = argparse.ArgumentParser()

    # Input Training tasks
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        help="The name of the task for training.")
    parser.add_argument("--run_name",
                        default=None,
                        type=str,
                        help="The name of the task for training.")

    # System related parameters
    parser.add_argument("--output_dir",
                        default=os.path.join(HOME_DATA_FOLDER, 'outputs'),
                        type=str,
                        help="The output directory where the model checkpoints will be written.")

    parser.add_argument("--log_every_step",
                        default=1,
                        type=int,
                        help="output to log every global x training steps, default is 1")
    
    parser.add_argument("--log_interval", type=int, default=500, help="Tensorboard logging interval.")
    parser.add_argument("--checkpoint_interval", type=int, default=4000, help="Checkpoint interval.")

    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--max_training_examples",
                        default=None,
                        type=int,
                        help="Total number of training examples to use.")
    
    # Training related parameters
    parser.add_argument('--seed',
                        type=int,
                        default=None,
                        help="random seed for initialization")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=4.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action="store_true",
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--student_hidden_layers',
                        type=int,
                        default=None,
                        help="number of transformer layers for student, default is None (use all layers)")
    parser.add_argument('--teacher_prediction',
                        type=str,
                        default=None,
                        help="teacher prediction file to guild the student's output")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    # Distillation related parameters
    parser.add_argument("--bert_model",
                        default=None,
                        type=str,
                        help="student bert model configuration folder")

    parser.add_argument("--encoder_checkpoint_student",
                        default=None,
                        type=str,
                        help="check point for student encoder")
    parser.add_argument("--cls_checkpoint_student",
                        default=None,
                        type=str,
                        help="check point for student classifier")
    parser.add_argument("--encoder_checkpoint_teacher",
                        default=None,
                        type=str,
                        help="check point for student encoder")
    parser.add_argument("--cls_checkpoint_teacher",
                        default=None,
                        type=str,
                        help="check point for student classifier")
    
    parser.add_argument("--output_all_encoded_layers",
                        default=False,
                        type=bool,
                        help="if output all encoded layers")
    parser.add_argument("--alpha",
                        default=0.95,
                        type=float,
                        help="alpha for distillation")
    parser.add_argument("--T",
                        default=10.,
                        type=float,
                        help="temperature for distillation")
    parser.add_argument("--beta",
                        default=0.0,
                        type=float,
                        help="weight for AT loss")
    parser.add_argument("--kd_model",
                        default="kd",
                        type=str,
                        help="KD model architecture, either kd, kd.full or kd.cls")
    parser.add_argument("--fc_layer_idx",
                        default=None,
                        type=str,
                        help="layers ids we will put FC layers on, only avaiable when kd_model is kd.full")
    parser.add_argument("--weights",
                        default=None,
                        type=str,
                        help="weight of each layer that we will put FC layers on, only available when kd_model is kd.full")
    parser.add_argument("--normalize_patience",
                        action="store_true",
                        help="normalize patience or not")
    # Distillation related parameters
    parser.add_argument(
        "--do_train", action="store_true", help="do training or not"
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="do evaluation during training or not"
    )
    
    # causal distillation related parameters.
    parser.add_argument(
        "--is_diito", action="store_true", help="Whether to use causal distillation"
    )
    parser.add_argument("--diito_type",
                        default="random",
                        type=str,
                        help="DIITO training architecture")
    
    parser.add_argument(
        "--neuron_mapping",
        default="full",
        type=str,
        help="DIITO neuron mapping.",
    )
    parser.add_argument(
        "--interchange_prop",
        default=0.3,
        type=float,
        help="Ratio of tokens to mask for interchange interventions. 1.0 means interchange all",
    )
    parser.add_argument(
        "--interchange_max_token",
        default=-1,
        type=int,
        help="Ratio of tokens to mask for interchange interventions. 1.0 means interchange all",
    )

    parser.add_argument(
        "--interchange_consecutive_only", 
        action="store_true", help="Whether to only interchange consecutive tokens"
    )
    parser.add_argument(
        "--data_augment", action="store_true", help="Whether to experiment with data augmentation"
    )
    parser.add_argument(
        "--data_pair", action="store_true", help="Whether to experiment with data pairing"
    )
    
    parser.add_argument(
        "--is_wandb", action="store_true", help="Whether to use wandb to report metrics to"
    )
    parser.add_argument("--wandb_metadata",
                        default="",
                        type=str,
                        help="You need to give this if you want to log your metrics to wandb in a format of (entity:project)")
    return parser


def complete_argument(
    args,
    is_debug=False,
):
    MODEL_FOLDER = os.path.join(HOME_DATA_FOLDER, 'models')
    if args.student_hidden_layers in [None, 'None']:
        args.student_hidden_layers = 12 if 'base' in args.bert_model else 24
    bert_arch = args.bert_model
    args.bert_model = os.path.join(MODEL_FOLDER, 'pretrained', args.bert_model)

    # student
    if args.encoder_checkpoint_student not in [None, 'None']:
        args.encoder_checkpoint_student = os.path.join(MODEL_FOLDER, args.encoder_checkpoint_student)
    else:
        args.encoder_checkpoint_student = os.path.join(
            MODEL_FOLDER, 'pretrained', bert_arch, 'pytorch_model.bin'
        )
        logger.info('encoder checkpoint not provided, use pre-trained at %s instead' % args.encoder_checkpoint_student)
    if args.cls_checkpoint_student not in [None, 'None']:
        args.cls_checkpoint_student = os.path.join(MODEL_FOLDER, args.cls_checkpoint_student)
        
    # teacher
    if args.encoder_checkpoint_teacher not in [None, 'None']:
        args.encoder_checkpoint_teacher = os.path.join(MODEL_FOLDER, args.encoder_checkpoint_teacher)
    else:
        args.encoder_checkpoint_teacher = os.path.join(
            MODEL_FOLDER, 'finetuned', bert_arch, args.task_name, 'pytorch_model.bin'
        )
        logger.info('encoder checkpoint not provided, use default directory for fine-tuned model at %s instead' % args.encoder_checkpoint_teacher)
    if args.cls_checkpoint_teacher not in [None, 'None']:
        args.cls_checkpoint_teacher = os.path.join(MODEL_FOLDER, args.cls_checkpoint_teacher)
    else:
        args.cls_checkpoint_teacher = os.path.join(
            MODEL_FOLDER, 'finetuned', bert_arch, args.task_name, 'pytorch_model.bin'
        )
        logger.info('encoder checkpoint not provided, use default directory for fine-tuned model at %s instead' % args.encoder_checkpoint_teacher)

    if args.kd_model == 'kd.cls':
        output_name = args.kd_model + '.' + str(args.normalize_patience) + '_' + args.task_name + '_nlayer.' + str(args.student_hidden_layers)
    else:
        output_name = args.kd_model + '_' + args.task_name + '_nlayer.' + str(args.student_hidden_layers)
    output_name += '_lr.' + str(args.learning_rate) + '_T.' + str(args.T) + '_alpha.' + str(args.alpha)
    output_name += '_beta.' + str(args.beta) + '_bs.' + str(args.train_batch_size)
    output_name += '_diito.{}_nm.{}_intprop.{}_intmax.{}_intconsec.{}_dtaug.{}_dtpair.{}_maxex.{}'.format(
            args.is_diito,
            args.neuron_mapping,
            args.interchange_prop,
            args.interchange_max_token,
            args.interchange_consecutive_only,
            args.data_augment,
            args.data_pair,
            args.max_training_examples,
        )
    args.output_dir = os.path.join(args.output_dir, output_name)
    args.run_name = output_name

    run = 1
    while os.path.exists(args.output_dir + '-run-' + str(run)):
        if is_folder_empty(args.output_dir + '-run-' + str(run)):
            logger.info('folder exist but empty, use it as output')
            break
        logger.info(args.output_dir + '-run-' + str(run) + ' exist, trying next')
        run += 1
    args.output_dir += '-run-' + str(run)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.task_name == 'MNLI':
        args.output_dir_mm = args.output_dir.replace('MNLI', 'MNLI-mm', 100)
        os.makedirs(args.output_dir_mm, exist_ok=True)
    
    if not is_debug:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cpu")
        n_gpu = 0
        
    args.device = device
    args.n_gpu = n_gpu
    logger.info("device: {} n_gpu: {}, 16-bits training: {}".format(device, n_gpu, args.fp16))

    if args.seed is None:
        args.seed = random.randint(0, 100000000)
        logger.info('random seed = %d' % args.seed)
    else:
        logger.info('fix seed = %d' % args.seed)

    return args


def get_predefine_argv(mode='glue', task_name='RTE', train_type='kd'):
    """
    the function return some pre-defined arguments for argument parser
    :param mode:  can only be 'glue' for now
    :param task_name:  one of the task name under glue
    :param train_type: could be 'finetune', 'kd' or 'kd.cls'
    :return:
    """
    if mode == 'race':
        raise NotImplementedError('Please run glue for now')
    elif mode == 'glue':
        argv = [
                '--task_name', task_name,
                '--bert_model', 'bert-base-uncased',
                '--max_seq_length', '128',
                '--train_batch_size', '32',
                '--learning_rate', '2e-5',
                '--num_train_epochs', '4',
                '--eval_batch_size', '32',
                '--gradient_accumulation_steps', '1',
                '--log_every_step', '1',
                '--output_dir', os.path.join(HOME_DATA_FOLDER, f'outputs/KD/{task_name}/teacher_12layer'),
                '--do_train',
                '--do_eval',
                # '--fp16', # in the original PKD repo, this is true.
            ]
        if train_type == 'finetune_teacher':
            argv += [
                '--student_hidden_layers', '12',
                '--kd_model', 'kd',
                '--alpha', '0.0',    # alpha = 0 is equivalent to fine-tuning for KD
            ]
        if train_type == 'finetune_student':
            argv += [
                '--student_hidden_layers', '6',
                '--kd_model', 'kd',
                '--alpha', '0.0',
            ]
        elif train_type == 'kd':
            argv += [
                '--student_hidden_layers', '6',
                '--kd_model', 'kd',
                '--alpha', '0.7',
                '--T', '20',
                '--teacher_prediction', f'/home/JJteam/Project/PatientTeacherforBERT/data/outputs/KD/{task_name}/{task_name}_normal_kd_teacher_12layer_result_summary.pkl',
            ]
        elif train_type == 'kd.cls':
            argv += [
                '--learning_rate', '1e-5',
                '--student_hidden_layers', '6',
                '--kd_model', 'kd.cls',
                '--alpha', '0.7',
                '--beta', '500',
                '--T', '10',
                '--teacher_prediction', f'/home/JJteam/Project/PatientTeacherforBERT/data/outputs/KD/{task_name}/{task_name}_patient_kd_teacher_12layer_result_summary.pkl',
                '--fc_layer_idx', '1,3,5,7,9',   # this for pkd-skip
                '--normalize_patience',
            ]
    else:
        raise NotImplementedError('training mode %s has not been implemented yet' % mode)
    return argv
