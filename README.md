![Python 3.7](https://img.shields.io/badge/python-3.7-blueviolet.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-MIT-05b502.svg?style=plastic)

# Causal Distillation for Natural Language Understanding Tasks (DIITO-XXS)

<div align="center">
  <img src="https://i.ibb.co/Q8NNHPJ/Screen-Shot-2021-12-06-at-4-53-28-PM.png" style="float:left" width="800px">
</div>
<p></p>

This is an **ONGOING** research effort. So, don't expect everything to be working. The is an extended implementation of our preprint [Causal Distillation for Language Models](https://zen-wu.social/papers/ACL22_CausalDistill.pdf) by applying the method to task-specific models (i.e., the teacher model here is a fine-tuned model). The codebased for the distillation method **the distillation interchange intervention training objective (DIITO)** can be found [here](https://github.com/frankaging/Causal-Distill).

We fork our main codebase from the [Huggingface Distillation Interface](https://github.com/huggingface/transformers/tree/master/examples/research_projects/distillation).

## Release Notes
:white_check_mark: 02/21/2022 Release this codebase for others who are interested in applying [DIITO](https://github.com/frankaging/Causal-Distill) to task-specific models.

If you experience any issues or have suggestions, please contact me either thourgh the issues page or at wuzhengx@stanford.edu. 

## Main Contents
* [Citation](#citation)
* [Requirements](#requirements)
* [Distillation](#distillation)

## Citation
If you use this repository, please cite the following two papers: [paper for interchange intervention training](https://arxiv.org/abs/2112.00826), and [paper for the our distillation method](https://arxiv.org/abs/2109.08994).
```stex
  @article{geiger-etal-2021-iit,
        title={Inducing Causal Structure for Interpretable Neural Networks}, 
        author={Geiger, Atticus and Wu, Zhengxuan and Lu, Hanson and Rozner, Josh and Kreiss, Elisa and Icard, Thomas and Goodman, Noah D. and Potts, Christopher},
        year={2021},
        eprint={2112.00826},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
  }

  @article{wu-etal-2021-distill,
        title={Causal Distillation for Language Models}, 
        author={Wu, Zhengxuan and Geiger, Atticus and Rozner, Josh and Kreiss, Elisa and Lu, Hanson and Icard, Thomas and Potts, Christopher and Goodman, Noah D.},
        year={2021},
        eprint={2112.02505},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
  }
```

## Requirements
- Python 3.6 or 3.7 are supported.
- Pytorch Version: 1.9.0
- Transfermers Version: 4.11.3
- Datasets Version: Version: 1.8.0
- Since we build our codebase off the [Huggingface Distillation Interface](https://github.com/huggingface/transformers/tree/master/examples/research_projects/distillation), please review their doc for requirements.

## Distillation
Now, here is an example for you to distill with our causal distillation objective or without,
```bash
python KD_training.py \
--task_name SST-2 \
--output_dir data/outputs/KD/SST-2/teacher_12layer/ \
--bert_model bert-base-uncased \
--max_seq_length 128 \
--train_batch_size 32 \
--learning_rate 2e-5 \
--num_train_epochs 5 \
--eval_batch_size 32 \
--gradient_accumulation_steps 1 \
--log_interval 10 \
--checkpoint_interval 100 \
--do_train \
--fp16 False \
--student_hidden_layers 6 \
--fc_layer_idx 1,3,5,7,9 \
--kd_model kd \
--alpha 0.7 \
--T 20 \
--is_wandb \
--wandb_metadata wuzhengx:DIITO-XXS \
--neuron_mapping full \
--is_diito \
--interchange_prop 0.3
```

