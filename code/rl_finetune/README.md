# RLFT for diffusion models
This code has the implementation of RLFT for diffusion models, an imitation reward, and other related rewards. Now it supports finetuning EDGE and POPDG, you can modify code to finetune your own base model with your rewards.

## Finetuning
```
bash run.sh exp_name gpu_parallel_num epoch_num batch_size
```
## Evaluation
```
# Before evaluation, you should make sure 
# you have the correct settings in metric computation scripts,
# and models in "EDGE.py".
bash eval.sh exp_name epoch_num motion_save_root ckpt_root cached_music_features
```
## Finetuning with your own base model and rewards
```
Coming soon!
```