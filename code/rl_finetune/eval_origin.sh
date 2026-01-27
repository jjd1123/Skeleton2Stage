#!/bin/bash

# CUDA_VISIBLE_DEVICES=1 python eval/eval_ground.py --motion_path motion/origin/
# CUDA_VISIBLE_DEVICES=1 python eval/eval_ground.py --motion_path motion/35_1500_1171/
# CUDA_VISIBLE_DEVICES=1 python eval/eval_ground.py --motion_path motion/30_add_reward/
# CUDA_VISIBLE_DEVICES=1 python eval/eval_ground.py --motion_path motion/proj_1
# CUDA_VISIBLE_DEVICES=1 python eval/eval_ground.py --motion_path motion/proj_2

# CUDA_VISIBLE_DEVICES=1 python eval/eval_pfc.py --motion_path motion/origin/
# CUDA_VISIBLE_DEVICES=1 python eval/eval_pfc.py --motion_path motion/35_1500_1171/
# CUDA_VISIBLE_DEVICES=1 python eval/eval_pfc.py --motion_path motion/30_add_reward/
# CUDA_VISIBLE_DEVICES=1 python eval/eval_pfc.py --motion_path motion/proj_1
# CUDA_VISIBLE_DEVICES=1 python eval/eval_pfc.py --motion_path motion/proj_2
source /data/PhysDanceRL/environment/environment.sh
file_paths="
    origin_pop_foot_mean
"
# epoch=1500
#     newfgc_per_frame_35_45_150
#     newfgc_per_frame_3_5_150
# "
# file_path1="newfgc_per_frame_3_5_150"
for file_path in $file_paths; do
    echo $file_path
    # python test.py --save_motions --use_cached_features \
    #                 --feature_cache_dir music/cached_features/ \
    #                 --motion_save_dir ./runs/$file_path
    python test.py --save_motions --use_cached_features \
                    --feature_cache_dir /data/PhysDanceRL/code/rl_finetune/music/pop_music/cached_features \
                    --motion_save_dir ./runs/$file_path \
                    # --checkpoint ../runs/train/$file_path/weights/train-$epoch.pt 
done
# python eval/eval_ground.py --motion_path runs/remove_k_vertical__0_frame_on_ground_4_6
# python eval/eval_ground.py --motion_path runs/remove_k_vertical__0_frame_on_ground
# python eval/eval_ground.py --motion_path runs/proj
# python eval/eval_ground.py --motion_path runs/origin
# python eval/eval_ground.py --motion_path runs/root_per_frame
# python eval/eval_ground.py --motion_path runs/fgc_per_frame
# python eval/eval_ground.py --motion_path runs/root_per_frame_4_150
for file_path in $file_paths; do
    python eval/eval_ground.py --motion_path runs/$file_path
    python eval/eval_pfc.py --motion_path runs/$file_path
    python eval/eval_ba.py --motion_path runs/$file_path
    python eval/eval_freezing.py --motion_path runs/$file_path
    python get_obj.py --motion_path runs/$file_path
    cd ../../environment/torch-mesh-isect
    python examples/detect_and_plot_collisions.py --motion_path ../../code/rl_finetune/runs/$file_path
    cd ../../code/rl_finetune
done
# python eval/eval_ground.py --motion_path runs/$file_path1
# python eval/eval_pfc.py --motion_path runs/$file_path1
# python eval/eval_ba.py --motion_path runs/$file_path1
# python eval/eval_pfc.py --motion_path runs/root_per_frame_4_150
# python eval/eval_pfc.py --motion_path runs/remove_k_vertical__0_frame_on_ground_4_6
# python eval/eval_pfc.py --motion_path runs/remove_k_vertical__0_frame_on_ground
# python eval/eval_pfc.py --motion_path runs/proj
# python eval/eval_pfc.py --motion_path runs/origin
# python eval/eval_pfc.py --motion_path runs/root_per_frame
# python eval/eval_pfc.py --motion_path runs/fgc_per_frame