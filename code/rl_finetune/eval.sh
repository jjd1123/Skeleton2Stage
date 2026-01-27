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
# Evaluation: FID_k: 57.2973, FID_g: 27.6393                                                         
# Evaluation: Dist_k: 2.0348, Dist_g: 2.1400
# cd /workspace/jja3wx/jidong/EDGE
source /data/PhysDanceRL/environment/environment.sh
file_paths="
    $1
"   
# newfgc_per_frame_30_40_55_150_popdg_newplayer01_axisz_pfc_mean155
# newfgc_per_frame_40_40_55_150_popdg_newplayer01_axisz_pfc
# newfgc_per_frame_30_50_155_150_axisz_0506
# newfgc_per_frame_35_50_150
# newfgc_per_frame_35_49_150
# newfgc_per_frame_35_50_150
# newfgc_per_frame_29_485_150_axisz_0501
# newfgc_per_frame_30_30_485_150_popdg_newplayer01_axisz_pfc
# newfgc_per_frame_29_48_150_axisz_0502
epoch=$2
#     newfgc_per_frame_35_45_150
#     newfgc_per_frame_3_5_150
# "
# file_path1="newfgc_per_frame_3_5_150"
for file_path in $file_paths; do
    echo $file_path
    savepath="/data/PhysDanceRL/code/rl_finetune/runs/$file_path-$epoch"
    # python test.py --save_motions --use_cached_features \
    #                 --feature_cache_dir music/cached_features/ \
    #                 --motion_save_dir ./runs/$file_path-$epoch\
    #                 --checkpoint /workspace/jja3wx/jidong/EDGE-main/runs/train/$file_path/weights/train-$epoch.pt 
    python test.py --save_motions --use_cached_features \
                    --feature_cache_dir music/cached_features/ \
                    --motion_save_dir $savepath \
                    --checkpoint /data/PhysDanceRL/runs/train/phys_knovertical_fgc_51_49_01_term3_4parallel/weights/train-1500.pt
    # python test.py --save_motions --use_cached_features \
    #                 --feature_cache_dir /data/PhysDanceRL/code/rl_finetune/music/pop_music/cached_features/ \
    #                 --motion_save_dir $savepath \
    #                 --checkpoint ../../runs/train/$file_path/weights/train-$epoch.pt 
done
# python eval/eval_ground.py --motion_path runs/remove_k_vertical__0_frame_on_ground_4_6
# python eval/eval_ground.py --motion_path runs/remove_k_vertical__0_frame_on_ground
# python eval/eval_ground.py --motion_path runs/proj
# python eval/eval_ground.py --motion_path runs/origin
# python eval/eval_ground.py --motion_path runs/root_per_frame
# python eval/eval_ground.py --motion_path runs/fgc_per_frame
# python eval/eval_ground.py --motion_path runs/root_per_frame_4_150
for file_path in $file_paths; do 
    # savepath=$file_path
    python eval/eval_ground.py --motion_path $savepath
    python eval/eval_pfc.py --motion_path $savepath
    python eval/eval_ba.py --motion_path $savepath
    python eval/eval_motioncritic.py --motion_path $savepath
    # python eval/eval_freezing.py --motion_path $savepath
    # cd model/POPDG
    # python eval/calculate_scores.py --motion_path $savepath
    # cd ../..
    python get_obj.py --motion_path $savepath
    cd ../../environment/torch-mesh-isect
    python examples/detect_and_plot_collisions.py --motion_path $savepath
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