#!/bin/bash

source ../../environment/environment.sh
file_paths="
    $1
"   

epoch=$2
save_root=$3
ckpt_root=$4
cached_features=$5

for file_path in $file_paths; do
    echo $file_path
    savepath="$save_root/$file_path-$epoch"
    python test.py --save_motions --use_cached_features \
                    --feature_cache_dir $cached_features \
                    --motion_save_dir $savepath \
                    --checkpoint $ckpt_root/$file_path/weights/train-$epoch.pt
done

for file_path in $file_paths; do 
    savepath="$save_root/$file_path-$epoch"
    python eval/eval_ground.py --motion_path $savepath
    python eval/eval_pfc.py --motion_path $savepath
    python eval/eval_ba.py --motion_path $savepath
    python eval/eval_motioncritic.py --motion_path $savepath
    python get_obj.py --motion_path $savepath
    cd ../../environment/torch-mesh-isect
    python examples/detect_and_plot_collisions.py --motion_path $savepath
    cd ../../code/rl_finetune
done