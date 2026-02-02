PATH_TO_VID2PLAYER3D=$1
cd $PATH_TO_VID2PLAYER3D
source ../../../environment/environment.sh
python embodied_pose/run.py --cfg amass_im_pretrain --rl_device cuda:0 --headless
python embodied_pose/run.py --cfg edge_im --rl_device cuda:0 --headless