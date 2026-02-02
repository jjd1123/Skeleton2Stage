PATH_TO_VID2PLAYER3D=$1
cd $PATH_TO_VID2PLAYER3D
source ../../../environment/environment.sh
python embodied_pose/run.py --headless --cfg amass_im_pretrain --play \
                                --checkpoint epoch10000    