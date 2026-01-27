PATH_TO_VID2PLAYER3D= 
cd $PATH_TO_VID2PLAYER3D
source ../../../environment/environment.sh
python embodied_pose/run.py --headless --cfg amass_im \
                                --checkpoint epoch10000    