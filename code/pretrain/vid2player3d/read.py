import joblib
import isaacgym
import torch

# shape = joblib.load("seq_name_splits.pkl")
# print(shape)
# motion = torch.load("data/motion_lib/amass/mlib_part_00000.pth")
# print(vars(motion)["_motion_fps"])
data = joblib.load("/mnt/Nerf001/jidong/UniversalHumanoidControl/sample_data/amass_copycat_occlusion_v2.pkl")
import ipdb
ipdb.set_trace()
print(data)