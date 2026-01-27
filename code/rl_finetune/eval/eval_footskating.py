from cgi import test
from turtle import left
import numpy as np
import torch
import os, sys
import argparse
sys.path.append("./")
from vis import SMPLSkeleton



def set_on_ground(data, ground_h=0):
    trans = data[0]
    q = data[1]
    positions = data[2]
    l_toe_h = positions[0, 10, 2]
    r_toe_h = positions[0, 11, 2]
    if abs(l_toe_h - r_toe_h) < 0.02:
        height = (l_toe_h + r_toe_h)/2
    else:
        height = min(l_toe_h, r_toe_h)
    data[2][:,:,2] = data[2][:,:,2] - (height -  ground_h)
    data[0][2] = data[0][2] - (height -  ground_h)
    return data

def calc_foot_skating_ratio(data,ground_height):
    trans = data[0] = torch.as_tensor(data[0])
    q = data[1] = torch.as_tensor(data[1])
    smpl = SMPLSkeleton("cuda")
    joint3d = (
            smpl.forward(q.unsqueeze(0).to("cuda"), trans.unsqueeze(0).to("cuda")).detach().cpu()
        )[0][:600]
    data.append(joint3d)
    data = set_on_ground(data, ground_height)
    model_xp = data[2]

    l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx = 7, 8, 10, 11
    relevant_joints = [l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx]
    pred_joint_xyz = model_xp[:, relevant_joints, :] 
    pred_vel = torch.zeros_like(pred_joint_xyz)
    pred_vel[:-1] = (
        pred_joint_xyz[1:, :, :] - pred_joint_xyz[:-1, :, :]
    )  # (S-1, 4, 3)
    left_foot_y_ankle = model_xp[ :, l_ankle_idx, 2]
    right_foot_y_ankle = model_xp[ :, r_ankle_idx, 2]
    left_foot_y_toe = model_xp[:, l_foot_idx, 2]
    right_foot_y_toe = model_xp[:, r_foot_idx, 2]

    print("pred_vel.shape", pred_vel.shape)
    left_fc_mask = (left_foot_y_ankle <= (ground_height +0.08)) & (left_foot_y_toe <= (ground_height +0.05))
    right_fc_mask = (right_foot_y_ankle <= (ground_height +0.08)) & (right_foot_y_toe <= (ground_height +0.05))
    # fc_mask_y = torch.cat([fc_mask_ankle, fc_mask_teo], dim=1).squeeze(0)
    left_pred_vel = torch.cat([pred_vel[:, 0:1, :], pred_vel[:, 2:3, :]], dim=1)
    right_pred_vel = torch.cat([pred_vel[:, 1:2, :], pred_vel[:, 3:4, :]], dim=1)
    # pred_vel[~fc_mask_v] = 0
    left_pred_vel[~left_fc_mask] = 0
    right_pred_vel[~right_fc_mask] = 0
    print("left_fc_mask", left_fc_mask.shape)
    left_static_num = torch.sum(left_fc_mask)
    print("left_static_num", left_static_num)
    right_static_num = torch.sum(right_fc_mask)
    print("right_static_num", right_static_num)
    # sys.exit(0)
    left_velocity_foot_tangent = torch.cat([left_pred_vel[:, :, 0:1], left_pred_vel[:, :, 2:3] ], dim=2)
    left_velocity_foot_tangent = torch.abs(torch.mean(left_velocity_foot_tangent, dim=-1))        # T, 4
    right_velocity_foot_tangent = torch.cat([right_pred_vel[:, :, 0:1], right_pred_vel[:, :, 2:3] ], dim=2)
    right_velocity_foot_tangent = torch.abs(torch.mean(right_velocity_foot_tangent, dim=-1))        # T, 4
    print("right_velocity_foot_tangent.shape", right_velocity_foot_tangent.shape)

    left_velocity_foot_tangent = left_velocity_foot_tangent > 0.01 #(0.05 / 30)          # 0.025     # 0.1/30
    right_velocity_foot_tangent = right_velocity_foot_tangent > 0.01 #(0.05 / 30)
    left_slide_frames = torch.any(left_velocity_foot_tangent, dim=-1)
    left_slide_num = torch.sum(left_slide_frames)
    left_ratio = left_slide_num / left_static_num
    # if torch.isnan(left_ratio):
    #     left_ratio = torch.zeros_like(left_ratio)
    left_ratio = left_ratio.item()
    print("left_ratio", left_ratio)

    right_slide_frames = torch.any(right_velocity_foot_tangent, dim=-1)
    right_slide_num = torch.sum(right_slide_frames)
    right_ratio = right_slide_num / right_static_num
    # if torch.isnan(right_ratio):
    #     right_ratio = torch.zeros_like(right_ratio)
    right_ratio = right_ratio.item()
    print("right_ratio", right_ratio)




    # static_ratio = (slide_num / static_num).item()
    # print("static_ratio", static_ratio)

    return left_ratio, right_ratio




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion_path",
        type=str,
        default="motions/",
        help="Where to load saved motions",
    )
    opt = parser.parse_args()
    test_dir = opt.motion_path
    # test_dir = './eval_20s/phy_eval'

    ground_height = 0
    left_ratio_list = []
    right_ratio_list = []
    ratio_list = []
    for file in os.listdir(test_dir):
        if file[-3:] != 'pkl':
            continue
        filepath = os.path.join(test_dir, file)
        data = np.load(filepath,allow_pickle=True) 
        trans = data["smpl_trans"]
        q = data["smpl_poses"].reshape(-1,24,3) 
        data = [trans,q]
        left_ratio, right_ratio = calc_foot_skating_ratio(data,ground_height=ground_height)
        left_ratio_list.append(left_ratio)
        right_ratio_list.append(right_ratio)
        
    print("final ")
    print("left_ratio:", np.mean(left_ratio_list))
    print("right_ratio:", np.mean(right_ratio_list))
    print("ratio:", (np.mean(right_ratio_list) + np.mean(left_ratio_list))/2 )
    print('test_dir', test_dir)