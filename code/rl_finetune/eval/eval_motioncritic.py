import argparse
import glob
import os
import pickle

import numpy as np
from tqdm import tqdm
import random
import torch
import sys
sys.path.append("./")
from vis import SMPLSkeleton
from model.imitation.embodied_pose.utils.edge.quaternion import ax_from_6v
from eval.MotionCritic.MotionCritic.lib.model.load_critic import load_critic
from pytorch3d.transforms import (RotateAxisAngle, axis_angle_to_quaternion,
                                  quaternion_multiply,
                                  quaternion_to_axis_angle)


# def zup_to_yup(trans, q):
#     """
#     将SMPL数据从Z-up转换到Y-up坐标系，保持与地面的距离关系
    
#     Z-up系统: 地面是XY平面(z=0)，z值表示高度
#     Y-up系统: 地面是XZ平面(y=0)，y值表示高度
    
#     转换逻辑：
#     - 高度维度直接映射: old_z -> new_y
#     - 地面坐标需要旋转90度
    
#     Args:
#         trans: torch.Tensor, shape (..., 3), SMPL平移向量 [x, y, z]
#         q: torch.Tensor, shape (..., 24, 3), SMPL姿态参数(axis-angle)
    
#     Returns:
#         trans_yup: 转换后的平移向量
#         q_yup: 转换后的姿态参数
#     """
#     # 转换平移向量
#     # Z-up: z是高度，(x,y)是地面坐标
#     # Y-up: y是高度，(x,z)是地面坐标
#     # 映射: x->x, y->z, z->y
#     trans_yup = trans.clone()
#     trans_yup[..., 0] = trans[..., 0]       # x保持不变(地面x方向)
#     trans_yup[..., 1] = trans[..., 2]       # 高度: old_z -> new_y
#     trans_yup[..., 2] = trans[..., 1]       # 地面: old_y -> new_z
    
#     # 转换姿态参数 (axis-angle的轴向量需要相同变换)
#     q_yup = q.clone()
#     q_yup[..., 0] = q[..., 0]       # 绕x轴的旋转不变
#     q_yup[..., 1] = q[..., 2]       # 绕z轴旋转 -> 绕y轴旋转
#     q_yup[..., 2] = q[..., 1]       # 绕y轴旋转 -> 绕z轴旋转
    
#     return trans_yup, q_yup

def zup_to_yup(trans,q):
    # 将 z-up 转换为 y-up（与你的代码相反的操作）
    root_q = q[:, :, :1, :]  # sequence x 1 x 3
    root_q_quat = axis_angle_to_quaternion(root_q)

    # 使用 -90 度旋转（或者说逆旋转）
    rotation = torch.Tensor(
        [0.7071068, -0.7071068, 0, 0]  # -90 degrees about the x axis
    ).cuda()
    root_q_quat = quaternion_multiply(rotation, root_q_quat)
    root_q = quaternion_to_axis_angle(root_q_quat)
    q[:, :, :1, :] = root_q

    # 旋转根位置
    pos_rotation = RotateAxisAngle(-90, axis="X", degrees=True).cuda()
    root_pos = pos_rotation.transform_points(trans)  
    # 这相当于 (y, z) -> (z, -y)
    return root_pos, q

def calc_physical_score(dir):
    scores = []
    names = []
    accelerations = []
    up_dir = 2  # z is up
    flat_dirs = [i for i in range(3) if i != up_dir]
    DT = 1 / 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    critic_model = load_critic("/data/PhysDanceRL/code/rl_finetune/eval/MotionCritic/motioncritic_pre.pth", device)

    it = glob.glob(os.path.join(dir, "*.pkl"))
    if len(it) > 1000:
        it = random.sample(it, 1000)
    for pkl in it:
        # info = np.load(pkl)
        # poses = info[:600,7:139].reshape(1,600, 22, 6)
        # DEBUG
        # poses = poses[:,:,:3]
        # poses = torch.as_tensor(poses)
        # q = torch.as_tensor(ax_from_6v(poses)).to("cuda")
        # trans = torch.as_tensor(info[:600, 4:7].reshape(1,-1,3)).to("cuda")
        info = np.load(pkl,allow_pickle=True)
        try:
            trans = torch.as_tensor(info["pos"][::2].reshape(1,-1,3)/info['scale'][0]).to("cuda")
            q = torch.as_tensor(info['q'][::2].reshape(1,-1,24,3)).to("cuda")
        except:
            trans = torch.as_tensor(info["smpl_trans"].reshape(1,-1,3)).to("cuda")
            q = torch.as_tensor(info['smpl_poses'].reshape(1,-1,24,3)).to("cuda")
        # smpl = SMPLSkeleton("cuda")
        # joint3d = (
        #         smpl.forward(q, trans).detach().cpu().numpy()
        #     )[0][:600]
        # trans = torch.zeros_like(trans)
        # q[:,:,0,:] = 0
        trans,q = zup_to_yup(trans,q)
        motion = torch.cat([trans.unsqueeze(-2)[:,:243],q[:,:243]],dim=-2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(motion.shape)
        with torch.no_grad():
            critic_scores = critic_model.module.batch_critic(motion)

        scores.append(critic_scores.detach().cpu().numpy())
        names.append(pkl)

    out = np.mean(scores) #* 10000
    with open(f"{dir}/metrics.txt","a") as f:
        f.write(f"motion critic: {out}\n")
    print(f"{dir} has a mean motion critic of {out}")


def parse_eval_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion_path",
        type=str,
        default="motions/",
        help="Where to load saved motions",
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_eval_opt()
    calc_physical_score(opt.motion_path)
