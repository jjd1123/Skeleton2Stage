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
import trimesh

# 读取人体的obj文件
def load_human_mesh(obj_file):
    mesh = trimesh.load(obj_file)
    return mesh

# 计算穿模面片数量
def calculate_penetration_faces(mesh, ground_height=0.0):
    # 获取网格的顶点和面片
    vertices = mesh.vertices
    faces = mesh.faces
    # print(faces.shape)
    # 计算每个顶点与地面的距离
    distances = vertices[:, 2] - ground_height
    
    # 标记哪些顶点在地面以下
    below_ground = distances < 0
    
    # 统计包含至少一个低于地面的顶点的面片
    penetration_faces = []
    for face in faces:
        if any(below_ground[face]):  # 如果面片的任意顶点低于地面
            penetration_faces.append(face)
    
    return len(penetration_faces)

import numpy as np

def calculate_ground_penetration_and_floating(motion_data, foot_joint_indices, ground_height=0.0, floating_threshold=0.05,up_dir = 2):
    """
    计算人体动作数据的地面穿模和悬浮指标
    
    参数：
    motion_data: numpy数组，形状为 (帧数, 关节点数, 3)
    foot_joint_indices: 脚部关节点索引列表
    ground_height: 地面高度（默认y=0）
    floating_threshold: 悬浮判断阈值（米）
    
    返回：
    包含各项指标的字典
    """
    y_coords = motion_data[:, :, up_dir]
    
    # 地面穿模计算
    below_ground = np.min(y_coords,axis=-1) < ground_height
    penetration_depths = ground_height - np.min(y_coords,axis=-1)[below_ground]
    
    # 悬浮计算
    # foot_heights = y_coords[:, foot_joint_indices]
    # print(y_coords.shape)
    foot_heights = np.min(y_coords,axis=-1)
    # print(foot_heights.shape)
    above_threshold = foot_heights > (ground_height + floating_threshold)
    
    # 统计指标
    metrics = {
        'ground_penetration': {
            'total_depth': np.sum(penetration_depths),
            'average_depth': np.mean(penetration_depths) if penetration_depths.size else 0.0,
            'max_depth': np.max(penetration_depths) if penetration_depths.size else 0.0,
            'affected_frames': np.any(below_ground, axis=0).sum(),
            'total_frames': motion_data.shape[0],
            'penetration_points': below_ground.sum(),
            "penetration_depths": penetration_depths
        },
        'floating': {
            'floating_instances': above_threshold.sum(),
            'floating_ratio': above_threshold.mean(),
            'average_height': np.mean(foot_heights[above_threshold] - ground_height) if above_threshold.any() else 0.0,
            'max_height': np.max(foot_heights - ground_height) if foot_heights.size else 0.0,
            'threshold': floating_threshold,
            "floating_heights": foot_heights[above_threshold] - ground_height
        }
    }
    
    return metrics

def calc_physical_score(dir):
    scores = []
    avg_pen = []
    avg_float = []
    pen = []
    floating = []
    names = []
    accelerations = []
    up_dir = 2  # z is up
    flat_dirs = [i for i in range(3) if i != up_dir]
    DT = 1 / 30

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
        # number = int(os.path.basename(pkl)[0])
        info = np.load(pkl,allow_pickle=True)
        # try:
        #     trans = torch.as_tensor(info["pos"][::2].reshape(1,-1,3)/info['scale'][0]).to("cuda")
        #     q = torch.as_tensor(info['q'][::2].reshape(1,-1,24,3)).to("cuda")
        # except:
        #     trans = torch.as_tensor(info['smpl_trans'][::2].reshape(1,-1,3)/info["smpl_scaling"][0]).to("cuda")
        #     q = torch.as_tensor(info['smpl_poses'][::2].reshape(1,-1,24,3)).to("cuda")
        # joint3d = info['pred_position'][::2][:600].reshape(-1,24,3)
        # import ipdb;ipdb.set_trace()
        try:
            trans = torch.as_tensor(info["pos"][::2].reshape(1,-1,3)/info['scale'][0]).to("cuda")
            q = torch.as_tensor(info['q'][::2].reshape(1,-1,24,3)).to("cuda")
            # print(1)
        except:
            trans = torch.as_tensor(info["smpl_trans"].reshape(1,-1,3)).to("cuda")
            q = torch.as_tensor(info['smpl_poses'].reshape(1,-1,24,3)).to("cuda")
            # print(1)
        smpl = SMPLSkeleton("cuda")
        joint3d = (
                smpl.forward(q, trans).detach().cpu().numpy()
            )[0][:600]
        # joint3d[...,2] = -joint3d[...,2]
        # print(joint3d.shape)
        # joint3d = joint3d[::2][:600]
        # joint3d = info["full_pose"]
        # import ipdb; ipdb.set_trace()
        # joint3d = np.load(pkl, allow_pickle=True).item()['pred_position'][::2].reshape(-1,24,3)[:600]
        # print(joint3d.shape)
        # l_toe_h = joint3d[:, 10, up_dir].mean()
        # r_toe_h = joint3d[:, 11, up_dir].mean()
        # if abs(l_toe_h - r_toe_h) < 0.02:
        #     height = (l_toe_h + r_toe_h)/2
        # else:
        #     height = min(l_toe_h, r_toe_h)
        height = joint3d[:,:,up_dir].min(axis = -1).mean()
        # height =0
        # 计算指标（假设脚部关节为7和8）
        metrics = calculate_ground_penetration_and_floating(
            joint3d,
            foot_joint_indices=[10, 11],
            ground_height=height,
            floating_threshold=0.05,
            up_dir = up_dir
        )
        
        # # 打印结果
        # print("地面穿模指标：")
        # print(f"总穿模深度: {metrics['ground_penetration']['total_depth']:.4f} m")
        # print(f"平均穿模深度: {metrics['ground_penetration']['average_depth']:.4f} m")
        # print(f"最大穿模深度: {metrics['ground_penetration']['max_depth']:.4f} m")
        # print(f"受影响帧数: {metrics['ground_penetration']['affected_frames']}/{metrics['ground_penetration']['total_frames']}")
        # print(f"总穿模点数: {metrics['ground_penetration']['penetration_points']}")
        
        # print("\n悬浮指标:")
        # print(f"悬浮次数: {metrics['floating']['floating_instances']}")
        # print(f"悬浮比例: {metrics['floating']['floating_ratio']:.2%}")
        # print(f"平均悬浮高度: {metrics['floating']['average_height']:.4f} m")
        # print(f"最大悬浮高度: {metrics['floating']['max_height']:.4f} m")
        # print(f"判断阈值: {metrics['floating']['threshold']} m")
        pen.extend(metrics['ground_penetration']['penetration_depths'])
        floating.extend(metrics['floating']['floating_heights'])
        avg_pen.append(metrics['ground_penetration']['average_depth'])
        avg_float.append(metrics['floating']['average_height'])
        # joint3d[..., up_dir] = joint3d[..., up_dir] - height

    with open(f"{dir}/metrics.txt","a") as f:
        f.write(f"Penetration depth: {np.sum(pen)/24000.0*1000}\n")
        f.write(f"Floating height: {np.sum(floating)/24000.0*1000}\n")
    print(dir)
    print("Average penetration depth:", np.sum(pen)/24000.0*1000)
    print("Average floating height:", np.sum(floating)/24000.0*1000) 
    # print("Average penetration depth:", np.mean(avg_pen)*1000)
    # print("Average floating height:", np.mean(avg_float)*1000)       

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