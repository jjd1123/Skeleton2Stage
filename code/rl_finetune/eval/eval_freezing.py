import numpy as np
import sys
sys.path.append("./model/imitation")
# from embodied_pose.players.im_player_diff import IK_solver
from embodied_pose.utils.edge.quaternion import ax_to_6v, ax_from_6v
import torch
from torch import nn
import glob
import os
import random
from tqdm import tqdm
import argparse

def compute_freezing(trans,pose_6d):
        n_joints = 24
        b, constant_length = trans.shape[:2]
        step = 1
        # trans = sample[...,4:7].reshape(b,3,constant_length//3,3) #b*l*3
        # pose_6d = sample[...,7:].reshape(b,3,constant_length//3,n_joints,6)
        # sample_x = normalizer.unnormalize(x)
        trans = trans.reshape(b,step,constant_length//step,3) #b*l*3
        pose_6d = pose_6d.reshape(b,step,constant_length//step,n_joints,6)
        trans_freeze = (torch.abs(trans[...,1:,:]-trans[...,:-1,:])).mean(dim=(-1,-2))
        # trans_x_freeze = ((trans_x[...,1:,:]-trans_x[...,:-1,:])**2).mean(dim=(-1,-2))
        pose_6d_freeze = (torch.abs(pose_6d[:,:,1:]-pose_6d[:,:,:-1])).mean(dim=(-1,-2,-3))
        # pose_6d_x_freeze = ((pose_6d_x[:,:,1:]-pose_6d_x[:,:,:-1])**2).mean(dim=(-1,-2,-3))
        # trans_rate = (trans_freeze/trans_x_freeze).mean(dim=1)
        # pose_rate = (pose_6d_freeze/pose_6d_x_freeze).mean(dim=1)
        # rate = torch.where((trans_rate+pose_rate)/2>1,1.0,(trans_rate+pose_rate)/2)
        # print(trans_freeze.shape)
        # print(trans_freeze.mean(),pose_6d_freeze.mean())
        rate = trans_freeze.mean(dim=1)+pose_6d_freeze.mean(dim=1)
        return rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion_path",
        type=str,
        default="motions/",
        help="Where to load saved motions",
    )
    opt = parser.parse_args()
    dir = opt.motion_path
    it = glob.glob(os.path.join(dir, "*.pkl"))
    # if len(it) > 1000:
    #     it = random.sample(it, 1000)
    tran = []
    poses = []
    joint3ds = []
    name = []
    for pkl in tqdm(it):
        info = np.load(pkl,allow_pickle=True)
        # q = torch.as_tensor(ax_from_6v(torch.as_tensor(info[:600,7:139].reshape(600, 22, 6))))
        # trans = info[:600, 4:7].reshape(-1,3)
        trans = info["smpl_trans"]#[::2]
        q = info["smpl_poses"].reshape(-1,24,3)#[::2]
        # joint3d = info.item()['pred_position'].reshape(-1,24,3)[::2][:150]
        # joint3ds.append(joint3d)
    # joint3d = np.concatenate(joint3ds,axis=0) 
    # print(joint3d.shape)   
    # ik = nn.DataParallel(IK_solver()).to("cuda")  
    # out = ik(torch.as_tensor(joint3d).to("cuda"))
    # pose_aa_gt_flat = out[...,3:69].cpu().numpy().reshape(-1,22,3)
    # trans_gt_flat =out[...,:3].cpu().numpy()  
        # hand = np.array([[0.0,0.0,0.0]]).repeat(q.shape[0],0)
        # q = np.concatenate([q,hand[:,None,:],hand[:,None,:]],axis=1)
    # tran = trans_gt_flat.reshape(-1,150,3)
    # q = pose_aa_gt_flat.reshape(-1,150,24,3)
        pose_6d = ax_to_6v(torch.as_tensor(q)).numpy()
        tran.append(trans)
        poses.append(pose_6d)
        name.append(pkl)
        # trans = torch.as_tensor(info["pos"][::2].reshape(1,150,3)).to("cuda")
        # q = torch.as_tensor(info['q'][::2].reshape(1,150,24,3)).to("cuda")
        
        # smpl = SMPLSkeleton("cuda")
        # full_pose = (
        #         smpl.forward(q, trans).detach().cpu().numpy()
        #     )
        # joint3d = info.item()['pred_position'].reshape(-1,24,3)[::2][:150]#full_pose[0]#info["full_pose"]
        # print(joint3d.shape)
    freeze_rate = []
    for i in range(len(it)):
        # print(name[i])
        trans = torch.as_tensor(np.concatenate(tran[i],axis=0).reshape(1,-1,3))[:,:600]
        posess = torch.as_tensor(np.concatenate(poses[i],axis=0).reshape(1,-1,24,6))[:,:600]
        # print(trans.shape)
        freeze_rate.append(compute_freezing(torch.as_tensor(trans),torch.as_tensor(posess)))
    # print(freeze_rate.reshape(-1,5))
    # print(freeze_rate.reshape(-1,5).mean(dim=-1))
    # print(freeze_rate.reshape(-1,5).std(dim=-1))
    # _,index = torch.sort(freeze_rate)
    # index = index.reshape(2,-1)
    # print(freeze_rate[index].shape)
    # print(freeze_rate[index].mean(dim=-1),freeze_rate[index].std(dim=-1))
    # index = index.reshape(4,-1)
    # print(freeze_rate[index].shape)
    # print(freeze_rate[index].mean(dim=-1),freeze_rate[index].std(dim=-1))
    # index = index.reshape(8,-1)
    # print(freeze_rate[index].shape)
    # print(freeze_rate[index].mean(dim=-1),freeze_rate[index].std(dim=-1))
    # index = index.reshape(16,-1)
    # print(freeze_rate[index].shape)
    # print(freeze_rate[index].mean(dim=-1),freeze_rate[index].std(dim=-1))
    # index = index.reshape(32,-1)
    # print(freeze_rate[index].shape)
    # print(freeze_rate[index].mean(dim=-1),freeze_rate[index].std(dim=-1))
    freeze_rate = torch.concatenate(freeze_rate,dim=0)
    with open(f"{dir}/metrics.txt","a") as f:
        f.write(f"Freezing rate: {freeze_rate.mean()}/{freeze_rate.std()}\n")
    print(freeze_rate.mean(),freeze_rate.std())