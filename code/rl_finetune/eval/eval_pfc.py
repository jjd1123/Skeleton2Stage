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


def calc_physical_score(dir):
    scores = []
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
        info = np.load(pkl,allow_pickle=True)
        try:
            trans = torch.as_tensor(info["pos"][::2].reshape(1,-1,3)/info['scale'][0]).to("cuda")
            q = torch.as_tensor(info['q'][::2].reshape(1,-1,24,3)).to("cuda")
        except:
            trans = torch.as_tensor(info["smpl_trans"].reshape(1,-1,3)).to("cuda")
            q = torch.as_tensor(info['smpl_poses'].reshape(1,-1,24,3)).to("cuda")
        smpl = SMPLSkeleton("cuda")
        joint3d = (
                smpl.forward(q, trans).detach().cpu().numpy()
            )[0][:600]
        # joint3d[...,2]=-joint3d[...,2]
        # joint3d = joint3d[::2][:600]
        # joint3d = info["full_pose"]
        # import ipdb; ipdb.set_trace()
        # joint3d = np.load(pkl, allow_pickle=True).item()['pred_position'][::2].reshape(-1,24,3)[:600]
        # print(joint3d.shape)
        
        root_v = (joint3d[1:, 0, :] - joint3d[:-1, 0, :]) / DT  # root velocity (S-1, 3)
        root_a = (root_v[1:] - root_v[:-1]) / DT  # (S-2, 3) root accelerations
        # clamp the up-direction of root acceleration
        root_a[:, up_dir] = np.maximum(root_a[:, up_dir], 0)  # (S-2, 3)
        # l2 norm
        root_a = np.linalg.norm(root_a, axis=-1)  # (S-2,)
        scaling = root_a.max()
        root_a /= scaling

        foot_idx = [7, 10, 8, 11]
        feet = joint3d[:, foot_idx]  # foot positions (S, 4, 3)
        foot_v = np.linalg.norm(
            feet[2:, :, flat_dirs] - feet[1:-1, :, flat_dirs], axis=-1
        )  # (S-2, 4) horizontal velocity
        foot_mins = np.zeros((len(foot_v), 2))
        foot_mins[:, 0] = np.minimum(foot_v[:, 0], foot_v[:, 1])
        foot_mins[:, 1] = np.minimum(foot_v[:, 2], foot_v[:, 3])

        foot_loss = (
            foot_mins[:, 0] * foot_mins[:, 1] * root_a
        )  # min leftv * min rightv * root_a (S-2,)
        foot_loss = foot_loss.mean()
        # print(pkl,foot_loss*10000)
        scores.append(foot_loss)
        names.append(pkl)
        accelerations.append(foot_mins[:, 0].mean())

    out = np.mean(scores) * 10000
    with open(f"{dir}/metrics.txt","a") as f:
        f.write(f"PFC: {out}\n")
    print(f"{dir} has a mean PFC of {out}")


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
