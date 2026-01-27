import numpy as np
import glob
from model.imitation.embodied_pose.run_player import get_player
import torch
from pytorch3d.transforms import (RotateAxisAngle, axis_angle_to_quaternion,
                                  quaternion_multiply,
                                  quaternion_to_axis_angle)
import pickle
from vis import SMPLSkeleton
from model.imitation.embodied_pose.players.im_player_diff import IK_solver

if __name__ == "__main__":
    motion_path = "/mnt/Nerf001/jidong/Bailando/experiments/actor_critic/eval/pkl/ep000010/"#EDGE/data/test/motions_sliced"
    save_dir = "./overall_eval/Bailando_ik_eval"
    motion_file = glob.glob(f"{motion_path}/*.npy")
    player = get_player(torch.device("cuda:0"))
    trans = []
    smpl_poses = []
    motions = []
    ik = IK_solver().to("cuda")
    for i in motion_file:
        motion = np.load(i,allow_pickle=True)
        motion = motion.item()['pred_position'].reshape(-1,24,3)[::2][:150]
        print(motion.shape)
        motions.append(motion)
        # smpl_poses.append(motion["q"][::2].reshape(-1,24,3))
        # trans.append(motion["pos"][::2])
    motion = np.stack(motions,axis=0).reshape(-1,24,3)
    out = ik(torch.as_tensor(motion).to("cuda"))
    pose = smpl_poses = out[...,3:75].reshape(-1,150,24,3)
    trans = out[...,:3].reshape(-1,150,3)
    # trans = torch.as_tensor(np.stack(trans,axis = 0))
    # smpl_poses = torch.as_tensor(np.stack(smpl_poses,axis = 0))
    # print(smpl_poses.shape)
    # root_q = smpl_poses[:, :, :1, :]  # sequence x 1 x 3
    # root_q_quat = axis_angle_to_quaternion(root_q)
    # rotation = torch.Tensor(
    #     [0.7071068, 0.7071068, 0, 0]
    # )  # 90 degrees about the x axis
    # root_q_quat = quaternion_multiply(rotation, root_q_quat)
    # root_q = quaternion_to_axis_angle(root_q_quat)
    # smpl_poses[:, :, :1, :] = root_q

    # # don't forget to rotate the root position too 😩
    # pos_rotation = RotateAxisAngle(90, axis="X", degrees=True)
    # trans = pos_rotation.transform_points(
    #     trans
    # )
    # sample = torch.concatenate([trans,smpl_poses.reshape(186,150,72)],axis=-1)
    # trans,pose = player.phys_proj_test(torch.as_tensor(sample),save_motion=True)
    smpl = SMPLSkeleton("cuda")
    print(trans.shape)
    print(pose.shape)
    full_pose =  smpl.forward(pose.reshape(-1,600,24,3),trans.reshape(-1,600,3)).detach().cpu().numpy().reshape(186,-1,24,3)
    import os
    os.makedirs(f"{save_dir}",exist_ok=True)
    for i in range(trans.shape[0]):
        pickle.dump({"smpl_trans":trans[i].detach().cpu().numpy(),"smpl_poses":pose[i].detach().cpu().numpy(),"full_pose":full_pose[i]},open(f"{save_dir}/{i}.pkl", "wb"))