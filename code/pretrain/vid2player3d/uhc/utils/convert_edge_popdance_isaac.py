import joblib
import numpy as np
import os
import sys
import argparse

sys.path.append(os.getcwd())

import isaacgym
import torch
from scipy.spatial.transform import Rotation as sRot
import yaml
from tqdm import tqdm

from uhc.smpllib.smpl_parser import SMPL_BONE_ORDER_NAMES as joint_names
from uhc.smpllib.smpl_local_robot import Robot as LocalRobot
from embodied_pose.utils.motion_lib import MotionLib
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from pytorch3d.transforms import (RotateAxisAngle, axis_angle_to_quaternion,
                                  quaternion_multiply,
                                  quaternion_to_axis_angle)
import pickle5

parser = argparse.ArgumentParser()
parser.add_argument('--edge_data', type=str, default="data/")
parser.add_argument('--out_dir', type=str, default="data/motion_lib/edge_pop/")
parser.add_argument('--num_seq', type=int, default=None)
parser.add_argument('--num_motion_libs', type=int, default=10)
args = parser.parse_args()

num_seq = args.num_seq
num_motion_libs = args.num_motion_libs

os.makedirs(args.out_dir, exist_ok=True)
meta_data = {
    "edge_data": args.edge_data,
    "num_seq": num_seq,
    "num_motion_libs": num_motion_libs
}
yaml.safe_dump(meta_data, open(f'{args.out_dir}/args.yml', 'w'))

edge_data = joblib.load(args.edge_data+"edge_processed_train_data.pkl")
info = joblib.load('data/misc/smpl_body_info.pkl')

bs,sq,channel = edge_data['q'].shape
edge_data['q'] = torch.from_numpy(edge_data['q']).reshape((bs, sq, -1, 3))

root_q = edge_data['q'][:, :, :1, :]  # sequence x 1 x 3
root_q_quat = axis_angle_to_quaternion(root_q)
rotation = torch.Tensor(
    [0.7071068, 0.7071068, 0, 0]
)  # 90 degrees about the x axis
root_q_quat = quaternion_multiply(rotation, root_q_quat)
root_q = quaternion_to_axis_angle(root_q_quat)
edge_data['q'][:, :, :1, :] = root_q

# don't forget to rotate the root position too 😩
pos_rotation = RotateAxisAngle(90, axis="X", degrees=True)
edge_data['pos'] = pos_rotation.transform_points(
    torch.tensor(edge_data['pos'])
) 
#get popdance data
pop_data = joblib.load(args.edge_data+"pop_processed_train_data.pkl")

bs,sq,channel = pop_data['q'].shape
pop_data['q'] = torch.from_numpy(pop_data['q']).reshape((bs, sq, -1, 3))

root_q = pop_data['q'][:, :, :1, :]  # sequence x 1 x 3
root_q_quat = axis_angle_to_quaternion(root_q)
rotation = torch.Tensor(
    [0.7071068, -0.7071068, 0, 0]
)  # 90 degrees about the x axis
root_q_quat = quaternion_multiply(rotation, root_q_quat)
root_q = quaternion_to_axis_angle(root_q_quat)
pop_data['q'][:, :, :1, :] = root_q

# don't forget to rotate the root position too 😩
pos_rotation = RotateAxisAngle(-90, axis="X", degrees=True)
pop_data['pos'] = pos_rotation.transform_points(
    torch.tensor(pop_data['pos'])
) 
pop_data['pos'][:,:,2] += 1
pop_data['pos'][:,:,1] -= 5

edge_data['pos'] = torch.cat([edge_data['pos'],pop_data['pos']],dim=0)
edge_data['q'] = torch.cat([edge_data['q'],pop_data['q']],dim=0)

# body_shapes
# all_beta = [x['beta'][:10] for x in edge_data.values()]
# _, index = np.unique([",".join([f"{x:.6f}" for x in beta]) for beta in all_beta], return_index=True)
# index.sort()
# beta_arr = [all_beta[i] for i in index]
# beta_mapping = dict()
# for i, beta in enumerate(beta_arr):
#     key = ",".join([f"{x:.6f}" for x in beta])
#     beta_mapping[key] = i
# print(f'edge data has {len(beta_mapping)} unique body shapes!')
# joblib.dump({'beta_arr': beta_arr, 'beta_mapping': beta_mapping}, f'{args.out_dir}/shape_data.pkl')

robot_cfg = {
    "mesh": True,
    "model": "smpl",
    "body_params": {},
    "joint_params": {},
    "geom_params": {},
    "actuator_params": {},
}

model_xml_path = f"/tmp/smpl/smpl_mesh_humanoid_v1_convert.xml"

smpl_local_robot = LocalRobot(
    robot_cfg,
    data_dir= "data/smpl",
    model_xml_path=model_xml_path
)


mujoco_joint_names = [
    'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee',
    'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax',
    'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder',
    'R_Elbow', 'R_Wrist', 'R_Hand'
]
smpl_2_mujoco = [
    joint_names.index(q) for q in mujoco_joint_names
    if q in joint_names
]

edge_full_motion_dict = {}
sequences = np.arange(edge_data['pos'].shape[0])
if num_seq is not None:
    sequences = sequences[:num_seq]

# seq_mapping = {seq_name.item(): seq_idx for seq_idx, seq_name in enumerate(sequences)}
motion_lib_seq_arr = np.array_split(sequences, num_motion_libs)
import ipdb;ipdb.set_trace()
seq_name_splits = {}
for i, seq_arr in enumerate(motion_lib_seq_arr):
    seq_name_splits[i] = seq_arr
# joblib.dump(seq_name_splits, f'{args.out_dir}/seq_name_splits.pkl')


for i, motion_lib_seqs in enumerate(motion_lib_seq_arr):
    
    motion_lib_input_dict = dict()

    for key_name in tqdm(motion_lib_seqs,desc=f"the {i} seq"):
        key_name = key_name.item()
        # smpl_data_entry = edge_data[key_name]
        # file_name = f"data/edge/singles/{key_name}.npy"
        seq_len = edge_data['pos'][key_name].shape[1]

        pose_aa = edge_data['q'][key_name].numpy().copy().reshape(sq,72)
        trans = edge_data['pos'][key_name].numpy().copy()
        
        beta = np.zeros(10)
        gender = "neutral"
        fps = 30.0

        if isinstance(gender, np.ndarray):
            gender = gender.item()
        if isinstance(gender, bytes):
            gender = gender.decode("utf-8")
        if gender == "neutral":
            gender_number = [0]
            smpl_parser = smpl_local_robot.smpl_parser_n
        elif gender == "male":
            gender_number = [1]
            smpl_parser = smpl_local_robot.smpl_parser_m
        elif gender == "female":
            gender_number = [2]
            smpl_parser = smpl_local_robot.smpl_parser_f
        else:
            import ipdb
            ipdb.set_trace()
            raise Exception("Gender Not Supported!!")

        batch_size = pose_aa.shape[0]
        pose_aa = np.concatenate([pose_aa[:, :66], np.zeros((batch_size, 6))], axis=1)  # TODO: need to extract correct handle rotations instead of zero
        pose_quat = sRot.from_rotvec(pose_aa.reshape(-1, 3)).as_quat().reshape(batch_size, 24, 4)[..., smpl_2_mujoco, :]
        smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None, ]), gender=gender_number)
        smpl_local_robot.write_xml()
        skeleton_tree = SkeletonTree.from_mjcf(model_xml_path)
        root_trans = trans + skeleton_tree.local_translation[0].numpy()
        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat),
            torch.from_numpy(root_trans),
            is_local=True)

        verts, joints = smpl_parser.get_joints_verts(
            pose=torch.from_numpy(pose_aa),
            th_betas=torch.from_numpy(beta[None, ]),
            th_trans=torch.from_numpy(trans)
        )

        # min_verts_h = verts[..., 2].min().item()
        min_verts_h = verts[..., 2].min(dim=-1)[0].mean().item()

        beta_key = ",".join([f"{x:.6f}" for x in beta])

        new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=fps)
        new_motion_out = new_motion.to_dict()
        new_motion_out['seq_name'] = key_name
        new_motion_out['seq_idx'] = key_name
        new_motion_out['trans'] = trans
        new_motion_out['root_trans'] = root_trans
        new_motion_out['pose_aa'] = pose_aa
        new_motion_out['beta'] = beta
        new_motion_out['beta_idx'] = 0
        new_motion_out['gender'] = gender
        new_motion_out['min_verts_h'] = min_verts_h
        new_motion_out['body_scale'] = 1.0
        new_motion_out['__name__'] = "SkeletonMotion"
        motion_lib_input_dict[key_name] = new_motion_out
        # import ipdb;ipdb.set_trace()
    motion_lib = MotionLib(motion_file=motion_lib_input_dict,
        dof_body_ids=info['dof_body_ids'],
        dof_offsets=info['dof_offsets'],
        key_body_ids=info['key_body_ids'],
        device='cpu',
        clean_up=True
    )

    torch.save(motion_lib, f"{args.out_dir}/mlib_part_{i:05d}.pth")

    del motion_lib_input_dict
    del motion_lib


smpl_local_robot.clean_up()
