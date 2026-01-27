import isaacgym
import torch
from torch import nn
from pathlib import Path
from typing import List, Dict
from colour import Color
from typing import Union
from loguru import logger
import os
from os import path as osp
import time
from tqdm.auto import tqdm
import functools
import joblib
import multiprocessing as mp


from scipy.spatial.transform import Rotation as sRot
from scipy.spatial.transform import RotationSpline 
import scipy.interpolate as interpolate

from trimesh import Trimesh
import numpy as np

from embodied_pose.players.im_player import ImitatorPlayer
from uhc.smpllib.smpl_local_robot import Robot as LocalRobot
from uhc.smpllib.smpl_parser import SMPL_Parser
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from embodied_pose.utils.motion_lib import MotionLib
from embodied_pose.utils.MDM.motion_process import recover_from_ric,process_file
# from MDM.visualize.simplify_loc2rot import joints2smpl
from embodied_pose.utils.MDM.skeleton import Skeleton
from embodied_pose.utils.MDM.smpl import SMPL
# from MDM.model.rotation2xyz import Rotation2xyz
from embodied_pose.utils.edge.quaternion import ax_from_6v, ax_to_6v
from support_path import *
try:
    from human_body_prior.body_model.body_model import BodyModel
    from human_body_prior.models.ik_engine import IK_Engine
    from human_body_prior.tools.omni_tools import create_list_chunks
    from human_body_prior.tools.omni_tools import copy2cpu as c2c
except:
    from human_body_prior.src.human_body_prior.body_model.body_model import BodyModel
    from human_body_prior.src.human_body_prior.models.ik_engine import IK_Engine
    from human_body_prior.src.human_body_prior.tools.omni_tools import create_list_chunks
    from human_body_prior.src.human_body_prior.tools.omni_tools import copy2cpu as c2c



class diff_player(ImitatorPlayer):
    def __init__(self,config):
        super().__init__(config)
        # self.mdm = self.task.mdm
        # self.joint2smpl = joints2smpl(num_frames=294,device_id=self.task.device_id, cuda=True if self.task.device_type == "cuda" or 
        #                                 self.task.device_type == "GPU" else False)
        # self.joint2smpl_120 = joints2smpl(num_frames=197,device_id=self.task.device_id, cuda=True if self.task.device_type == "cuda" or 
        #                                 self.task.device_type == "GPU" else False)
        self.skeleton = Skeleton(torch.from_numpy(t2m_raw_offsets),
                                    t2m_kinematic_chain,"cpu")
        example_id = "000021"
        example_data = np.load(os.path.join(f"./{ROOT_PATH}/data/misc", example_id + '.npy')) # (180, 52, 3)
        self.target_offset = self.skeleton.get_offsets_joints(torch.from_numpy(example_data[0]))
        # self.rot2xyz = Rotation2xyz(device='cpu')
        # self.faces = self.rot2xyz.smpl_model.faces
        self.gender_beta = torch.zeros(11)
        self.step = 1000
        self.save=True
        # self.ik_solver = nn.DataParallel(IK_solver()).to(self.device)
       
    def run(self):
        denoised_steps = [0]
        use_fast = True 
        motion_length = torch.tensor([self.mdm.n_frames]).to(self.device)
        # while True:
        prompt = input("your prompt...")
        args, model_kwargs, model, diffusion, data= self.mdm.args, self.mdm.model_kwargs, self.mdm.model, self.mdm.diffusion, self.mdm.data
        model_kwargs['y']['text'] = prompt
        batch_size = 1
        fps = 12.5 if args.dataset == 'kit' else 20
        
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(batch_size, device=self.device) * args.guidance_param
        sample_fn = diffusion.ddim_sample
        shape = (batch_size, self.mdm.model.njoints,self.mdm.model.nfeats,self.mdm.max_frames)
        img = torch.randn((100,)+shape[1:],device = self.device)[[0]]
        indices = list(range(diffusion.num_timesteps))[::-1]
        for i in tqdm(indices):
            self.step = i
            t = torch.tensor([i]*shape[0],device = self.device)
            with torch.no_grad():
                if i in denoised_steps:
                    denoised_fn = functools.partial(self.phys_proj,data)
                    if use_fast:
                        denoised_fn = functools.partial(self.phys_proj_fast,data,motion_length)
                else:
                    denoised_fn = None
                out = sample_fn(
                        model,
                        img,
                        t,
                        clip_denoised=False,
                        denoised_fn = denoised_fn,
                        model_kwargs = model_kwargs,
                        eta = 0.0
                    )
                img = out["sample"]
        self.get_results(data,img)
            
    def save_pkl(self,pose,betas,trans,i=0):
        out = dict()
        out.update({1:{'pred_cam':None,'betas':betas,'pose':pose,'trans':trans}})
        if not os.path.exists("./results"):
            os.makedirs("./results",exist_ok=True)
        joblib.dump(out,f'./results/pose_{i}.pkl')
    
    # def get_results(self, data, sample):
    #     smpl_parser = get_smpl_parser(0)
    #     n_joints = 22 
    #     sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
    #     mdm_jts_smpl_24 = recover_from_ric(sample, n_joints).cpu().numpy().squeeze()
    #     # trans = mdm_jts_smpl_24[:,0]
    #     motion_dict = IK_solver(mdm_jts_smpl_24)
    #     # motion_tensor, opt_dict = self.joint2smpl_120.joint2smpl(mdm_jts_smpl_24)
    #     verts, joints = smpl_parser.get_joints_verts(
    #         pose=torch.from_numpy(motion_dict["poses"][...,:72]),
    #         th_betas=self.gender_beta[1:][None, ],
    #         th_trans=torch.from_numpy(motion_dict['trans'])
    #     )
    #     # height = verts[..., 2].min().item()
    #     height = verts[..., 2].min(dim=-1)[0].mean().item()
    #     verts[..., 2] -= height
    #     # verts[..., 2] -= torch.min(verts[..., 2]).item()
    #     self.save_pkl(motion_dict["poses"][...,:72],self.gender_beta.numpy()
    #                   ,motion_dict["trans"].squeeze())
    #     smpl_model = SMPL().eval().to(self.device)
    #     for i in range(verts.shape[0]):
    #         mesh = Trimesh(vertices=verts[i].cpu().tolist(),faces=smpl_model.faces)
    #         mesh.export(f'./results/mesh_{i}.obj')
    
    # def get_results_from_jts(self, mdm_jts_smpl_24):
    #     smpl_parser = get_smpl_parser(0)
    #     trans = mdm_jts_smpl_24[:,0] #seq*1*3
    #     motion_dict = IK_solver(mdm_jts_smpl_24.astype(np.float32))
    #     # motion_tensor, opt_dict = self.joint2smpl_120.joint2smpl(mdm_jts_smpl_24)
    #     verts, joints = smpl_parser.get_joints_verts(
    #         pose=torch.from_numpy(motion_dict["poses"][...,:72]),
    #         th_betas=self.gender_beta[1:][None, ],
    #         th_trans=torch.from_numpy(motion_dict['trans'])
    #     )
    #     height = verts[..., 2].min(dim=-1)[0].mean().item()
    #     verts[..., 2] -= height
    #     smpl_model = SMPL().eval().to(self.device)
    #     for i in range(verts.shape[0]):
    #         mesh = Trimesh(vertices=verts[i].cpu().tolist(),faces=smpl_model.faces)
    #         mesh.export(f'./results/pos/mesh_{i}.obj')
    
    def get_results_from_rot(self,pose_aa,root,gender_beta,motion_length):
        smpl_parser = get_smpl_parser(int(gender_beta[0].item()))
        pose_quat = sRot.from_rotvec(pose_aa.reshape(-1,3)
                                     ).as_quat().reshape(root.shape[0],-1,24,4)[...,:22,:]
        for i in range(pose_aa.shape[0]):
            length = int(motion_length[i].item())
            root_origin,pose_aa_origin,_ = self.change_fps(root[i,:length],
                                                           pose_quat[i,:length],
                                                           1/30.0,1/20.0)
            verts,_ = smpl_parser.get_joints_verts(
                pose=torch.from_numpy(pose_aa_origin),
                th_betas = gender_beta[1:][None,],
                th_trans = root_origin)
            height = verts[..., 2].min(dim=-1)[0].mean().item()
            verts[..., 2] -= height
            smpl_model = SMPL().eval().to(self.device)
            self.save_pkl(pose_aa_origin,self.gender_beta.numpy()
                      ,root_origin,i)
            for j in range(verts.shape[0]):
                mesh = Trimesh(vertices=verts[j].cpu().tolist(),faces=smpl_model.faces)
                os.makedirs(f'./results/rot/{i}',exist_ok=True)
                mesh.export(f'./results/rot/{i}/mesh_{j}.obj')
    
    def change_fps(self, trans, pose_quat, dt, target_dt):
        """
        No Batch dim

        Args:
            trans (array): _description_
            pose_quat (array): _description_
            dt (float): _description_
            target_dt (float): _description_

        Returns:
            tensor: trans
            array: pose_aa
            array: pose_quat
        """
        
        pose_quat = np.vstack((pose_quat, pose_quat[-1].reshape(-1, 22, 4)))
        trans = np.vstack((trans, trans[-1].reshape(-1, 3)))
        pose_quat = np.stack(
            [get_rot(pose_quat[:, i, :], dt, target_dt) for i in range(pose_quat.shape[1])], axis=1
        )
        hand = np.array([[0.0,0.0,0.0,1.0]]).repeat(pose_quat.shape[0],0)
        pose_quat = np.concatenate([pose_quat,hand[:,None,:],hand[:,None,:]],axis=1)
        pose_aa = sRot.from_quat(pose_quat.reshape(-1,4)).as_rotvec().reshape(-1,72)
        trans = torch.from_numpy(interpolate_trans(trans, dt, target_dt))
        return trans,pose_aa,pose_quat
    
    # def phys_proj(self,data,sample):
    #     # batch_size = 1
    #     n_joints = 22 if sample.shape[1] == 263 else 21
    #     sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
    #     mdm_jts_smpl_24 = recover_from_ric(sample, n_joints).squeeze().cpu().numpy()
        
    #     mdm_jts_smpl_24 = fps_20_to_30(mdm_jts_smpl_24)
        
    #     mat_to_isaac = sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False).as_matrix()
    #     mdm_jts_smpl_24 = np.matmul(mdm_jts_smpl_24, mat_to_isaac)

        
    #     motion_tensor, opt_dict = self.joint2smpl.joint2smpl(mdm_jts_smpl_24)
        
    #     pose_aa = opt_dict["pose"].reshape(-1, 72).cpu().numpy()
    #     trans = torch.from_numpy(mdm_jts_smpl_24[:,0]) #seq_len*3
    #     hand = np.array([[0.0,0.0,0.0]]).repeat(pose_aa.shape[0],0)
    #     pose_aa = np.concatenate([pose_aa[:,:66],hand,hand],axis=1) #seq_len*72
    #     pose_quat = sRot.from_rotvec(pose_aa.reshape(-1, 3)).as_quat().reshape(-1, 24, 4)
    #     pose_quat = torch.from_numpy(pose_quat)[..., smpl_2_mujoco, :]

    #     trans,smpl_parser = fix_height_smpl_vanilla(
    #         pose_aa=torch.from_numpy(pose_aa),
    #         th_trans=trans,
    #         th_betas=self.gender_beta[1:],
    #         gender= self.gender_beta[0].item(),
    #         seq_name=None       
    #     )
    #     ################################
    #     # trans: torch, seq_len*3
    #     # pose_aa: numpy, seq_len*72, smpl
    #     # pose_quat: torch, seq_len*24*4, mujoco
    #     # gender_beta: torch , 11
    #     ################################
    #     fps = 30.0
    #     motion_lib_input_dict = dict()
    #     info = joblib.load('data/misc/smpl_body_info.pkl')
    #     robot_cfg = {
    #         "mesh": True,
    #         "model": "smpl",
    #         "body_params": {},
    #         "joint_params": {},
    #         "geom_params": {},
    #         "actuator_params": {},
    #     }

    #     model_xml_path = f"/tmp/smpl/smpl_mesh_humanoid_v1_convert.xml"

    #     smpl_local_robot = LocalRobot(
    #         robot_cfg,
    #         data_dir= "data/smpl",
    #         model_xml_path=model_xml_path
    #     )
    #     smpl_local_robot.load_from_skeleton(betas=self.gender_beta[1:][None, ], 
    #                                         gender=[self.gender_beta[0].item()])
    #     smpl_local_robot.write_xml()
    #     skeleton_tree = SkeletonTree.from_mjcf(model_xml_path)
    #     root_trans = trans + skeleton_tree.local_translation[0]
        
    #     new_sk_state = SkeletonState.from_rotation_and_root_translation(
    #         skeleton_tree,
    #         pose_quat,
    #         root_trans,
    #         is_local=True)
        
    #     verts, joints = smpl_parser.get_joints_verts(
    #         pose=torch.from_numpy(pose_aa),
    #         th_betas=self.gender_beta[1:][None, ],
    #         th_trans=trans
    #     )
    #     # min_verts_h = verts[..., 2].min().item()
    #     min_verts_h = verts[..., 2].min(dim=-1)[0].mean().item()
    #     # min_verts_h = torch.min(verts[0, :, 2])
        

    #     new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=fps)
    #     new_motion_out = new_motion.to_dict()
    #     new_motion_out['seq_name'] = "mdm"
    #     new_motion_out['seq_idx'] = 0
    #     new_motion_out['trans'] = trans.numpy()
    #     new_motion_out['root_trans'] = root_trans.numpy()
    #     new_motion_out['pose_aa'] = pose_aa
    #     new_motion_out['beta'] = self.gender_beta[1:].numpy()
    #     new_motion_out['beta_idx'] = 0
    #     new_motion_out['gender'] = ["neutral","male","female"][int(self.gender_beta[0].item())]
    #     new_motion_out['min_verts_h'] = min_verts_h
    #     new_motion_out['body_scale'] = 1.0
    #     new_motion_out['__name__'] = "SkeletonMotion"
    #     motion_lib_input_dict['mdm'] = new_motion_out

    #     motion_lib = MotionLib(motion_file=motion_lib_input_dict,
    #         dof_body_ids=info['dof_body_ids'],
    #         dof_offsets=info['dof_offsets'],
    #         key_body_ids=info['key_body_ids'],
    #         device= self.device,
    #         clean_up=True
    #     )
        
    #     self.task._motion_lib = motion_lib

    #     pose_quat_new, root_trans_new, joint_position = self.imitation(root_trans.shape[0])
    #     pose_quat = pose_quat_new.cpu().numpy()[...,mujoco_2_smpl,:]
    #     pose_aa = sRot.from_quat(pose_quat.reshape(-1, 4)).as_rotvec().reshape(-1, 72)
    #     root_trans = root_trans_new.cpu().numpy()
    #     # joint_position = joint_position.cpu().numpy()[...,mujoco_2_smpl,:]
        
    #     self.get_results_from_rot(pose_aa,root_trans,self.gender_beta)
        
    #     # mat_to_mdm = sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False).as_matrix()
    #     # mdm_jts_smpl_24 = np.matmul(joint_position, mat_to_mdm.T)        
    #     # verts, joints = smpl_parser.get_joints_verts(
    #     #     pose=torch.from_numpy(pose_aa),
    #     #     th_betas=gender_beta[1:][None, ],
    #     #     th_trans=torch.from_numpy(root_trans)
    #     #     )

    #     # min_verts_h = verts[..., 2].min(dim=-1)[0].mean().item()
    #     # mdm_jts_smpl_24[...,1] -= min_verts_h
    #     # mdm_jts_smpl_24[...,1] -= torch.min(verts[:,:,2]).item()
    #     mdm_jts_smpl_24 = fps_30_to_20(mdm_jts_smpl_24)
    #     data_vec, global_positions,\
    #     positions, l_velocity = process_file(positions=mdm_jts_smpl_24[:,:22,:],
    #                                             feet_thre=0.002,tgt_offsets=self.target_offset,
    #                                             face_joint_indx=[2, 1, 17, 16],
    #                                             fid_l=[7,10],fid_r=[8,11],
    #                                             n_raw_offsets= torch.from_numpy(t2m_raw_offsets),
    #                                             kinematic_chain=t2m_kinematic_chain,skeleton=self.skeleton)
    #     data_vec = torch.from_numpy(data_vec).unsqueeze(0).unsqueeze(0).float()
    #     data_vec = ((data_vec-data.dataset.t2m_dataset.mean)/data.dataset.t2m_dataset.std).permute(0,3,1,2)
    #     self.get_results(data,data_vec)
    #     return data_vec.to(self.device)
    @staticmethod
    def process_motion_lib(pose_aa_gt,trans_gt,i,gender_beta,skeleton_tree):
         ################################
        # trans: tensor, seq_len*3
        # pose_aa: numpy, seq_len*24*3, smpl
        # pose_quat: numpy, seq_len*24*4, mujoco
        # gender_beta: tensor , 11
        ################################
        fps=30
        pose_quat_gt = sRot.from_rotvec(pose_aa_gt.reshape(-1, 3)).as_quat().reshape(-1, 22, 4)
        hand = np.array([[0.0,0.0,0.0,1.0]]).repeat(pose_quat_gt.shape[0],0)
        pose_quat_gt = np.concatenate([pose_quat_gt,hand[:,None,:],hand[:,None,:]],axis=1)
        pose_aa_gt =  sRot.from_quat(pose_quat_gt.reshape(-1,4)).as_rotvec().reshape(-1,72)
        
        pose_quat_gt = pose_quat_gt[..., smpl_2_mujoco, :]
        
        # trans_gt,_ = fix_height_smpl_vanilla(
        #     pose_aa=torch.from_numpy(pose_aa_gt),
        #     th_trans=trans_gt,
        #     th_betas=gender_beta[1:],
        #     gender= gender_beta[0].item(),
        #     seq_name=None       
        # )
        smpl_parser = get_smpl_parser(int(gender_beta[0].item()))
        
        root_trans_gt = trans_gt + skeleton_tree.local_translation[0]
        
        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat_gt),
            root_trans_gt,
            is_local=True)
        
        verts_gt, _ = smpl_parser.get_joints_verts(
            pose=torch.from_numpy(pose_aa_gt),
            th_betas=gender_beta[1:][None, ],
            th_trans=trans_gt
        )
        # min_verts_h = verts[..., 2].min().item()
        min_verts_h = verts_gt[..., 2].min(dim=-1)[0].mean().item()
        # min_verts_h = torch.min(verts_gt[0, :, 2])
        # import ipdb;ipdb.set_trace()
        # min_verts_h = torch.zeros(1)
        motion_lib_input_dict = {}
        new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=fps)
        new_motion_out = new_motion.to_dict()
        new_motion_out['seq_name'] = str(i)
        new_motion_out['seq_idx'] = i
        new_motion_out['trans'] = trans_gt.numpy()
        new_motion_out['root_trans'] = root_trans_gt.numpy()
        new_motion_out['pose_aa'] = pose_aa_gt
        new_motion_out['beta'] = gender_beta[1:].numpy()
        new_motion_out['beta_idx'] = 0
        new_motion_out['gender'] = ["neutral","male","female"][int(gender_beta[0].item())]
        new_motion_out['min_verts_h'] = min_verts_h
        new_motion_out['body_scale'] = 1.0
        new_motion_out['__name__'] = "SkeletonMotion"
        motion_lib_input_dict[str(i)] = new_motion_out
        # print("done!",end="\r")
        return motion_lib_input_dict


    def phys_proj_fast(self,data,motion_length_gt,sample):
        """
        Batch operation of motion

        Args:
            data (dataloader): use it to get dataset-related data 
            sample (torch.tensor(cuda)): B*feat_dim*seq_len*joint_num
            motion_length (torch.tensor(cuda)): #TODO

        Returns:
            sample (torch.tensor(cuda)): B*feat_dim*seq_len*joint_num
            motion_length : #TODO
        """
        n_joints = 22 if sample.shape[1] == 263 else 21
        dt = 1/20.0
        target_dt = 1/30.0
        gender_beta = self.gender_beta #use a template gender beta
        constant_length = sample.cpu().shape[3]
        upsample_max_length = -1
        try:
            sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
        except:
            sample = data.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
        mdm_jts_smpl_24_gt = np.squeeze(recover_from_ric(sample, n_joints).cpu().numpy(),axis=1)
        
        
        # mdm_jts_smpl_24_gt = mdm_jts_smpl_24_gt.transpose((1,0,2,3)).repeat(20,axis=1)
        # motion_length_gt = np.array([20]*mdm_jts_smpl_24_gt.shape[0])
        # print(mdm_jts_smpl_24_gt.shape)

        # self.get_results_from_jts(mdm_jts_smpl_24_gt[0])
        mdm_jts_smpl_24_gt_flat = []
        for i in range(mdm_jts_smpl_24_gt.shape[0]):
            mdm_jts_smpl_24_gt_flat.append(mdm_jts_smpl_24_gt[i,:int(motion_length_gt[i].item())])
        mdm_jts_smpl_24_gt_flat = np.concatenate(mdm_jts_smpl_24_gt_flat,axis=0)
        #transform y up to z up
        # mat_to_isaac = sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False).as_matrix()
        # mdm_jts_smpl_24 = np.matmul(mdm_jts_smpl_24, mat_to_isaac)
        out = self.ik_solver(mdm_jts_smpl_24_gt_flat)
        pose_aa_gt_flat = out[...,3:69].cpu().numpy()
        trans_gt_flat =out[...,:3].cpu().numpy()
    
        fps = 30.0
        motion_lib_input_dict = dict()
        info = joblib.load('data/misc/smpl_body_info.pkl')
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
        smpl_local_robot.load_from_skeleton(betas=gender_beta[1:][None, ], gender=[gender_beta[0].item()])
        smpl_local_robot.write_xml()
        skeleton_tree = SkeletonTree.from_mjcf(model_xml_path)
        

        for i in tqdm(range(motion_length_gt.shape[0])):
            ################################
            # trans: tensor, seq_len*3
            # pose_aa: numpy, seq_len*72, smpl
            # pose_quat: numpy, seq_len*24*4, mujoco
            # gender_beta: tensor , 11
            ################################
            pose_aa_gt = pose_aa_gt_flat[sum(motion_length_gt[:i]):sum(motion_length_gt[:i+1])].reshape(-1,22,3)
            pose_quat_gt = sRot.from_rotvec(pose_aa_gt.reshape(-1, 3)).as_quat().reshape(-1, 22, 4)
            trans_gt = trans_gt_flat[sum(motion_length_gt[:i]):sum(motion_length_gt[:i+1])]
            
            trans_gt,pose_aa_gt,pose_quat_gt = self.change_fps(trans_gt,pose_quat_gt,dt,target_dt)
            pose_quat_gt = pose_quat_gt[..., smpl_2_mujoco, :]
            upsample_max_length = max(upsample_max_length,trans_gt.shape[0])
            # trans_gt,_ = fix_height_smpl_vanilla(
            #     pose_aa=torch.from_numpy(pose_aa_gt),
            #     th_trans=trans_gt,
            #     th_betas=gender_beta[1:],
            #     gender= gender_beta[0].item(),
            #     seq_name=None       
            # )
            smpl_parser = get_smpl_parser(int(gender_beta[0].item()))
            
            root_trans_gt = trans_gt + skeleton_tree.local_translation[0]
            
            new_sk_state = SkeletonState.from_rotation_and_root_translation(
                skeleton_tree,
                torch.from_numpy(pose_quat_gt),
                root_trans_gt,
                is_local=True)
            
            verts_gt, _ = smpl_parser.get_joints_verts(
                pose=torch.from_numpy(pose_aa_gt),
                th_betas=gender_beta[1:][None, ],
                th_trans=trans_gt
            )
            # min_verts_h = verts[..., 2].min().item()
            min_verts_h = verts_gt[..., 2].min(dim=-1)[0].mean().item()
            # min_verts_h = torch.min(verts_gt[0, :, 2])
            # import ipdb;ipdb.set_trace()
            # min_verts_h = torch.zeros(1)

            new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=fps)
            new_motion_out = new_motion.to_dict()
            new_motion_out['seq_name'] = str(i)
            new_motion_out['seq_idx'] = i
            new_motion_out['trans'] = trans_gt.numpy()
            new_motion_out['root_trans'] = root_trans_gt.numpy()
            new_motion_out['pose_aa'] = pose_aa_gt
            new_motion_out['beta'] = gender_beta[1:].numpy()
            new_motion_out['beta_idx'] = 0
            new_motion_out['gender'] = ["neutral","male","female"][int(gender_beta[0].item())]
            new_motion_out['min_verts_h'] = min_verts_h
            new_motion_out['body_scale'] = 1.0
            new_motion_out['__name__'] = "SkeletonMotion"
            motion_lib_input_dict[str(i)] = new_motion_out

        motion_lib = MotionLib(motion_file=motion_lib_input_dict,
            dof_body_ids=info['dof_body_ids'],
            dof_offsets=info['dof_offsets'],
            key_body_ids=info['key_body_ids'],
            device= self.device,
            clean_up=True
        )
        if self.step==0:
            torch.save(motion_lib, f"test/mlib_part_0.pth")
        self.task._motion_lib = motion_lib

        pose_quat_batch, root_trans_batch, joint_position, motion_length_batch = self.imitation(upsample_max_length,motion_length_gt.shape[0])
        motion_length_pred = motion_length_batch.cpu().numpy()
        pose_quat_pred = pose_quat_batch.cpu().numpy()[...,mujoco_2_smpl,:]
        pose_aa_pred = sRot.from_quat(pose_quat_pred.reshape(-1, 4)).as_rotvec().reshape(pose_quat_pred.shape[0],-1 ,72)
        root_trans_pred = root_trans_batch.cpu().numpy()
        # joint_position = joint_position.cpu().numpy()[...,mujoco_2_smpl,:]
        if self.step ==0:
            self.get_results_from_rot(pose_aa_pred,root_trans_pred,motion_length_pred)
        pose_aa_flat = []
        root_trans_flat = []
        motion_length_pred_origin = []
        for i in range(pose_aa_pred.shape[0]):
            length = 182#int(motion_length_pred[i].item())
            root_trans_origin,pose_aa_origin,_ = self.change_fps(root_trans_pred[i,:length],
                                                            pose_quat_pred[i,:length,:22],
                                                            target_dt,dt)
            motion_length_pred_origin.append(root_trans_origin.shape[0])
            root_trans_flat.append(root_trans_origin)
            pose_aa_flat.append(pose_aa_origin)
        pose_aa_flat = np.concatenate(pose_aa_flat,axis=0)
        root_trans_flat = torch.concatenate(root_trans_flat,dim=0) 
           
        verts_pred_flat, mdm_jts_smpl_24_pred_flat = smpl_parser.get_joints_verts(
            pose=torch.from_numpy(pose_aa_flat.reshape(-1,72)),
            th_betas=gender_beta[1:][None, ],
            th_trans=root_trans_flat.reshape(-1,3)
            )
        
        mat_to_mdm = sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False).as_matrix()
        mdm_jts_smpl_24_pred_flat = np.matmul(mdm_jts_smpl_24_pred_flat, mat_to_mdm.T)
        data_vec_batch = []
        for i in tqdm(range(len(motion_length_pred_origin))):
            verts_pred = verts_pred_flat[sum(motion_length_pred_origin[:i]):sum(motion_length_pred_origin[:i+1])]
            mdm_jts_smpl_24_pred = mdm_jts_smpl_24_pred_flat[sum(motion_length_pred_origin[:i]):sum(motion_length_pred_origin[:i+1])]
            height = verts_pred[..., 2].min(dim=-1)[0].mean().item()
            mdm_jts_smpl_24_pred[..., 1] -= height
        # min_verts_h = verts[..., 2].min(dim=-1)[0].mean().item()
        # mdm_jts_smpl_24[...,1] -= min_verts_h
        # mdm_jts_smpl_24[...,1] -= torch.min(verts[:,:,2]).item()
        # mdm_jts_smpl_24_pred = fps_30_to_20(mdm_jts_smpl_24_pred)
        # self.get_results_from_jts(mdm_jts_smpl_24)
            mdm_jts_smpl_24_pred = np.concatenate([mdm_jts_smpl_24_pred[...,:22,:],
                                                   mdm_jts_smpl_24_gt[i,
                                                    motion_length_pred_origin[i]:]],axis=0)
            if mdm_jts_smpl_24_pred.shape[0] != constant_length+1:
                mdm_jts_smpl_24_pred = np.concatenate([mdm_jts_smpl_24_pred,
                                                       mdm_jts_smpl_24_pred[[-1]]],axis=0)
            data_vec, global_positions,\
            positions, l_velocity = process_file(positions=mdm_jts_smpl_24_pred[:,:22,:],
                                                    feet_thre=0.002,tgt_offsets=self.target_offset,
                                                    face_joint_indx=[2, 1, 17, 16],
                                                    fid_l=[7,10],fid_r=[8,11],
                                                    n_raw_offsets= torch.from_numpy(t2m_raw_offsets),
                                                    kinematic_chain=t2m_kinematic_chain,skeleton=self.skeleton)
            data_vec_batch.append(data_vec[None,:,:])
        data_vec = np.concatenate(data_vec_batch,axis=0)
        data_vec = torch.from_numpy(data_vec).unsqueeze(1).float()
        try:
            data_vec = ((data_vec-data.dataset.t2m_dataset.mean)/data.dataset.t2m_dataset.std).permute(0,3,1,2)
        except:
            data_vec = ((data_vec-data.t2m_dataset.mean)/data.t2m_dataset.std).permute(0,3,1,2)
        return data_vec.to(self.device)
    
    def phys_proj_bailando(self,sample):
        """
        Batch operation of motion

        Args:
            sample: b*t*72

        Returns:
            reward
        """
        print("Start_compute!")
        b,t = sample.shape[:2]
        if len(sample.shape) == 3:
            sample = sample.view(b,t,24,3)
            
        if b < 128:
            times = int(128/b) + 1
            sample = torch.concatenate([sample]*times,dim = 0)[:128]
        # dt = 1/60.0
        # target_dt = 1/30.0
        gender_beta = self.gender_beta #use a template gender beta
        # constant_length = sample.cpu().shape[1]


        mdm_jts_smpl_24_gt_flat = sample[:,::2].view(int(128*t/2),24,3)
        #transform y up to z up
        # mat_to_isaac = sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False).as_matrix()
        # mdm_jts_smpl_24 = np.matmul(mdm_jts_smpl_24, mat_to_isaac)
        start_time = time.time()
        out = self.ik_solver(mdm_jts_smpl_24_gt_flat)
        print(f"cost time:{time.time()-start_time}")
        # self.get_results_from_rot(motion_dict['poses'][...,:72].reshape(b,int(t/2),24,3),
        #                           motion_dict['trans'].reshape(b,int(t/2),3),
        #                           gender_beta,
        #                           np.array([int(t/2)]*b))
        # import ipdb; ipdb.set_trace()
        pose_aa_gt_flat = out[...,3:69].cpu().numpy().reshape(128,int(t/2),22,3)
        trans_gt_flat =out[...,:3].cpu().reshape(128,int(t/2),3)
    
        motion_lib_input_dict = dict()
        info = joblib.load(f'{ROOT_PATH}/data/misc/smpl_body_info.pkl')
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
            data_dir= f"{ROOT_PATH}/data/smpl",
            model_xml_path=model_xml_path
        )
        smpl_local_robot.load_from_skeleton(betas=gender_beta[1:][None, ], gender=[gender_beta[0].item()])
        smpl_local_robot.write_xml()
        skeleton_tree = SkeletonTree.from_mjcf(model_xml_path)
        with joblib.Parallel(n_jobs=-1) as executor:
            result = executor(joblib.delayed(diff_player.process_motion_lib)(pose_aa_gt_flat[i],
                                                                             trans_gt_flat[i],
                                                                             i,gender_beta,
                                                                             skeleton_tree) for i in range(128))
            # import ipdb;ipdb.set_trace()
            for future in result:
                motion_lib_input_dict.update(future)    

        motion_lib = MotionLib(motion_file=motion_lib_input_dict,
            dof_body_ids=info['dof_body_ids'],
            dof_offsets=info['dof_offsets'],
            key_body_ids=info['key_body_ids'],
            device= self.device,
            clean_up=True
        )
        # if self.step==0:
        #     torch.save(motion_lib, f"test/mlib_part_0.pth")
        self.task._motion_lib = motion_lib

        reward = self.imitation(t//2,128,return_reward = True)
        reward = reward.view(128,t//8,4).mean(dim=-1).squeeze()[:b]
        return reward
    
    def phys_proj_edge(self,normalizer,motion_length_gt,sample,return_reward=False,device=None,multi_process = "thread",accelerator = None):
        """
        Batch operation of motion

        Args:
            sample (torch.tensor(cuda)): B*feat_dim*seq_len*joint_num
            motion_length (torch.tensor(cuda)): #TODO

        Returns:
            sample (torch.tensor(cuda)): B*feat_dim*seq_len*joint_num
            motion_length : #TODO
        """
        if device==None:
            device = self.device
        else:
            self.device = device
        sample = sample.detach().cpu()
        B=22
        n_joints = 24
        gender_beta = self.gender_beta #use a template gender beta
        b, constant_length = sample.shape[:2]
        sample = normalizer.unnormalize(sample)
        if b < B:
            times = int(B/b) + 1
            sample = torch.concatenate([sample]*times,dim = 0)[:B]
        trans = sample[...,4:7] #b*l*3
        pose_6d = sample[...,7:].reshape(B,constant_length,n_joints,6)
        pose_aa  = ax_from_6v(pose_6d)#b*l*24*3
        
    
        fps = 30.0
        motion_lib_input_dict = dict()
        model_xml_path = f"/tmp/smpl/smpl_mesh_humanoid_v1_convert.xml"
        info = joblib.load(f'{ROOT_PATH}/data/misc/smpl_body_info.pkl')
        if accelerator != None:
            # if accelerator.is_main_process:
            model_xml_path = f"/tmp/smpl/smpl_mesh_humanoid_v1_convert{str(accelerator.device)[-1]}.xml"
            robot_cfg = {
                "mesh": True,
                "model": "smpl",
                "body_params": {},
                "joint_params": {},
                "geom_params": {},
                "actuator_params": {},
            }

            smpl_local_robot = LocalRobot(
                robot_cfg,
                data_dir= f"{ROOT_PATH}/data/smpl",
                model_xml_path=model_xml_path
            )
            smpl_local_robot.load_from_skeleton(betas=gender_beta[1:][None, ], gender=[gender_beta[0].item()])
            smpl_local_robot.write_xml()
            # accelerator.wait_for_everyone()
        else:
            robot_cfg = {
                "mesh": True,
                "model": "smpl",
                "body_params": {},
                "joint_params": {},
                "geom_params": {},
                "actuator_params": {},
            }

            smpl_local_robot = LocalRobot(
                robot_cfg,
                data_dir= f"{ROOT_PATH}/data/smpl",
                model_xml_path=model_xml_path
            )
            smpl_local_robot.load_from_skeleton(betas=gender_beta[1:][None, ], gender=[gender_beta[0].item()])
            smpl_local_robot.write_xml()
        skeleton_tree = SkeletonTree.from_mjcf(model_xml_path)
        

        # for i in tqdm(range(b)):
        #     ################################
        #     # trans: tensor, seq_len*3
        #     # pose_aa: numpy, seq_len*72, smpl
        #     # pose_quat: numpy, seq_len*24*4, mujoco
        #     # gender_beta: tensor , 11
        #     ################################
        #     pose_aa_gt = pose_aa[i].numpy()
        #     pose_quat_gt = sRot.from_rotvec(pose_aa_gt.reshape(-1, 3)).as_quat().reshape(-1, 24, 4)
        #     trans_gt = trans[i]
            
        #     pose_quat_gt = pose_quat_gt[..., smpl_2_mujoco, :]
        #     # trans_gt,_ = fix_height_smpl_vanilla(
        #     #     pose_aa=torch.from_numpy(pose_aa_gt),
        #     #     th_trans=trans_gt,
        #     #     th_betas=gender_beta[1:],
        #     #     gender= gender_beta[0].item(),
        #     #     seq_name=None       
        #     # )
        #     smpl_parser = get_smpl_parser(int(gender_beta[0].item()))
            
        #     root_trans_gt = trans_gt + skeleton_tree.local_translation[0]
            
        #     new_sk_state = SkeletonState.from_rotation_and_root_translation(
        #         skeleton_tree,
        #         torch.from_numpy(pose_quat_gt),
        #         root_trans_gt,
        #         is_local=True)
            
        #     verts_gt, _ = smpl_parser.get_joints_verts(
        #         pose=torch.from_numpy(pose_aa_gt),
        #         th_betas=gender_beta[1:][None, ],
        #         th_trans=trans_gt
        #     )
        #     # min_verts_h = verts[..., 2].min().item()
        #     min_verts_h = verts_gt[..., 2].min(dim=-1)[0].mean().item()
        #     # min_verts_h = torch.min(verts_gt[0, :, 2])
        #     # import ipdb;ipdb.set_trace()
        #     # min_verts_h = torch.zeros(1)

        #     new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=fps)
        #     new_motion_out = new_motion.to_dict()
        #     new_motion_out['seq_name'] = str(i)
        #     new_motion_out['seq_idx'] = i
        #     new_motion_out['trans'] = trans_gt.numpy()
        #     new_motion_out['root_trans'] = root_trans_gt.numpy()
        #     new_motion_out['pose_aa'] = pose_aa_gt
        #     new_motion_out['beta'] = gender_beta[1:].numpy()
        #     new_motion_out['beta_idx'] = 0
        #     new_motion_out['gender'] = ["neutral","male","female"][int(gender_beta[0].item())]
        #     new_motion_out['min_verts_h'] = min_verts_h
        #     new_motion_out['body_scale'] = 1.0
        #     new_motion_out['__name__'] = "SkeletonMotion"
        #     motion_lib_input_dict[str(i)] = new_motion_out
        if multi_process != None:
            if multi_process=="thread":
                with joblib.Parallel(n_jobs=-1,backend="threading") as executor:
                    result = executor(joblib.delayed(diff_player.process_motion_lib)(pose_aa[i,:,:22],
                                                                                    trans[i],
                                                                                    i,gender_beta,
                                                                                    skeleton_tree) for i in range(B))
                    # import ipdb;ipdb.set_trace()
                    for future in result:
                        motion_lib_input_dict.update(future)
            else:
                with joblib.Parallel(n_jobs=-1) as executor:
                    result = executor(joblib.delayed(diff_player.process_motion_lib)(pose_aa[i,:,:22],
                                                                                    trans[i],
                                                                                    i,gender_beta,
                                                                                    skeleton_tree) for i in range(B))
                    # import ipdb;ipdb.set_trace()
                    for future in result:
                        motion_lib_input_dict.update(future)
        else:
            for i in tqdm(range(B)):
                motion_lib_input_dict.update(diff_player.process_motion_lib(pose_aa[i,:,:22],
                                                trans[i],
                                                i,gender_beta,
                                                skeleton_tree))
        
        motion_lib = MotionLib(motion_file=motion_lib_input_dict,
            dof_body_ids=info['dof_body_ids'],
            dof_offsets=info['dof_offsets'],
            key_body_ids=info['key_body_ids'],
            device= device,
            clean_up=True
        )
        # if self.save:
        #     torch.save(motion_lib, f"test/mlib_part_0.pth")
        self.task._motion_lib = motion_lib
        if return_reward==True:
            reward = self.imitation(constant_length,B,True)
            return reward[:b]
        pose_quat_batch, root_trans_batch, joint_position, motion_length_batch = self.imitation(constant_length,B,False)
        joint_position = joint_position[...,mujoco_2_smpl,:].cpu()
        # import ipdb;ipdb.set_trace()
        feet = joint_position[:,:, (7, 8, 10, 11)]
        feetv = np.zeros((b,constant_length,4))
        feetv = np.linalg.norm(feet[:,1:] - feet[:,:-1], axis=-1)
        contact = torch.from_numpy(feetv < 0.01)
        motion_length_pred = motion_length_batch.cpu().numpy()
        pose_quat_pred = pose_quat_batch.cpu().numpy()[...,mujoco_2_smpl,:]
        pose_aa_pred = sRot.from_quat(pose_quat_pred.reshape(-1, 4)).as_rotvec().reshape(pose_quat_pred.shape[0],-1 ,24,3)
        pose_6d_pred = ax_to_6v(torch.from_numpy(pose_aa_pred))
        root_trans_pred = root_trans_batch.cpu()
        # joint_position = joint_position.cpu().numpy()[...,mujoco_2_smpl,:]
        # if self.step ==0:
        #     self.get_results_from_rot(pose_aa_pred,root_trans_pred,gender_beta,motion_length_pred)
        data_vec = torch.concatenate([contact,root_trans_pred[:,:constant_length],
                                   pose_6d_pred[:,:constant_length].reshape(b,constant_length,-1)],dim=-1)

        # pose_aa_flat = []
        # root_trans_flat = []
        # motion_length_pred_origin = []
        # for i in range(pose_aa_pred.shape[0]):
        #     length = int(motion_length_pred[i].item())
        #     root_trans_origin,pose_aa_origin,_ = self.change_fps(root_trans_pred[i,:length],
        #                                                     pose_quat_pred[i,:length,:22],
        #                                                     target_dt,dt)
        #     motion_length_pred_origin.append(root_trans_origin.shape[0])
        #     root_trans_flat.append(root_trans_origin)
        #     pose_aa_flat.append(pose_aa_origin)
        # pose_aa_flat = np.concatenate(pose_aa_flat,axis=0)
        # root_trans_flat = torch.concatenate(root_trans_flat,dim=0) 
           
        # verts_pred_flat, mdm_jts_smpl_24_pred_flat = smpl_parser.get_joints_verts(
        #     pose=torch.from_numpy(pose_aa_flat.reshape(-1,72)),
        #     th_betas=gender_beta[1:][None, ],
        #     th_trans=root_trans_flat.reshape(-1,3)
        #     )
        
        # mat_to_mdm = sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False).as_matrix()
        # mdm_jts_smpl_24_pred_flat = np.matmul(mdm_jts_smpl_24_pred_flat, mat_to_mdm.T)
        # data_vec_batch = []
        # for i in tqdm(range(len(motion_length_pred_origin))):
        #     verts_pred = verts_pred_flat[sum(motion_length_pred_origin[:i]):sum(motion_length_pred_origin[:i+1])]
        #     mdm_jts_smpl_24_pred = mdm_jts_smpl_24_pred_flat[sum(motion_length_pred_origin[:i]):sum(motion_length_pred_origin[:i+1])]
        #     height = verts_pred[..., 2].min(dim=-1)[0].mean().item()
        #     mdm_jts_smpl_24_pred[..., 1] -= height
        # # min_verts_h = verts[..., 2].min(dim=-1)[0].mean().item()
        # # mdm_jts_smpl_24[...,1] -= min_verts_h
        # # mdm_jts_smpl_24[...,1] -= torch.min(verts[:,:,2]).item()
        # # mdm_jts_smpl_24_pred = fps_30_to_20(mdm_jts_smpl_24_pred)
        # # self.get_results_from_jts(mdm_jts_smpl_24)
        #     mdm_jts_smpl_24_pred = np.concatenate([mdm_jts_smpl_24_pred[...,:22,:],
        #                                            mdm_jts_smpl_24_gt[i,
        #                                             motion_length_pred_origin[i]:]],axis=0)
        #     if mdm_jts_smpl_24_pred.shape[0] != constant_length + 1:
        #         mdm_jts_smpl_24_pred = np.concatenate([mdm_jts_smpl_24_pred,
        #                                                mdm_jts_smpl_24_pred[[-1]]],axis=0)
        #     data_vec, global_positions,\
        #     positions, l_velocity = process_file(positions=mdm_jts_smpl_24_pred[:,:22,:],
        #                                             feet_thre=0.002,tgt_offsets=self.target_offset,
        #                                             face_joint_indx=[2, 1, 17, 16],
        #                                             fid_l=[7,10],fid_r=[8,11],
        #                                             n_raw_offsets= torch.from_numpy(t2m_raw_offsets),
        #                                             kinematic_chain=t2m_kinematic_chain,skeleton=self.skeleton)
        #     data_vec_batch.append(data_vec[None,:,:])
        # data_vec = np.concatenate(data_vec_batch,axis=0)
        # data_vec = torch.from_numpy(data_vec).unsqueeze(1).float()
        # try:
        #     data_vec = ((data_vec-data.dataset.t2m_dataset.mean)/data.dataset.t2m_dataset.std).permute(0,3,1,2)
        # except:
        #     data_vec = ((data_vec-data.t2m_dataset.mean)/data.t2m_dataset.std).permute(0,3,1,2)
        normalizer.normalize(data_vec)
        return data_vec.to(self.device).type(torch.float32)
    
    def imitation(self,length,motion_num=1,return_reward=False):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None
        if return_reward==False:
            pose = torch.zeros((motion_num,length+1,24,4),dtype=torch.float32,device=self.device)
            root_pos = torch.zeros((motion_num,length+1,3),dtype=torch.float32,device=self.device)
            joint_position = torch.zeros((motion_num,length+1,24,3),dtype=torch.float32,device=self.device)
            motion_length = torch.zeros(motion_num,dtype=torch.float32,device=self.device)
        else:
            reward = torch.zeros((motion_num,length),dtype=torch.float32,device=self.device)
        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn
        env_ids = list(range(motion_num))
        obs_dict = self.env_reset()
        batch_size = motion_num
        batch_size = self.get_batch_size(obs_dict['obs'], batch_size)
        # import ipdb;ipdb.set_trace()
        if return_reward == False:
            pose[:,0],root_pos[:,0],joint_position[:,0] = self.get_mdm_state(env_ids)
            
        if need_init_rnn:
            self.init_rnn()
            need_init_rnn = False

        cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        print_game_res = False

        done_indices = []

        self.task.render_vis(init=True)

        prev_dones = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        for n in range(length):
            t = n%self.task.context_length
            if n>0 and t==0:
                self.task._init_context(self.task._reset_ref_motion_ids, self.task._cur_ref_motion_times)
            
            obs_dict['t'] = t
            obs_dict['global_t_offset'] = n-t
            if has_masks:
                masks = self.env.get_action_mask()
                action = self.get_masked_action(obs_dict, masks, is_determenistic)
            else:
                action = self.get_action(obs_dict, is_determenistic)

            obs_dict, r, done, info =  self.env_step(self.env, action)
            cr += r
            steps += 1
            if return_reward==False:
                pose[:,n+1],root_pos[:,n+1],joint_position[:,n+1] = self.get_mdm_state(env_ids)
            else:
                if n<length-1:
                    reward[done==0,n] = r[done==0]
                else:
                    reward[prev_dones==0,n] = r[prev_dones==0]
            self.task.render_vis()

            self._post_step(info)

            if render:
                self.env.render(mode = 'human')
                time.sleep(self.render_sleep)

            step_dones = done * (1 - prev_dones)
            if return_reward==False:
                motion_length[step_dones==1] = n+2#
            all_done_indices = step_dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
            done_count = len(done_indices)
            games_played += done_count

            if done_count > 0:
                if self.is_rnn:
                    for s in self.states:
                        s[:,all_done_indices,:] = s[:,all_done_indices,:] * 0.0

                cur_rewards = cr[done_indices].sum().item()
                cur_steps = steps[done_indices].sum().item()

                cr = cr * (1.0 - done.float())
                steps = steps * (1.0 - done.float())
                sum_rewards += cur_rewards
                sum_steps += cur_steps

                game_res = 0.0
                if isinstance(info, dict):
                    if 'battle_won' in info:
                        print_game_res = True
                        game_res = info.get('battle_won', 0.5)
                    if 'scores' in info:
                        print_game_res = True
                        game_res = info.get('scores', 0.5)
                if self.print_stats:
                    if print_game_res:
                        print('reward:', cur_rewards/done_count, 'steps:', cur_steps/done_count, 'w:', game_res)
                    else:
                        print('reward:', cur_rewards/done_count, 'steps:', cur_steps/done_count)

                sum_game_res += game_res

            done_indices = done_indices[:, 0]

            prev_dones = done.clone()
        if return_reward:
            return reward    
        return pose,root_pos,joint_position,motion_length
        
    def get_mdm_state(self,env_ids):
        root_pos = self.task._humanoid_root_states[env_ids,0:3]
        dof_pos = self.task._dof_pos[env_ids]
        root_rot = self.task._humanoid_root_states[env_ids, 3:7]
        pose_quat = torch.from_numpy(sRot.from_rotvec(
                                    dof_pos.reshape(-1, 3).cpu().numpy()
                                    ).as_quat().reshape(-1, 23, 4)).to(self.device)
        pose_quat = torch.concatenate([root_rot[:,None,:],pose_quat],dim=1)
        joint_position = self.task._rigid_body_pos[env_ids]

        return pose_quat,root_pos,joint_position

class SourceKeyPoints(nn.Module):
        def __init__(self,
                    bm: Union[str, BodyModel],
                    n_joints: int=22,
                    kpts_colors: Union[np.ndarray, None] = None ,
                    num_betas=16
                    ):
            super(SourceKeyPoints, self).__init__()

            self.bm = BodyModel(bm,num_betas=num_betas, persistant_buffer=False) if isinstance(bm, str) else bm
            self.bm_f = []#self.bm.f
            self.n_joints = n_joints
            self.kpts_colors = np.array([Color('grey').rgb for _ in range(n_joints)]) if kpts_colors == None else kpts_colors

        def forward(self, body_parms):
            new_body = self.bm(**body_parms)


            return {'source_kpts':new_body.Jtr[:,:self.n_joints], 'body': new_body}

mujoco_2_smpl = [0, 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12, 14, 19, 13, 15, 20, 16, 21, 17, 22, 18, 23]            
smpl_2_mujoco = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 22, 14, 17, 19, 21, 23]            
STANDING_POSE = np.array([[[-0.1443, -0.9426, -0.2548],
         [-0.2070, -0.8571, -0.2571],
         [-0.0800, -0.8503, -0.2675],
         [-0.1555, -1.0663, -0.3057],
         [-0.2639, -0.5003, -0.2846],
         [-0.0345, -0.4931, -0.3108],
         [-0.1587, -1.2094, -0.2755],
         [-0.2534, -0.1022, -0.3361],
         [-0.0699, -0.1012, -0.3517],
         [-0.1548, -1.2679, -0.2675],
         [-0.2959, -0.0627, -0.2105],
         [-0.0213, -0.0424, -0.2277],
         [-0.1408, -1.4894, -0.2892],
         [-0.2271, -1.3865, -0.2622],
         [-0.0715, -1.3832, -0.2977],
         [-0.1428, -1.5753, -0.2303],
         [-0.3643, -1.3792, -0.2646],
         [ 0.0509, -1.3730, -0.3271],
         [-0.3861, -1.1423, -0.3032],
         [ 0.0634, -1.1300, -0.3714],
         [-0.4086, -0.9130, -0.2000],
         [ 0.1203, -0.8943, -0.3002],
         [-0.4000, -0.8282, -0.1817],
         [ 0.1207, -0.8087, -0.2787]]]).repeat(180, axis = 0)

t2m_raw_offsets = np.array([[0,0,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,0,1],
                           [0,0,1],
                           [0,1,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,0,1],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0]])

t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]

def get_rot(rot, dt, target_dt):
    # rot = rot[..., [1, 2, 3, 0]]
    nframe = rot.shape[0]
    x = np.arange(nframe) * dt
    x_target = np.arange(int(nframe * dt / target_dt)) * target_dt

    rotations = sRot.from_quat(rot)
    spline = RotationSpline(x, rotations)

    return spline(x_target, 0).as_quat()#[:, [3, 0, 1, 2]]


def interpolate_trans(y, dt, target_dt):
    x = np.arange(y.shape[0]) * dt
    x_target = np.arange(int(y.shape[0] * dt / target_dt)) * target_dt
    cs = interpolate.CubicSpline(x, y)
    return cs(x_target)

def fps_20_to_30(mdm_jts):
    jts = []
    N = mdm_jts.shape[0]
    for i in range(mdm_jts.shape[1]):
        int_x = mdm_jts[:, i, 0]
        int_y = mdm_jts[:, i, 1]
        int_z = mdm_jts[:, i, 2]
        x = np.arange(0, N)
        f_x = interpolate.interp1d(x, int_x, kind='linear')
        f_y = interpolate.interp1d(x, int_y, kind='linear')
        f_z = interpolate.interp1d(x, int_z, kind='linear')
        
        new_x = f_x(np.linspace(0, N-1, int(N * 1.5)))
        new_y = f_y(np.linspace(0, N-1, int(N * 1.5)))
        new_z = f_z(np.linspace(0, N-1, int(N * 1.5)))
        jts.append(np.stack([new_x, new_y, new_z], axis = 1))
    jts = np.stack(jts, axis = 1)
    return jts

def fps_30_to_20(mdm_jts):
    jts = []
    N = mdm_jts.shape[0]
    for i in range(mdm_jts.shape[1]):
        int_x = mdm_jts[:, i, 0]
        int_y = mdm_jts[:, i, 1]
        int_z = mdm_jts[:, i, 2]
        x = np.arange(0, N)
        f_x = interpolate.interp1d(x, int_x, kind='linear')
        f_y = interpolate.interp1d(x, int_y, kind='linear')
        f_z = interpolate.interp1d(x, int_z, kind='linear')
        
        new_x = f_x(np.linspace(0, N-1, int(N * 2/3)))
        new_y = f_y(np.linspace(0, N-1, int(N * 2/3)))
        new_z = f_z(np.linspace(0, N-1, int(N * 2/3)))
        jts.append(np.stack([new_x, new_y, new_z], axis = 1))
    jts = np.stack(jts, axis = 1)
    return jts

def get_smpl_parser(gender):
    if gender == 0:
        smpl_parser =  SMPL_Parser(model_path=f"{ROOT_PATH}/data/smpl", gender="neutral", use_pca=False, create_transl=False)
    elif gender == 1:
        smpl_parser =  SMPL_Parser(model_path=f"{ROOT_PATH}/data/smpl", gender="male", use_pca=False, create_transl=False)
    elif gender == 2:
        smpl_parser = SMPL_Parser(model_path=f"{ROOT_PATH}/data/smpl", gender="female", use_pca=False, create_transl=False)
    else:
        print(gender)
        raise Exception("Gender Not Supported!!")
    return smpl_parser

def fix_height_smpl_vanilla(pose_aa, th_trans, th_betas, gender, seq_name):
    # no filtering, just fix height
    # gender = gender.item() if isinstance(gender, np.ndarray) else gender
    # if isinstance(gender, bytes):
    #     gender = gender.decode("utf-8")

    if gender == 0:
        smpl_parser =  SMPL_Parser(model_path=f"{ROOT_PATH}/data/smpl", gender="neutral", use_pca=False, create_transl=False)
    elif gender == 1:
        smpl_parser =  SMPL_Parser(model_path=f"{ROOT_PATH}/data/smpl", gender="male", use_pca=False, create_transl=False)
    elif gender == 2:
        smpl_parser = SMPL_Parser(model_path=f"{ROOT_PATH}/data/smpl", gender="female", use_pca=False, create_transl=False)
    else:
        print(gender)
        raise Exception("Gender Not Supported!!")

    batch_size = pose_aa.shape[0]
    verts, jts = smpl_parser.get_joints_verts(pose_aa[0:1], th_betas.repeat((1, 1)), th_trans=th_trans[0:1])

    # vertices = verts[0].numpy()
    gp = torch.min(verts[0, :, 2])

    # if gp < 0:
    th_trans[:, 2] -= gp

    return th_trans,smpl_parser

def transform_smpl_coordinate(bm_fname: Path, trans: np.ndarray,
                              root_orient: np.ndarray, betas: np.ndarray,
                              rotxyz: Union[np.ndarray, List]) -> Dict:
    """
    rotates smpl parameters while taking into account non-zero center of rotation for smpl
    Parameters
    ----------
    bm_fname: body model filename
    trans: Nx3
    root_orient: Nx3
    betas: num_betas
    rotxyz: desired XYZ rotation in degrees

    Returns
    -------

    """
    if isinstance(rotxyz, list):
        rotxyz = np.array(rotxyz).reshape(1, 3)
    if betas.ndim == 1: betas = betas[None]
    if betas.ndim == 2 and betas.shape[0] != 1:
        logger.warning(
            f'betas should be the same for the entire sequence. 2D np.array with 1 x num_betas: {betas.shape}. taking the mean')
        betas = np.mean(betas, keepdims=True, axis=0)
    transformation_euler = np.deg2rad(rotxyz)

    coord_change_matrot = sRot.from_euler('XYZ', transformation_euler.reshape(1, 3)).as_matrix().reshape(3, 3)
    bm = BodyModel(bm_fname=bm_fname,
                   num_betas=betas.shape[1])
    pelvis_offset = c2c(bm(**{'betas': torch.from_numpy(betas).type(torch.float32)}).Jtr[[0], 0])

    root_matrot = sRot.from_rotvec(root_orient).as_matrix().reshape([-1, 3, 3])

    transformed_root_orient_matrot = np.matmul(coord_change_matrot, root_matrot.T).T
    transformed_root_orient = sRot.from_matrix(transformed_root_orient_matrot).as_rotvec()
    transformed_trans = np.matmul(coord_change_matrot, (trans + pelvis_offset).T).T - pelvis_offset

    return {'root_orient': transformed_root_orient.astype(np.float32),
            'trans': transformed_trans.astype(np.float32), }

class IK_solver(nn.Module):
    def __init__(self,surface_model_type="smplh", gender="neutral",verbosity=0):
        super().__init__()
        self.comp_device = torch.device('cuda')
        self.surface_model_type = surface_model_type
        self.gender = gender
        self.verbosity = verbosity
        support_dir = f'./{ROOT_PATH}/human_body_prior/support_data/dowloads'
        vposer_expr_dir = osp.join(support_dir,'vposer_v2_05') #'TRAINED_MODEL_DIRECTORY'  in this directory the trained model along with the model code exist
        self.bm_fname =  osp.join(support_dir,f'models/{self.surface_model_type}/{self.gender}/model.npz')#'PATH_TO_SMPLX_model.npz'  obtain from https://smpl-x.is.tue.mpg.de/downloads
        self.n_joints = 22
        self.num_betas = 16

        red = Color("red")
        blue = Color("blue")
        self.kpts_colors = [c.rgb for c in list(red.range_to(blue,self.n_joints))]
        # create source and target key points and make sure they are index aligned
        self.data_loss = data_loss = torch.nn.MSELoss(reduction='sum')
        stepwise_weights = [
            {'data': 10., 'poZ_body': .01, 'betas': .5},
                            ]
        #from 300 to 500
        optimizer_args = {'type':'LBFGS', 'max_iter':300, 'lr':1, 'tolerance_change': 1e-4, 'history_size':200}
        self.ik_engine = IK_Engine(vposer_expr_dir=vposer_expr_dir,
                            verbosity=self.verbosity,
                            display_rc= (2, 2),
                            data_loss=data_loss,
                            stepwise_weights=stepwise_weights,
                            optimizer_args=optimizer_args)
        self.source_pts = SourceKeyPoints(bm=self.bm_fname, n_joints=self.n_joints,kpts_colors=self.kpts_colors
                                        ,num_betas=self.num_betas)
    
    def forward(self, mdm_data):      
        batch_size = mdm_data.shape[0]
        all_results = {}
        batched_frames = create_list_chunks(np.arange(batch_size), batch_size, overlap_size=0,
                                            cut_smaller_batches=False)
        # if verbosity < 2:
        #     batched_frames = tqdm(batched_frames, desc='VPoser Advanced IK')
        for cur_frame_ids in batched_frames:

            target_pts = mdm_data[cur_frame_ids, :self.n_joints]

            ik_res = self.ik_engine(self.source_pts, target_pts, {})

            ik_res_detached = {k: c2c(v) for k, v in ik_res.items()}
            # nan_mask = np.isnan(ik_res_detached['trans']).sum(-1) != 0
            # if nan_mask.sum() != 0: raise ValueError('Sum results were NaN!')
            for k, v in ik_res_detached.items():
                if k not in all_results: all_results[k] = []
                all_results[k].append(v)

        d = {k: np.concatenate(v, axis=0) for k, v in all_results.items()}
        # d['betas'] = np.median(d['betas'], axis=0)
        d['betas'] = np.zeros(self.num_betas)
        # d['trans'] = trans #TODO:changed
        transformed_d = transform_smpl_coordinate(bm_fname=self.bm_fname, trans=d['trans'],
                                                root_orient=d['root_orient'],
                                                betas=d['betas'], rotxyz=[90, 0, 0])
        d.update(transformed_d)
        d['poses'] = np.concatenate([d['root_orient'], d['pose_body'], np.zeros([len(d['pose_body']), 99])],
                                    axis=1)

        d['surface_model_type'] = self.surface_model_type
        d['gender'] = self.gender
        d['mocap_framerate'] = 20
        d['num_betas'] = self.num_betas

        del batched_frames
        del ik_res
        del ik_res_detached
        torch.cuda.empty_cache()

        return torch.concatenate([torch.from_numpy(d['trans']),
                                  torch.from_numpy(d['poses'])],dim=1).to(self.comp_device).clone().detach()
