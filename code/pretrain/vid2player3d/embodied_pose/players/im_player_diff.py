import torch
import os
import numpy as np
import time
from tqdm.auto import tqdm
import functools
import joblib

from scipy.spatial.transform import Rotation as sRot
import scipy.interpolate as interpolate
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.players import rescale_actions, unsqueeze_obs
from rl_games.common.player import BasePlayer

# from MDM.mdm_talker import MDMTalker
from players.im_player import ImitatorPlayer
from MDM.data_loaders.humanml.scripts.motion_process import recover_from_ric,get_quaternion
from uhc.smpllib.smpl_local_robot import Robot as LocalRobot
from uhc.smpllib.smpl_parser import SMPL_Parser
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from embodied_pose.utils.motion_lib import MotionLib


class diff_player(ImitatorPlayer):
    def __init__(self,config):
       super().__init__(config)
       self.mdm = self.task.mdm
       
    def run(self):
        denoised_steps = [3,2,1,0]
        while True:
            prompt = input("your prompt...")
            args, model_kwargs, model, diffusion, data= self.mdm.args, self.mdm.model_kwargs, self.mdm.model, self.mdm.diffusion, self.mdm.data
            model_kwargs['y']['text'] = prompt
            batch_size = 1
            fps = 12.5 if args.dataset == 'kit' else 20
            
            if args.guidance_param != 1:
                model_kwargs['y']['scale'] = torch.ones(batch_size, device=self.device) * args.guidance_param
            sample_fn = diffusion.ddim_sample
            shape = (batch_size, self.mdm.model.njoints,self.mdm.model.nfeats,self.mdm.n_frames)
            img = torch.randn(*shape,device = self.device)
            indices = list(range(diffusion.num_timesteps))[::-1]
            for i in tqdm(indices):
                t = torch.tensor([i]*shape[0],device = self.device)
                with torch.no_grad():
                    if i in denoised_steps:
                        denoised_fn = functools.partial(self.phys_proj,data)
                    else:
                        denoised_fn = None
                    out = sample_fn(
                            model,
                            img,
                            t,
                            clip_denoised=False,
                            denoised_fn = denoised_fn,
                            model_kwargs = model_kwargs,
                            eta = 0
                        )
                    img = out["sample"]
           
    def phys_proj(self,data,sample):
        gender_beta = self.task._reset_ref_motion_bodies[0].cpu() 
        n_joints = 22 if sample.shape[1] == 263 else 21
        sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
        sample = recover_from_ric(sample, n_joints)
        sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
        sample = sample.cpu().numpy()
        # hand_len = 0.08824
        offset_height = 0.92
        offset_xy = np.zeros([sample.shape[0], 24, 3])
        mdm_jts = sample.transpose(0, 3, 1, 2).reshape(1, -1, 22, 3)
        
        mdm_jts_smpl_24 = mdm_jts.squeeze()
        # direction = (mdm_jts[...,  -2, :] - mdm_jts[...,  -4, :])
        # left = mdm_jts[...,  -2, :] + direction/np.linalg.norm(direction) * hand_len
        # direction = (mdm_jts[...,  -1, :] - mdm_jts[...,  -3, :])
        # right = mdm_jts[...,  -1, :] + direction/np.linalg.norm(direction) * hand_len
        # mdm_jts_smpl_24 = np.concatenate([mdm_jts, left[...,  None, :], right[..., None, :]], axis = -2).squeeze()
        
        mat = sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False).as_matrix()
        mdm_jts_smpl_24 = np.matmul(mdm_jts_smpl_24, mat.dot(mat))
        offset = - offset_height - mdm_jts_smpl_24[ 0:1, 0:1, 1]
        mdm_jts_smpl_24[..., 1] += offset
        mdm_jts_smpl_24[..., [0, 2]] -= mdm_jts_smpl_24[:1, :1, [0, 2]] - offset_xy[:1, :1, [0, 2]]
        mdm_jts_smpl_24 = fps_20_to_30(mdm_jts_smpl_24)
        
        mat_to_isaac = sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False).as_matrix()
        mdm_jts_smpl_24 = np.matmul(mdm_jts_smpl_24, mat_to_isaac.T)
        
        trans = torch.from_numpy(mdm_jts_smpl_24[:,0]) #seq_len*3
        # pose_quat = 
        # pose_quat, r_velocity, velocity, r_rot = get_quaternion(mdm_jts_smpl_24) #seq_len*22*4
        hand = np.array([[0.0,0.0,0.0,1.0]]).repeat(pose_quat.shape[0],0)
        pose_quat = np.concatenate([pose_quat,hand[:,None,:],hand[:,None,:]],axis=-2) #seq_len*24*4
        pose_aa = sRot.from_quat(pose_quat.reshape(-1,4)).as_rotvec().reshape(-1,72).copy() #(seq_len)*72
        import ipdb;ipdb.set_trace()
        pose_quat = torch.from_numpy(pose_quat)[..., smpl_2_mujoco, :]
        trans,smpl_parser = fix_height_smpl_vanilla(
            pose_aa=torch.from_numpy(pose_aa),
            th_trans=trans,
            th_betas=gender_beta[1:],
            gender= gender_beta[0].item(),
            seq_name=None       
        )
        ################################
        # trans: torch, seq_len*3
        # pose_aa: numpy, seq_len*72, smpl
        # pose_quat: torch, seq_len*24*4, mujoco
        # gender_beta: torch , 11
        ################################
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
        root_trans = trans + skeleton_tree.local_translation[0]
        
        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            pose_quat,
            root_trans,
            is_local=True)
        
        verts, joints = smpl_parser.get_joints_verts(
            pose=torch.from_numpy(pose_aa),
            th_betas=gender_beta[1:][None, ],
            th_trans=trans
        )

        # min_verts_h = verts[..., 2].min().item()
        min_verts_h = verts[..., 2].min(dim=-1)[0].mean().item()

        

        new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=fps)
        new_motion_out = new_motion.to_dict()
        new_motion_out['seq_name'] = "mdm"
        new_motion_out['seq_idx'] = 0
        new_motion_out['trans'] = trans.numpy()
        new_motion_out['root_trans'] = root_trans.numpy()
        new_motion_out['pose_aa'] = pose_aa[...,smpl_2_mujoco,:]
        new_motion_out['beta'] = gender_beta[1:].numpy()
        new_motion_out['beta_idx'] = 0
        new_motion_out['gender'] = ["neutral","male","female"][int(gender_beta[0].item())]
        new_motion_out['min_verts_h'] = min_verts_h
        new_motion_out['body_scale'] = 1.0
        new_motion_out['__name__'] = "SkeletonMotion"
        motion_lib_input_dict['mdm'] = new_motion_out

        motion_lib = MotionLib(motion_file=motion_lib_input_dict,
            dof_body_ids=info['dof_body_ids'],
            dof_offsets=info['dof_offsets'],
            key_body_ids=info['key_body_ids'],
            device= self.device,
            clean_up=True
        )
        
        self.task._motion_lib = motion_lib
        self.task._reset_ref_motion_ids = torch.tensor([0]).to(self.device)
        something = self.imitation(root_trans.shape[0])
        # trans = mdm_jts_smpl_24[:,[0]]
        # mdm_jts_smpl_24_orig = mdm_jts_smpl_24.copy()
        # mdm_jts_smpl_24 = mdm_jts_smpl_24 - trans
        
        
        # import ipdb
        # ipdb.set_trace()
        
         
        
        # print(rot.shape)
        pass       
            
    def imitation(self,length):
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

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn
        env_ids = 0
        obs_dict = self.env_reset()
        batch_size = 1
        batch_size = self.get_batch_size(obs_dict['obs'], batch_size)

        if need_init_rnn:
            self.init_rnn()
            need_init_rnn = False

        cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        print_game_res = False

        done_indices = []

        # self.task.render_vis(init=True)

        prev_dones = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        for t in range(length):
            if t==0:
                self.task._init_context(self.task._reset_ref_motion_ids, self.task._cur_ref_motion_times)
            
            obs_dict['t'] = t
            obs_dict['global_t_offset'] = 0
            if has_masks:
                masks = self.env.get_action_mask()
                action = self.get_masked_action(obs_dict, masks, is_determenistic)
            else:
                action = self.get_action(obs_dict, is_determenistic)

            obs_dict, r, done, info =  self.env_step(self.env, action)

            cr += r
            steps += 1

            # self.task.render_vis()

            self._post_step(info)

            if render:
                self.env.render(mode = 'human')
                time.sleep(self.render_sleep)

            step_dones = done * (1 - prev_dones)
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
                if batch_size//self.num_agents == 1 or games_played >= n_games:
                    break
            
            done_indices = done_indices[:, 0]

            prev_dones = done.clone()
            
            
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
         [ 0.1207, -0.8087, -0.2787]]]).repeat(5, axis = 0)

def fps_20_to_30(mdm_jts):
    jts = []
    N = mdm_jts.shape[0]
    for i in range(mdm_jts.shape[1]):
        int_x = mdm_jts[:, i, 0]
        int_y = mdm_jts[:, i, 1]
        int_z = mdm_jts[:, i, 2]
        x = np.arange(0, N)
        f_x = interpolate.interp1d(x, int_x)
        f_y = interpolate.interp1d(x, int_y)
        f_z = interpolate.interp1d(x, int_z)
        
        new_x = f_x(np.linspace(0, N-1, int(N * 1.5)))
        new_y = f_y(np.linspace(0, N-1, int(N * 1.5)))
        new_z = f_z(np.linspace(0, N-1, int(N * 1.5)))
        jts.append(np.stack([new_x, new_y, new_z], axis = 1))
    jts = np.stack(jts, axis = 1)
    return jts

def fix_height_smpl_vanilla(pose_aa, th_trans, th_betas, gender, seq_name):
    # no filtering, just fix height
    # gender = gender.item() if isinstance(gender, np.ndarray) else gender
    # if isinstance(gender, bytes):
    #     gender = gender.decode("utf-8")

    if gender == 0:
        smpl_parser =  SMPL_Parser(model_path="data/smpl", gender="neutral", use_pca=False, create_transl=False)
    elif gender == 1:
        smpl_parser =  SMPL_Parser(model_path="data/smpl", gender="male", use_pca=False, create_transl=False)
    elif gender == 2:
        smpl_parser = SMPL_Parser(model_path="data/smpl", gender="female", use_pca=False, create_transl=False)
    else:
        print(gender)
        raise Exception("Gender Not Supported!!")

    batch_size = pose_aa.shape[0]
    verts, jts = smpl_parser.get_joints_verts(pose_aa[0:1], th_betas.repeat((1, 1)), th_trans=th_trans[0:1])

    # vertices = verts[0].numpy()
    gp = torch.min(verts[:, :, 2])

    # if gp < 0:
    th_trans[:, 2] -= gp

    return th_trans,smpl_parser