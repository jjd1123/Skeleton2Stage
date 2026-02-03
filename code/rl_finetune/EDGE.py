# ===== Standard library =====
import multiprocessing
import os
import pickle
import time
import math
import random
import copy
import gc
from functools import partial
from pathlib import Path

# ===== Third-party =====
import yaml
from easydict import EasyDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.cuda.amp import autocast

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from accelerate.utils import set_seed

from tqdm import tqdm

from scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema
from scipy.spatial.transform import Rotation as sRot

# ===== Local imports =====
from dataset.dance_dataset import AISTPPDataset, AISTPPDataset_RL
from dataset.preprocess import increment_path

from model.adan import Adan
from model.diffusion import GaussianDiffusion
from model.model import DanceDecoder
from model.stat_tracking import PerPromptStatTracker

from vis import SMPLSkeleton

from model.Bailando.sep_vqvae import SepVQVAE
from model.imitation.embodied_pose.utils.edge.quaternion import ax_from_6v

# ===== Optional (may not exist in all environments) =====
try:
    from model.imitation.embodied_pose.run_player import get_agent, get_player
    from model.Bailando.utils.kinetic import KineticFeatures
except ImportError:
    pass

# Hard Code. True means finetuning PopDG.
PopDance = False
if PopDance == True:
    from model.POPDG.model.model import Model
    from dataset.load_popdanceset import PopDanceSet

def wrap(x):
    return {f"module.{key}": value for key, value in x.items()}

def unwrap(x):
    return {key.replace("module.",""):value for key,value in x.items()}

def maybe_wrap(x, num):
    return x if num == 1 else wrap(x)

def load_reward_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

class EDGE:
    def __init__(
        self,
        feature_type,
        checkpoint_path="",
        normalizer=None,
        EMA=True,
        learning_rate=1e-6,
        weight_decay=1e-4,
    ):
        # accelerate init
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(mixed_precision="fp16",
                                       kwargs_handlers=[ddp_kwargs])
        state = AcceleratorState()
        set_seed(42,True)
        torch.backends.cudnn.deterministic = True
        num_processes = state.num_processes

        # model's config init
        use_baseline_feats = feature_type == "baseline"

        pos_dim = 3
        rot_dim = 24 * 6  # 24 joints, 6dof
        # popdance: [trans: 0-2, pose_6d: 3-146, contact: 147-155]
        # EDGE; [contact: 0-3, trans: 4-6, pose_6d: 7-150]
        if PopDance == True:
            self.repr_dim = repr_dim = pos_dim + rot_dim + 9
        else:
            self.repr_dim = repr_dim = pos_dim + rot_dim + 4

        feature_dim = 35 if use_baseline_feats else 4800

        horizon_seconds = 5
        FPS = 30
        self.horizon = horizon = horizon_seconds * FPS

        self.accelerator.wait_for_everyone()

        # get checkpoint
        checkpoint = None
        if checkpoint_path != "":
            if PopDance == True and "checkpoint.pt" in checkpoint_path:
                checkpoint_path = "./model/POPDG/checkpoint.pt"
            checkpoint = torch.load(
                checkpoint_path, map_location=self.accelerator.device
            )
            print(checkpoint_path)
            self.normalizer = checkpoint["normalizer"]

        # define lora config
        # lora_config = LoraConfig(
        #     target_modules=[r"seqTransDecoder.*linear.",r"seqTransDecoder.*attn",r"seqTransDecoder.*film.","input_projection","final_layer"],
        #     # modules_to_save=["final_layer"],
        #     r=32,
        #     lora_alpha=16
        # )

        # define models
        if PopDance == True:
            model = Model(
            nfeats=repr_dim,
            nframes=self.horizon,
            latent_dim=512,
            ff_dim=1024,
            num_layers=8,
            num_heads=8,
            dropout=0.0,
            music_feature_dim=feature_dim,
            activation=F.gelu,
            )
        else:
            model = DanceDecoder(
                nfeats=repr_dim,
                seq_len=horizon,
                latent_dim=512,
                ff_size=1024,
                num_layers=8,
                num_heads=8,
                dropout=0.0,
                cond_feature_dim=feature_dim,
                activation=F.gelu,
            )
        # model.load_state_dict(
        #         maybe_wrap(
        #             checkpoint["ema_state_dict" if EMA else "model_state_dict"],
        #             1,
        #         )
        #     )
        
        # get lora model
        # model = get_peft_model(model,lora_config)
        # print(model.print_trainable_parameters())
        # print(self.accelerator.local_process_index)
        
        # prepare for diffusion
        smpl = SMPLSkeleton(self.accelerator.device)
        diffusion = GaussianDiffusion(
            model,
            horizon,
            repr_dim,
            smpl,
            schedule="cosine",
            n_timestep=1000,
            predict_epsilon=False,
            loss_type="l2",
            use_p2=False,
            cond_drop_prob=0.25,
            guidance_weight=2,
        )
        
        # get vqvae features
        # with open("./model/Bailando/config.yaml") as f:
        #     hps = yaml.load(f,yaml.SafeLoader)
        # self.vqvae = SepVQVAE(EasyDict(hps).structure).to(self.accelerator.device)
        # self.vqvae.eval()
        # vq_checkpoint = torch.load("./model/Bailando/sep_vqvae/epoch_500.pt")
        # self.vqvae.load_state_dict(unwrap(vq_checkpoint["model"]))
        

        print(
            "Model has {} parameters".format(sum(y.numel() for y in model.parameters()))
        )

        self.model = self.accelerator.prepare(model)
        self.diffusion = diffusion.to(self.accelerator.device)
        optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)#weight_decay=weight_decay)
        self.optim = self.accelerator.prepare(optim)
        # self.optim.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint_path != "":
            self.model.load_state_dict(
                maybe_wrap(
                    checkpoint["ema_state_dict" if EMA else "model_state_dict"],
                    num_processes,
                )
            )
            self.diffusion.master_model.load_state_dict(
                maybe_wrap(
                    checkpoint["ema_state_dict" if EMA else "model_state_dict"],
                    1,
                )
            )
            # used for calculate KL_divergence
            # self.diffusion.origin_model = copy.deepcopy(self.diffusion.model)
            # self.diffusion.origin_model.load_state_dict(
            #     maybe_wrap(
            #         checkpoint["ema_state_dict" if EMA else "model_state_dict"],
            #         1,
            #     )
            # )
        # if str(self.accelerator.device)[-1] == "a":
        #     self.accelerator.state.device = torch.device("cuda:0")
        # print(self.accelerator.device)

    def pfc_score(self,x):
        b,constant_length = x.shape[:2]
        smpl = self.diffusion.smpl
        n_joints = 24
        sample = self.normalizer.unnormalize(x)
        if PopDance == True:
            trans = sample[...,:3].reshape(b,constant_length,3).clone()
            pose_6d = sample[...,3:3+24*6].reshape(b,constant_length,n_joints,6)
        else:
            trans = sample[...,4:7].reshape(b,constant_length,3).clone() #b*l*3
            pose_6d = sample[...,7:].reshape(b,constant_length,n_joints,6)
        pose_aa = ax_from_6v(pose_6d)
        joint = smpl.forward(
            pose_aa,trans
            ).cpu().numpy()

        up_dir = 2  # z is up
        flat_dirs = [i for i in range(3) if i != up_dir]
        DT = 1 / 30
        joint3d = joint
        root_v = (joint3d[:, 1:, 0, :] - joint3d[:, :-1, 0, :]) / DT  # root velocity (b, S-1, 3)
        root_a = (root_v[:, 1:] - root_v[:, :-1]) / DT  # (b, S-2, 3) root accelerations
        # clamp the up-direction of root acceleration
        root_a[:, :, up_dir] = np.maximum(root_a[:, :, up_dir], 0)  # (b, S-2, 3)
        # l2 norm
        root_a = np.linalg.norm(root_a, axis=-1)  # (b, S-2,)
        scaling = root_a.max(axis=-1, keepdims=True)
        root_a /= scaling

        foot_idx = [7, 10, 8, 11]
        feet = joint3d[:, :, foot_idx]  # foot positions (b, S, 4, 3)
        foot_v = np.linalg.norm(
            feet[:, 2:, :, flat_dirs] - feet[:, 1:-1, :, flat_dirs], axis=-1
        )  # (b, S-2, 4) horizontal velocity
        foot_mins = np.zeros((foot_v.shape[0],foot_v.shape[1], 2))
        foot_mins[:, :, 0] = np.minimum(foot_v[:, :, 0], foot_v[:, :, 1])
        foot_mins[:, :, 1] = np.minimum(foot_v[:, :, 2], foot_v[:, :, 3])

        foot_loss = (
            foot_mins[:, :, 0] * foot_mins[:, :, 1] * root_a
        )  # min leftv * min rightv * root_a (b, S-2,)
        foot_loss = np.exp(-1000*foot_loss.mean(axis=-1))
        return torch.as_tensor(foot_loss)

    def compute_reward(self,x,x_origin):
        B,constant_length = x.shape[:2]
        smpl = self.diffusion.smpl
        n_joints = 24
        x = self.normalizer.unnormalize(x)
        x_origin = self.normalizer.unnormalize(x_origin)
        trans = x[...,4:7] #b*l*3
        pose_aa  = ax_from_6v(x[...,7:].reshape(B,constant_length,n_joints,6))
        trans_origin = x_origin[...,4:7]
        pose_aa_origin  = ax_from_6v(x_origin[...,7:].reshape(B,constant_length,n_joints,6))
        x = smpl.forward(
            pose_aa,trans
            ).cpu().numpy()
        x_origin = smpl.forward(
            pose_aa_origin,trans_origin
            ).cpu().numpy()
        mat_to_mdm = sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False).as_matrix()
        x = torch.from_numpy(np.matmul(x, mat_to_mdm.T)).to(self.accelerator.device).reshape(B,constant_length,-1)
        x_origin = torch.from_numpy(np.matmul(x_origin,mat_to_mdm.T)).to(self.accelerator.device).reshape(B,constant_length,-1)
        root = x[..., :3] # the root
        x = x - torch.tile(root, (1,1,24))  # Calculate relative offset with respect to root
        x[..., :3] = root
        root_origin = x_origin[..., :3] # the root
        x_origin = x_origin - torch.tile(root_origin, (1,1,24))  # Calculate relative offset with respect to root
        x_origin[..., :3] = root_origin
        quants = self.vqvae.encode(x)
        quants_origin = self.vqvae.encode(x_origin)
        # if self.accelerator.is_main_process:
        #     print(quants[0][0][0])
        #     print(quants_origin[0][0][0])
        
        return torch.exp(-100*((quants[0][0]-quants_origin[0][0])**2+
                               (quants[1][0]-quants_origin[1][0])**2)).mean(dim=1).mean(dim=1)
        

    def compute_freezing(self,recon_x,normalizer):
        n_joints = 24
        chosen_joints = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                         16,17,18,19,20,21,22,23]
        b, constant_length = recon_x.shape[:2]
        sample = normalizer.unnormalize(recon_x)
        trans = sample[...,4:7].reshape(b,1,constant_length,3) #b*l*3
        pose_6d = sample[...,7:].reshape(b,1,constant_length,n_joints,6)
        # sample_x = normalizer.unnormalize(x)
        # trans_x = sample_x[...,4:7].reshape(b,3,constant_length//3,3) #b*l*3
        # pose_6d_x = sample_x[...,7:].reshape(b,3,constant_length//3,n_joints,6)
        trans_freeze = ((trans[...,1:,:]-trans[...,:-1,:])**2).mean(dim=(-1,-2))
        # trans_x_freeze = ((trans_x[...,1:,:]-trans_x[...,:-1,:])**2).mean(dim=(-1,-2))
        pose_6d_freeze = ((pose_6d[:,:,1:,chosen_joints]-pose_6d[:,:,:-1,chosen_joints])**2).mean(dim=(-1,-2,-3))
        # pose_6d_x_freeze = ((pose_6d_x[:,:,1:]-pose_6d_x[:,:,:-1])**2).mean(dim=(-1,-2,-3))
        # trans_rate = (trans_freeze/trans_x_freeze).mean(dim=1)
        # pose_rate = (pose_6d_freeze/pose_6d_x_freeze).mean(dim=1)
        # rate = torch.where((trans_rate+pose_rate)/2>1,1.0,(trans_rate+pose_rate)/2)
        rate = trans_freeze.mean(dim=1)+pose_6d_freeze.mean(dim=1)
        return rate
    
    def compute_k(self,recon_x,normalizer):
        n_joints = 24
        chosen_joints = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                         16,17,18,19,20,21,22,23]
        b, constant_length = recon_x.shape[:2]
        sample = normalizer.unnormalize(recon_x)
        if PopDance == True:
            trans = sample[...,:3].reshape(b,constant_length,3).clone()
        else:
            trans = sample[...,4:7].reshape(b,constant_length,3).clone() #b*l*3
        trans[...,2] = 0 #modified to improve foot ground contact
        if PopDance == True:
            pose_6d = sample[...,3:3+24*6].reshape(b,constant_length,n_joints,6)
        else:
            pose_6d = sample[...,7:].reshape(b,constant_length,n_joints,6)
        pose_aa = ax_from_6v(pose_6d)
        smpl = self.diffusion.smpl
        position = smpl.forward(pose_aa,trans).detach().clone().cpu().numpy()
        # position = position - trans[...,2].numpy()
        ks = []
        # print(position.shape)
        k = KineticFeatures(position,1/30.0,"z")
        kd = np.zeros(position.shape[0])
        for j in chosen_joints:
            kd += (k.average_energy_expenditure(j))
            kd += (k.average_kinetic_energy(j))
        kd /= (len(chosen_joints)*2)
        ks = torch.tensor(kd)
        return ks
    
    def compute_fgc(self,recon_x,normalizer):
        #foot_ground contact
        n_joints = 24
        chosen_joints = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                         16,17,18,19,20,21,22,23]
        b, constant_length = recon_x.shape[:2]
        sample = normalizer.unnormalize(recon_x)
        if PopDance == True:
            trans = sample[...,:3].reshape(b,constant_length,3).clone()
            pose_6d = sample[...,3:3+24*6].reshape(b,constant_length,n_joints,6)
        else:
            trans = sample[...,4:7].reshape(b,constant_length,3).clone() #b*l*3
            pose_6d = sample[...,7:].reshape(b,constant_length,n_joints,6)
        pose_aa = ax_from_6v(pose_6d)
        smpl = self.diffusion.smpl
        position = smpl.forward(pose_aa,trans).detach().clone().cpu().numpy() #b*l*J*3
        lowest_joint = np.min(position[...,2],axis=-1)
        ground = lowest_joint.mean(axis=-1)[...,None]
        if_penetration = lowest_joint < ground
        if_floating = lowest_joint > ground
        penetration_reward = np.exp(-100*((lowest_joint-ground)**2)*if_penetration)
        floating_reward = np.exp(-100*((lowest_joint-ground)**2)*if_floating)
        # position = position - trans[...,2].numpy()

        # sample_x = normalizer.unnormalize(x)
        # trans_x = sample_x[...,4:7].reshape(b,3,constant_length//3,3) #b*l*3
        # pose_6d_x = sample_x[...,7:].reshape(b,3,constant_length//3,n_joints,6)
        # trans_freeze = ((trans[...,1:,:]-trans[...,:-1,:])**2).mean(dim=(-1,-2))
        # trans_x_freeze = ((trans_x[...,1:,:]-trans_x[...,:-1,:])**2).mean(dim=(-1,-2))
        # pose_6d_freeze = ((pose_6d[:,:,1:,chosen_joints]-pose_6d[:,:,:-1,chosen_joints])**2).mean(dim=(-1,-2,-3))
        # pose_6d_x_freeze = ((pose_6d_x[:,:,1:]-pose_6d_x[:,:,:-1])**2).mean(dim=(-1,-2,-3))
        # trans_rate = (trans_freeze/trans_x_freeze).mean(dim=1)
        # pose_rate = (pose_6d_freeze/pose_6d_x_freeze).mean(dim=1)
        # rate = torch.where((trans_rate+pose_rate)/2>1,1.0,(trans_rate+pose_rate)/2)
        # rate = trans_freeze.mean(dim=1)+pose_6d_freeze.mean(dim=1)
        return torch.tensor(penetration_reward+floating_reward)
    
    def compute_ba(self, recon_x, music_beats, normalizer):
        n_joints = 24
        b, constant_length = recon_x.shape[:2]
        sample = normalizer.unnormalize(recon_x)
        if PopDance == True:
            trans = sample[...,:3].reshape(b,constant_length,3).clone()
            pose_6d = sample[...,3:3+24*6].reshape(b,constant_length,n_joints,6)
        else:
            trans = sample[...,4:7].reshape(b,constant_length,3).clone() #b*l*3
            pose_6d = sample[...,7:].reshape(b,constant_length,n_joints,6)
        pose_aa = ax_from_6v(pose_6d)
        smpl = self.diffusion.smpl
        joint3d = smpl.forward(pose_aa,trans).reshape(b,constant_length,24*3).detach().clone().cpu().numpy()
        joint3d_60 = np.zeros((b,constant_length*2,72))
        joint3d_60[:,:-1:2] = joint3d
        joint3d_60[:,1::2] = joint3d
        joint3d = joint3d_60
        # roott = joint3d[:, :1, :3]
        # joint3d = joint3d - np.tile(roott, (1, 1, 24)) 
        joint3d = joint3d.reshape(b, -1, 24, 3)
        # keypoints = np.array(keypoints).reshape(-1, 24, 3)
        # keypoints = keypoints - keypoints[:,[0]]
        kinetic_vels = np.mean(np.sqrt(np.sum((joint3d[:,1:] - joint3d[:,:-1]) ** 2, axis=3)), axis=2)
        bas = []
        for index, music_beat in enumerate(music_beats):
            kinetic_vel = kinetic_vels[index]
            kinetic_vel = G(kinetic_vel, 5)
            motion_beats = argrelextrema(kinetic_vel, np.less)
            ba = 0
            # print(len(music_beats))
            for bb in music_beat:
                ba +=  np.exp(-np.min((motion_beats[0] - bb)**2) / 2 / 9)
            bas.append(ba / len(music_beat))
        return torch.tensor(bas,dtype=torch.float32)


    def finetune(self,epoch):
        # if self.accelerator.is_main_process:
        id = self.accelerator.process_index
        # if imitation_x.shape[0]<8192:
        #     print('less than 8192')
        #     times = int(8192/imitation_x.shape[0]) + 1
        #     imitation_x = torch.concatenate([imitation_x]*times,dim = 0)[:8192]

        # l = imitation_x.shape[0]
        # l = int(l/8)
        # imitation_x = imitation_x[id*l:(id+1)*l]
        # print(imitation_x.shape)
        # l = imitation_x.shape[0]
        # l=int(l/8)
        # for s in load_loop(range(8)):
        #     player.phys_proj_edge(self.normalizer,1,imitation_x[s*l:(s+1)*l],return_reward=True,
        #                                 device = self.accelerator.device,multi_process="thread",
        #                                 accelerator=self.accelerator,steps=s,save_motion=True)
        # self.accelerator.wait_for_everyone()
        # # if self.accelerator.is_main_process:
        # player.env.task.gym.destroy_sim(player.env.task.sim)
        # del player
        agent = get_agent(self.accelerator.device)
        agent.restore("latestnew")
        agent.pepare_accelerate(self.accelerator)
        agent.train(epoch)
        agent.task.gym.destroy_sim(agent.task.sim)
        del agent
        self.accelerator.wait_for_everyone()
        player = get_player(self.accelerator.device)
        # self.accelerator.wait_for_everyone()
        player.restore("latest")
        gc.collect()
        torch.cuda.empty_cache() 
        return player

    def train_rl(self, opt, player):
        # training setup
        # finetune means using the motion generated by diffusion model to finetune the imitation policy.
        # use_isaac means using the imitation policy to compute the imitation reward.
        reward_cfg = load_reward_config(opt.reward_config)
        finetune = False
        use_isaac = reward_cfg["imitation"]["enabled"]
        inner_epoch = 1
        sample_steps = 1
        logging_backend = opt.log_choice


        # the training dataset needs extra processing and is named train_rl_tensor_dataset; while the test dataset are the same dataset as EDGE.
        if PopDance==True:
            opt.processed_data_dir = "./data/dataset_backups/popdg"
        train_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"train_rl_tensor_dataset.pkl"
        )
        test_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"test_tensor_dataset.pkl"
        )
        # During training, we don't use test dataset, so we comment them out. 
        # If you want to do visualization during training, you can follow the guidance in EDGE  
        if (
            not opt.no_cache
            and os.path.isfile(train_tensor_dataset_path)
            and os.path.isfile(test_tensor_dataset_path)
        ):
            train_dataset = pickle.load(open(train_tensor_dataset_path, "rb"))
            # test_dataset = pickle.load(open(test_tensor_dataset_path, "rb"))
        else:
            train_dataset = AISTPPDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=True,
                force_reload=opt.force_reload,
            )
            # test_dataset = pickle.load(open(test_tensor_dataset_path, "rb"))
            # test_dataset = AISTPPDataset(
            #     data_path=opt.data_path,
            #     backup_path=opt.processed_data_dir,
            #     train=False,
            #     normalizer=train_dataset.normalizer,
            #     force_reload=opt.force_reload,
            # )
            # cache the dataset in case
            if self.accelerator.is_main_process:
                pickle.dump(train_dataset, open(train_tensor_dataset_path, "wb"))
                # pickle.dump(test_dataset, open(test_tensor_dataset_path, "wb"))
        
        # generate index to mark different conditions. This is used to compute the per-condition reward's mean.
        # test_dataset.generate_idx()
        train_dataset.generate_idx()
        # set normalizer
        if self.normalizer == None:
            # print("Using test dataset normalizer!!!")
            # self.normalizer = test_dataset.normalizer
            print("Using test dataset normalizer!!!")
            self.normalizer = train_dataset.normalizer
        
        # set reward tracker
        stat_tracker = PerPromptStatTracker(32 ,16) # track for imitation reward
        # kl_tracker = PerPromptStatTracker(32,16)
        freezing_tracker = PerPromptStatTracker(32,16)
        fgc_tracker = PerPromptStatTracker(32*150 ,16*150)
        # ba_tracker = PerPromptStatTracker(32,16)
        # root_tracker = PerPromptStatTracker(32*150 ,16*150)
        if PopDance == True:
            pfc_tracker = PerPromptStatTracker(32,16)   
        
        # data loaders
        # decide number of workers based on cpu count
        num_cpus = multiprocessing.cpu_count()
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=0,#min(int(num_cpus * 0.75), 32),
            pin_memory=True,
            drop_last=True,
            # collate_fn=train_dataset.collate_fn
            # sampler= RandomSampler(train_dataset,replacement=True,num_samples=opt.batch_size*2),
        )
        # test_data_loader = DataLoader(
        #     test_dataset,
        #     batch_size=186,#opt.batch_size,
        #     shuffle=False,
        #     num_workers=2,
        #     pin_memory=True,
        #     drop_last=True,
        # )


        train_data_loader = self.accelerator.prepare(train_data_loader)
        train_data_loader_iter = iter(train_data_loader)
        # boot up multi-gpu training. test dataloader is only on main process
        load_loop = (
            # partial(tqdm, position=1, desc="Batch")
            # if self.accelerator.is_main_process
            # else
            lambda x: x
        )
        if self.accelerator.is_main_process:
            save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
            opt.exp_name = save_dir.split("/")[-1]
            save_dir = Path(save_dir)
            wdir = save_dir / "weights" #TODO: remove absolute path
            wdir.mkdir(parents=True, exist_ok=True)
            if logging_backend=="wandb":
                import wandb
                wandb.init(project=opt.wandb_pj_name, name=opt.exp_name)
            elif logging_backend=="swanlab":
                import swanlab 
                swanlab.init(project=opt.wandb_pj_name, workspace=opt.workspace, experiment_name=opt.exp_name)
            else:
                from torch.utils.tensorboard import SummaryWriter
                tb_logger = SummaryWriter(f"tensorboard/{opt.exp_name}")

        self.accelerator.wait_for_everyone()
        for epoch in range(1, opt.epochs + 1):
            rewards = []
            latents = []
            timesteps = []
            log_probs = []
            conds = []
            indexs = []
            # kl_rewards = []
            alives = []
            freezings = []
            # imitation_x = []
            motion_libs = []
            x_starts = []
            fgcs = []
            # root_rewards = []
            pfcs = []
            # bas = []
            self.eval()
            start_sampling = time.time()
            # with autocast(dtype=torch.float32):
            with torch.no_grad():
                for sample_step in range(sample_steps):
                    # for step, (x, cond, filename, wavnames,index) in enumerate(
                    #     load_loop(train_data_loader)
                    # ): 
                        step = 0
                        try:
                            x,cond,filename,wavnames,index = next(train_data_loader_iter)
                        except:
                            train_data_loader_iter = iter(train_data_loader)
                            x,cond,filename,wavnames,index = next(train_data_loader_iter)
                        
                        x = x.to(self.accelerator.device).repeat((4,1,1))
                        cond = cond.to(self.accelerator.device).repeat((4,1,1))
                        index = index.to(self.accelerator.device).repeat((4))
                        # beats = beats * 4
                        shape = x.shape
                        if PopDance == True:
                            shape = torch.zeros((shape[0],shape[1],3+24*6+9))
                            shape = shape.shape
                        time_step,latent,cond,log_prob,recon_x,x_start = self.diffusion.ddim_sample(shape,cond,return_logprob=True)
                        x_starts.append(x_start)
                        freezing_rate = self.compute_k(recon_x.detach().clone(),self.normalizer).to(self.accelerator.device)
                        fgc = self.compute_fgc(recon_x.detach().clone(),self.normalizer).to(self.accelerator.device)
                        # ba = self.compute_ba(recon_x.detach().clone(),beats,self.normalizer).to(self.accelerator.device)
                        if PopDance == True:
                            pfc = self.pfc_score(recon_x.detach().clone()).to(self.accelerator.device)
                        if use_isaac:
                            motion_lib = player.phys_proj_edge(self.normalizer,1,recon_x.detach().clone().cpu(),return_reward=True,
                                                device = self.accelerator.device,multi_process="thread",
                                                accelerator=self.accelerator,steps=step,save_motion=True,batch_size=opt.batch_size*4)
                            motion_libs.append(motion_lib)
                        else:
                            reward = torch.zeros(shape[0],device=self.accelerator.device)
                            alive = reward.clone()
                            alives.append(alive)
                            rewards.append(self.accelerator.gather(reward).cpu()
                                        .reshape(self.accelerator.num_processes, -1))
                        freezings.append(self.accelerator.gather(freezing_rate).cpu()
                                        .reshape(self.accelerator.num_processes, -1))
                        fgcs.append(self.accelerator.gather(fgc).cpu()
                                        .reshape(self.accelerator.num_processes, -1, 150))
                        # bas.append(self.accelerator.gather(ba).cpu()
                        #                 .reshape(self.accelerator.num_processes, -1))
                        if PopDance == True:
                            pfcs.append(self.accelerator.gather(pfc).cpu()
                                            .reshape(self.accelerator.num_processes, -1))
                        # kl_rewards.append(self.accelerator.gather(kl_reward).cpu()
                                            # .reshape(self.accelerator.num_processes, -1))
                        if len(timesteps) == 0:
                            timesteps.append(time_step)
                        latents.append(latent)
                        log_probs.append(log_prob)
                        conds.append(cond)#.extend(filename*4)
                        indexs.append(self.accelerator.gather(index).cpu()
                                        .reshape(self.accelerator.num_processes,-1))
            if finetune:
                self.accelerator.wait_for_everyone()
                player.env.task.gym.destroy_sim(player.env.task.sim)
                del player
                player = self.finetune(epoch)
            if use_isaac:
                start_isaac = time.time()
                for motion_lib in motion_libs:
                    reward,alive = player.phys_proj_edge(self.normalizer,1,motion_lib,return_reward=True,
                                        device = self.accelerator.device,multi_process="thread",
                                        accelerator=self.accelerator,steps=0,save_motion=False,batch_size=opt.batch_size*4)
                    # reward[reward>0.8]=0.8
                    alives.append(alive.cpu())
                    rewards.append(self.accelerator.gather(reward).cpu()
                                    .reshape(self.accelerator.num_processes, -1, 1))
                    # del root_reward
                    # root_rewards.append(self.accelerator.gather(root_reward).cpu()
                    #                 .reshape(self.accelerator.num_processes, -1, 150))    
                # print("isaac_time:",time.time()-start_isaac)
            # concat all data 
            conds = torch.concatenate(conds,dim=0).cpu()
            timesteps = timesteps[0]
            latents = torch.concatenate(latents,dim=0).cpu()
            log_probs = torch.concatenate(log_probs,dim=0).cpu()
            rewards = torch.concatenate(rewards,dim=1).view(-1)#.mean(dim=-1)
            # root_rewards = torch.concatenate(root_rewards,dim=1).view(-1,150)
            avg_reward = rewards.mean()
            # avg_root_reward = root_rewards.mean()
            # kl_rewards = torch.concatenate(kl_rewards,dim=1).view(-1)
            # avg_kl_reward = kl_rewards.mean()
            indexs = torch.concatenate(indexs,dim=1).view(-1)
            alives = torch.concatenate(alives,dim=0)
            alives = alives.detach().cpu()
            freezings= torch.concatenate(freezings,dim=1).view(-1)
            fgcs= torch.concatenate(fgcs,dim=1).view(-1,150)
            # bas = torch.concatenate(bas,dim=1).view(-1)
            if PopDance == True:
                pfcs = torch.concatenate(pfcs,dim=1).view(-1)
            avg_freezing = freezings.mean()
            std_freezing = freezings.std()
            avg_fgc = fgcs.mean()
            # avg_ba = bas.mean()
            if PopDance == True:
                avg_pfc = pfcs.mean()
            x_starts = torch.concatenate(x_starts,dim=0).cpu()

            # update reward tracker and get normalized reward
            # kl_rewards = torch.as_tensor(kl_tracker.update(indexs,kl_rewards))
            freezings = torch.as_tensor(freezing_tracker.update(indexs,freezings))
            rewards = torch.as_tensor(stat_tracker.update(indexs,rewards))
            fgcs = torch.as_tensor(fgc_tracker.update(indexs,fgcs))
            # bas = torch.as_tensor(ba_tracker.update(indexs,bas))
            if PopDance == True:
                pfcs = torch.as_tensor(pfc_tracker.update(indexs,pfcs))
            # root_rewards = torch.as_tensor(root_tracker.update(indexs,root_rewards))
            norm_reward = rewards.mean()
            reward_list = []
            for key in reward_cfg.keys():
                if reward_cfg[key]["enabled"]:
                    if key == "imitation":
                        reward_list.append(reward_cfg[key]["weight"]*rewards.reshape(-1,1))
                    elif key == "antifreezing":
                        reward_list.append(reward_cfg[key]["weight"]*freezings.reshape(-1,1))
                        
                    elif key == "fgcontact":
                        reward_list.append(reward_cfg[key]["weight"]*fgcs)
                    elif key == "pfc" and PopDance == True:
                        reward_list.append(reward_cfg[key]["weight"]*pfcs.reshape(-1,1))
            rewards = sum(reward_list)
            # weight = 0.0
            # fgc_weight = 0.01
            # # bas_weight = 0.00
            # pfc_weight = 0.0
            # if PopDance == True:
            #     rewards =  (weight*freezings).reshape(-1,1)+(1-weight)*rewards.reshape(-1,1)+fgc_weight*fgcs+pfc_weight*pfcs.reshape(-1,1)#root_rewards#
            # else:
            #     rewards =  (weight*freezings).reshape(-1,1)+(1-weight)*rewards.reshape(-1,1)+fgc_weight*fgcs#+bas_weight*bas.reshape(-1,1)
            #TODO maybe reduce fgcs weights
            # only for multi-gpu
            rewards = (rewards.reshape(self.accelerator.num_processes,-1)[self.accelerator.process_index]).reshape(opt.batch_size*4,-1)
            
            if self.accelerator.is_main_process:
                # quants_reward = torch.concatenate(all_rewards1,dim=0).mean()
                # isacc_reward = torch.concatenate(all_rewards_isaac,dim=0).mean()
                if PopDance == True:
                    log_dict = {
                            "imitation_weight": reward_cfg["imitation"]["weight"],
                            "freezing_weight":reward_cfg["antifreezing"]["weight"],
                            "fgc_weight": reward_cfg["fgcontact"]["weight"],
                            "pfc_weight": reward_cfg["pfc"]["weight"],
                            "Train reward": avg_reward,
                            # "kl reward": avg_kl_reward,
                            # "kl_norm_reward": kl_rewards.mean(),
                            "Train_norm_reward": norm_reward,
                            "total reward": avg_reward+avg_freezing,
                            "freeze": avg_freezing,
                            "freeze_norm": freezings.mean(),
                            "freeze_std":std_freezing,
                            "fgc":avg_fgc,
                            "fgc_norm":fgcs.mean(),
                            "pfc":avg_pfc,
                            "pfc_norm":pfcs.mean(),
                            # "root_reward":avg_root_reward
                            # "Quant reward": quants_reward,
                            # "Isacc reward": isacc_reward
                        }
                else:
                    log_dict = {
                            "imitation_weight": reward_cfg["imitation"]["weight"],
                            "freezing_weight": reward_cfg["antifreezing"]["weight"],
                            "fgc_weight": reward_cfg["fgcontact"]["weight"],
                            # "pfc_weight": pfc_weight,
                            "Train reward": avg_reward,
                            # "kl reward": avg_kl_reward,
                            # "kl_norm_reward": kl_rewards.mean(),
                            # "Train_norm_reward": norm_reward,
                            "total reward": avg_reward+avg_freezing,
                            "freeze": avg_freezing,
                            "freeze_norm": freezings.mean(),
                            "freeze_std":std_freezing,
                            "fgc":avg_fgc,
                            "fgc_norm":fgcs.mean(),
                            # "ba_weight": bas_weight,
                            # "avg_ba": avg_ba,
                            # "pfc":avg_pfc,
                            # "pfc_norm":pfcs.mean(),
                            # "root_reward":avg_root_reward
                            # "Quant reward": quants_reward,
                            # "Isacc reward": isacc_reward
                        }
                if logging_backend=="wandb":
                    wandb.log(log_dict)
                elif logging_backend=="swanlab":
                    swanlab.log(log_dict)
                else:
                    for k,v in log_dict.items():
                        tb_logger.add_scalar(k,v,epoch)
            del fgc,log_prob,x,cond,x_start,recon_x,index,latent,freezing_rate,alive,motion_lib,motion_libs,reward
            # latents = self.accelerator.gather(latents.to(self.accelerator.device)).cpu()
            # log_probs  = self.accelerator.gather(log_probs.to(self.accelerator.device)).cpu()
            # conds = self.accelerator.gather(conds.to(self.accelerator.device)).cpu()
            # alives = self.accelerator.gather(alives.to(self.accelerator.device)).cpu()
            # x_starts = self.accelerator.gather(x_starts.to(self.accelerator.device)).cpu()
            # print(latents.shape)
            # print(log_probs.shape)
            # print(conds.shape)
            # print(alives.shape)
            # print(x_starts.shape)
            # print(rewards.shape)
            gc.collect()
            torch.cuda.empty_cache()
            RL_dataset =  AISTPPDataset_RL(rewards,timesteps,latents,log_probs,conds,alives,x_starts)
            num_cpus = multiprocessing.cpu_count()
            RL_data_loader = DataLoader(
                RL_dataset,
                batch_size=opt.batch_size*4,
                shuffle=True,
                num_workers=0,#min(int(num_cpus * 0.75), 32),
                pin_memory=True,
                drop_last=True,
            )   
            # RL_data_loader = self.accelerator.prepare(RL_data_loader)
            self.accelerator.wait_for_everyone() 
            # loss_record=[]   
            # kl_loss_re = []
            # l_re = []
            self.train()
            for inner in range(1,inner_epoch+1):
                for step, (reward,latent,next_latent,logprob,feature,time_step,alive,x_starts) in enumerate(load_loop(RL_data_loader)):
                    # try:
                    #     x,cond,filename,wavnames,index = next(simple_data_loader_iter)
                    # except:
                    #     simple_data_loader_iter = iter(simple_data_loader)
                    #     x,cond,filename,wavnames,index = next(simple_data_loader_iter)
                    # x = x.to(self.accelerator.device)
                    # cond = cond.to(self.accelerator.device)
                    # with self.accelerator.accumulate(self.diffusion.model):
                    perms = torch.stack(
                        [
                            torch.randperm(50, device=self.accelerator.device)
                            for _ in range(opt.batch_size*4)
                        ]
                    )
                    perms_help = torch.arange(opt.batch_size*4,device=self.accelerator.device)[:,None]
                    reward = reward.to(self.accelerator.device)
                    # alive = alive.to(self.accelerator.device)
                    latent = latent.to(self.accelerator.device)[perms_help,perms]
                    next_latent = next_latent.to(self.accelerator.device)[perms_help,perms]
                    # logprob = logprob.to(self.accelerator.device)[perms_help,perms]
                    feature = feature.to(self.accelerator.device)
                    time_step = time_step.to(self.accelerator.device)[perms_help,perms]
                    x_starts = x_starts.to(self.accelerator.device)[perms_help,perms]
                    for j in range(50):
                        # time_cond = torch.full((256,), time_step[0,j], device=self.accelerator.device, dtype=torch.long)
                        # total_loss, (loss, v_loss, fk_loss, foot_loss) = self.diffusion(
                        #     x, cond, t_override=time_step[0,j]
                        # )
                        pred_noise, x_start, *_ = self.diffusion.model_predictions(latent[:,j], feature,
                                                                                time_step[:,j], 
                                                                                clip_x_start = self.diffusion.clip_denoised)
                        alpha = self.diffusion.alphas_cumprod[time_step[:,j].cpu()].to(self.accelerator.device).unsqueeze(1).unsqueeze(1)
                        # print(alpha.shape)
                        prev_timestep = time_step[:,j] - 20
                        alpha_next = torch.where(
                                    prev_timestep >= 0,
                                    self.diffusion.alphas_cumprod[prev_timestep.cpu()],
                                    # alpha.squeeze(1).squeeze(1)
                                    torch.full_like(self.diffusion.alphas_cumprod[prev_timestep.cpu()],1.0),
                                ).to(self.accelerator.device).unsqueeze(1).unsqueeze(1)
                        #to compute logprob(x_0|x_1)e
                        eta = 1
                        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                        
                        if not torch.all((x_start-x_starts[:,j])==0):
                            print("maybe numerical error?")
                            print(f"at {j}")
                        c = (1 - alpha_next - sigma ** 2).sqrt()
                        # prev_sample_mean_kl = alpha_next.sqrt()*x_start_kl+c*pred_noise_kl
                        prev_sample_mean = alpha_next.sqrt()*x_start+c*pred_noise
                        prev_sample = next_latent[:,j]
                        sigma = torch.clip(sigma,min=1e-6)
                        log_prob_new = (-((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (sigma**2))
                            - torch.log(sigma)
                            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi,device=self.accelerator.device))))
                        log_prob_new = log_prob_new.mean(dim=tuple(range(2,log_prob_new.ndim)))
                        # mask = torch.ones_like(log_prob_new,device=self.accelerator.device)
                        # mask[prev_timestep<0] = 0
                        # log_prob_new = log_prob_new*mask
                        if torch.isinf(log_prob_new.sum()) or torch.isnan(log_prob_new.sum()):
                            print(f"meet inf or nan at {time_step[:,j]}")
                        advantages = torch.clamp(reward,-5.0,5.0)
                        # ratio = torch.exp(log_prob_new - logprob[:,j].detach())
                        # if self.accelerator.is_main_process:
                            # print(time_step[ratio!=1.0,j].shape)
                            # print(log_prob_new)#19结果不一致，网络存在随机性？
                            # print(ratio)
                            # break
                        unclipped_loss = -advantages * log_prob_new#ratio
                        # clipped_loss = -advantages * torch.clamp(
                        #     ratio,
                        #     1.0 - 0.0001,
                        #     1.0 + 0.0001,
                        # )
                        loss = torch.mean(unclipped_loss)/50.0
                        # loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))/50.0#*alive)
                        # kl_loss = torch.mean((x_start-x_start_kl)**2)
                        # total_loss = loss + 0.1*kl_loss
                        # loss_record.append(total_loss)
                        # kl_loss_re.append(kl_loss)
                        # l_re.append(loss)
                        self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), 1.0
                    )
                    # print("clip")
                self.optim.step()
                self.optim.zero_grad()
                # ema update and train loss update only on main
                if self.accelerator.is_main_process:# and self.accelerator.sync_gradients:
                    # log_dict = {
                    #     "Train Loss": loss.mean(),
                    #     "V Loss": v_loss.mean(),
                    #     "FK Loss": fk_loss.mean(),
                    #     "Foot Loss": foot_loss.mean(),
                    # }
                    # wandb.log(log_dict)
                    # for k,v in log_dict.items():
                    #     tb_logger.add_scalar(k,v,epoch)
                    if step % opt.ema_interval == 0:
                        self.diffusion.ema.update_model_average(
                            self.diffusion.master_model, self.diffusion.model
                        )
                assert self.accelerator.sync_gradients
            print("epoch_time:",time.time()-start_sampling)
            # loss_record = torch.stack(loss_record).mean()
            # loss_record = self.accelerator.gather(loss_record).cpu().mean()
            # kl_loss_re = torch.stack(kl_loss_re).mean()
            # kl_loss_re = self.accelerator.gather(kl_loss_re).cpu().mean()
            # l_re = torch.stack(l_re).mean()
            # l_re = self.accelerator.gather(l_re).cpu().mean()
            # if self.accelerator.is_main_process:
            #     log_dict.update({
            #         "total_loss": loss_record,
            #         "kl_loss": kl_loss_re,
            #         "a_loss": l_re
            #     })
            #     wandb.log(log_dict)
            # del loss_record
            # del kl_loss_re
            # del l_re
            # Save model
            del RL_data_loader
            del RL_dataset
            if (epoch % opt.save_interval) == 0:
                # everyone waits here for the val loop to finish ( don't start next train epoch early)
                self.accelerator.wait_for_everyone()
                # save only if on main thread
                if self.accelerator.is_main_process:
                    self.eval()
                    # log
                    ckpt = {
                        "ema_state_dict": self.diffusion.master_model.state_dict(),
                        "model_state_dict": self.accelerator.unwrap_model(
                            self.model
                        ).state_dict(),
                        "optimizer_state_dict": self.optim.state_dict(),
                        "normalizer": self.normalizer,
                    }
                    torch.save(ckpt, os.path.join(wdir, f"train-{epoch}.pt"))
                    # # generate a sample
                    # render_count = 2
                    # shape = (render_count, self.horizon, self.repr_dim)
                    # print("Generating Sample")
                    # # draw a music from the test dataset
                    # (x, cond, filename, wavnames,*_) = next(iter(test_data_loader))
                    # cond = cond.to(self.accelerator.device)
                    # self.diffusion.render_sample(
                    #     shape,
                    #     cond[:render_count],
                    #     self.normalizer,
                    #     epoch,
                    #     os.path.join(opt.render_dir, "train_" + opt.exp_name),
                    #     name=wavnames[:render_count],
                    #     sound=True,
                    # )
                    print(f"[MODEL SAVED at Epoch {epoch}]")
            # gc.collect()
            # torch.cuda.empty_cache() 
        if self.accelerator.is_main_process:
            if logging_backend == "wandb":
                wandb.finish()
            elif logging_backend == "swanlab":
                swanlab.finish()
            else:
                tb_logger.close()
    
    def eval(self):
        self.diffusion.eval()

    def train(self):
        self.diffusion.train()
        if self.diffusion.origin_model!=None:
            self.diffusion.origin_model.eval()

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)

    def train_loop(self, opt):
        # load datasets
        train_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"train_tensor_dataset.pkl"
        )
        test_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"test_tensor_dataset.pkl"
        )
        if (
            not opt.no_cache
            and os.path.isfile(train_tensor_dataset_path)
            and os.path.isfile(test_tensor_dataset_path)
        ):
            train_dataset = pickle.load(open(train_tensor_dataset_path, "rb"))
            test_dataset = pickle.load(open(test_tensor_dataset_path, "rb"))
        else:
            train_dataset = AISTPPDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=True,
                force_reload=opt.force_reload,
            )
            test_dataset = AISTPPDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=False,
                normalizer=train_dataset.normalizer,
                force_reload=opt.force_reload,
            )
            # cache the dataset in case
            if self.accelerator.is_main_process:
                pickle.dump(train_dataset, open(train_tensor_dataset_path, "wb"))
                pickle.dump(test_dataset, open(test_tensor_dataset_path, "wb"))

        # set normalizer
        self.normalizer = test_dataset.normalizer

        # data loaders
        # decide number of workers based on cpu count
        num_cpus = multiprocessing.cpu_count()
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=min(int(num_cpus * 0.75), 32),
            pin_memory=True,
            drop_last=True,
        )
        test_data_loader = DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

        train_data_loader = self.accelerator.prepare(train_data_loader)
        # boot up multi-gpu training. test dataloader is only on main process
        load_loop = (
            partial(tqdm, position=1, desc="Batch")
            if self.accelerator.is_main_process
            else lambda x: x
        )
        if self.accelerator.is_main_process:
            save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
            opt.exp_name = save_dir.split("/")[-1]
            wandb.init(project=opt.wandb_pj_name, name=opt.exp_name)
            save_dir = Path(save_dir)
            wdir = save_dir / "weights"
            wdir.mkdir(parents=True, exist_ok=True)

        self.accelerator.wait_for_everyone()
        for epoch in range(1, opt.epochs + 1):
            avg_loss = 0
            avg_vloss = 0
            avg_fkloss = 0
            avg_footloss = 0
            # train
            self.train()
            for step, (x, cond, filename, wavnames) in enumerate(
                load_loop(train_data_loader)
            ):
                total_loss, (loss, v_loss, fk_loss, foot_loss) = self.diffusion(
                    x, cond, t_override=None
                )
                self.optim.zero_grad()
                self.accelerator.backward(total_loss)

                self.optim.step()

                # ema update and train loss update only on main
                if self.accelerator.is_main_process:
                    avg_loss += loss.detach().cpu().numpy()
                    avg_vloss += v_loss.detach().cpu().numpy()
                    avg_fkloss += fk_loss.detach().cpu().numpy()
                    avg_footloss += foot_loss.detach().cpu().numpy()
                    if step % opt.ema_interval == 0:
                        self.diffusion.ema.update_model_average(
                            self.diffusion.master_model, self.diffusion.model
                        )
            # Save model
            if (epoch % opt.save_interval) == 0:
                # everyone waits here for the val loop to finish ( don't start next train epoch early)
                self.accelerator.wait_for_everyone()
                # save only if on main thread
                if self.accelerator.is_main_process:
                    self.eval()
                    # log
                    avg_loss /= len(train_data_loader)
                    avg_vloss /= len(train_data_loader)
                    avg_fkloss /= len(train_data_loader)
                    avg_footloss /= len(train_data_loader)
                    log_dict = {
                        "Train Loss": avg_loss,
                        "V Loss": avg_vloss,
                        "FK Loss": avg_fkloss,
                        "Foot Loss": avg_footloss,
                    }
                    wandb.log(log_dict)
                    ckpt = {
                        "ema_state_dict": self.diffusion.master_model.state_dict(),
                        "model_state_dict": self.accelerator.unwrap_model(
                            self.model
                        ).state_dict(),
                        "optimizer_state_dict": self.optim.state_dict(),
                        "normalizer": self.normalizer,
                    }
                    torch.save(ckpt, os.path.join(wdir, f"train-{epoch}.pt"))
                    # generate a sample
                    render_count = 2
                    shape = (render_count, self.horizon, self.repr_dim)
                    print("Generating Sample")
                    # draw a music from the test dataset
                    (x, cond, filename, wavnames) = next(iter(test_data_loader))
                    cond = cond.to(self.accelerator.device)
                    self.diffusion.render_sample(
                        shape,
                        cond[:render_count],
                        self.normalizer,
                        epoch,
                        os.path.join(opt.render_dir, "train_" + opt.exp_name),
                        name=wavnames[:render_count],
                        sound=True,
                    )
                    print(f"[MODEL SAVED at Epoch {epoch}]")
        if self.accelerator.is_main_process:
            wandb.run.finish()

    def render_sample(
        self, data_tuple, label, render_dir, render_count=-1, fk_out=None, render=True,player=None,i=None,motion=None
    ):
        _, cond, wavname = data_tuple
        assert len(cond.shape) == 3
        if render_count < 0:
            render_count = len(cond)
        shape = (render_count, self.horizon, self.repr_dim)
        cond = cond.to(self.accelerator.device)
        # print(cond.shape)
        self.diffusion.render_sample(
                shape,
                cond[:render_count],
                self.normalizer,
                label,
                render_dir,
                name=wavname[:render_count],
                sound=True,
                mode="long",
                fk_out=fk_out,
                render=render,
                i = i,
                player = player,
                motion=motion
            )
