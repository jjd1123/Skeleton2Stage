from trimesh import Trimesh
import numpy as np
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm
import os
import glob
import sys
sys.path.append("./model/imitation")
from model.imitation.embodied_pose.players.im_player_diff import IK_solver
from model.imitation.support_path import *
from model.imitation.uhc.smpllib.smpl_parser import SMPL_Parser
from model.imitation.embodied_pose.utils.MDM.smpl import SMPL
import torch
from torch import nn
import smplx
import argparse

def get_joints_verts(pose, th_betas=None, th_trans=None, root_trans=None, root_scale=None,model=None):
        """
        Pose should be batch_size x 72
        """
        if pose.shape[1] != 72:
            pose = pose.reshape(-1, 72)

        pose = pose.float()
        if th_betas is not None:
            th_betas = th_betas.float()

            if th_betas.shape[-1] == 16:
                th_betas = th_betas[:, :10]

        batch_size = pose.shape[0]
        # print(model)
        smpl_output = model(
            betas=th_betas,
            transl=th_trans,
            body_pose=pose[:, 3:],
            global_orient=pose[:, :3],
        )
        # print(smpl_output)
        vertices = smpl_output.vertices
        print(vertices.shape)
        joints = smpl_output.joints[:, :24]
        # joints = smpl_output.joints[:,JOINST_TO_USE]
        if root_trans is not None:
            if root_scale is None:
                root_scale = torch.ones_like(root_trans[:, 0])
            cur_root_trans = joints[:, [0], :]
            vertices[:] = (vertices - cur_root_trans) * root_scale[:, None, None] + root_trans[:, None, :]
            joints[:] = (joints - cur_root_trans) * root_scale[:, None, None] + root_trans[:, None, :]
        return vertices, joints


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


def get_results_from_rot(pose_aa,root,gender_beta,save_path):
    smpl_parser = get_smpl_parser(int(gender_beta[0].item()))
    # pose_quat = sRot.from_rotvec(pose_aa.reshape(-1,3)
    #                                 ).as_quat().reshape(root.shape[0],-1,24,4)[...,:22,:]
    for i in tqdm(range(len(pose_aa))):#.shape[0])):
        # length = pose_aa.shape[1]
        # root_origin,pose_aa_origin,_ = self.change_fps(root[i,:length],
        #                                                 pose_quat[i,:length],
        #                                                 1/30.0,1/20.0)
        verts,_ = smpl_parser.get_joints_verts(
            pose=torch.from_numpy(pose_aa[i]),
            th_betas = gender_beta[1:][None,],
            th_trans = root[i])
        height = verts[..., 2].min(dim=-1)[0].mean().item()
        verts[..., 2] -= height
        smpl_model = SMPL().eval()
        # self.save_pkl(pose_aa_origin,self.gender_beta.numpy()
        #             ,root_origin,i)
        for j in range(verts.shape[0]):
            mesh = Trimesh(vertices=verts[j].cpu().tolist(),faces=smpl_model.faces)
            os.makedirs(f'{save_path}/rot/{i}',exist_ok=True)
            mesh.export(f'{save_path}/rot/{i}/mesh_{j}.obj')
            
if __name__ == "__main__":
    # path = "/workspace/jja3wx/jidong/EDGE/runs/newfgc_per_frame_4_150-760"#"../mint/pkl"#
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion_path",
        type=str,
        default="motions/",
        help="Where to load saved motions",
    )
    opt = parser.parse_args()
    path = opt.motion_path
    #
    filenames = glob.glob(f"{path}/*.pkl")
    root = []
    pose_aa = []
    joint3ds = []
    # for idx,filename in enumerate(filenames):
    #     filename = filename.split('/')[-1]
    #     with open('map_g.txt','a') as f:
    #         f.write(f'{idx} {filename}\n')
    # import ipdb; ipdb.set_trace()
    # male_smpl = get_smpl_parser(1)
    # offsets = male_smpl.get_offsets()
    # import ipdb; ipdb.set_trace()
    print(filenames)
    # import ipdb;ipdb.set_trace()
    for filename in filenames:
        pose_all = np.load(file=filename,allow_pickle=True)
        print(pose_all.keys())
        # try:
        #     trans = pose_all["smpl_trans"].squeeze()[::2]/pose_all["smpl_scaling"]
        #     pose = pose_all["smpl_poses"].squeeze()[::2]
        # except:
        #     trans = pose_all["pos"].squeeze()[::2]/pose_all["scale"]
        #     pose = pose_all["q"].squeeze()[::2]

        # trans[:,1] -= trans[:,1].mean()
        # print(pose_all["smpl_trans"].shape)
        # root.append(torch.as_tensor(trans))
        # pose_aa.append(pose)
        root.append(torch.as_tensor(pose_all["smpl_trans"].squeeze()[:600]))
        pose_aa.append(pose_all["smpl_poses"].squeeze()[:600])
        # root.append(pose_all["pos"][::2])
        # pose_aa.append(pose_all["q"][::2])
        # joint3d = pose_all.item()['pred_position'].reshape(-1,24,3)[::2][:600]
        # joint3ds.append(joint3d)
    # joint3d = torch.as_tensor(np.concatenate(joint3ds,axis=0))  
    # smpl = get_smpl_parser(0).to("cuda")
    # smpl.device = torch.device("cuda")
    # trans = torch.zeros((joint3d.shape[0],3)).to("cuda").reshape(10,-1,3)
    # pose_aa = torch.zeros((joint3d.shape[0],72)).to("cuda").reshape(10,-1,72)
    # trans = nn.Parameter(trans.detach(),requires_grad=True)
    # pose_aa = nn.Parameter(pose_aa.detach(),requires_grad=True)
    # betas = torch.zeros((joint3d.shape[0],10)).to("cuda").reshape(10,-1,10)
    # betas = nn.Parameter(betas.detach(),requires_grad=True)
    # joint3d = joint3d.reshape(10,-1,24,3).to("cuda")
    # arg = {'type':'LBFGS', 'max_iter':500, 'lr':1, 'tolerance_change': 1e-4, 'history_size':200}
    # opt = torch.optim.LBFGS([trans,pose_aa],
    #                         lr=1,
    #                         max_iter=500,
    #                         tolerance_change=1e-4,
    #                         max_eval=None,
    #                         history_size=200,
    #                         line_search_fn='strong_wolfe')
    ##########################################################
    # opt = torch.optim.Adam([trans,pose_aa],lr=0.01)
    # last_loss = 1
    # early_stop = 0
    # losses = []
    # for i in range(1,6000):
    #     for j in range(pose_aa.shape[0]):
    #         _,joint = get_joints_verts(pose_aa[j],torch.zeros(10)[None,].to("cuda"),trans[j],model=smpl)
    #         loss = torch.mean((joint-joint3d[j])**2)
    #         losses.append(loss.item())
    #         opt.zero_grad()
    #         loss.backward()
    #         opt.step()
    #     if i % 2000 == 0:
    #         for para in opt.param_groups:
    #             para["lr"] = para["lr"]/2
    #     print(f"epoch {i}:{sum(losses)/len(losses)}")
    ####################################################################
        # if (last_loss-loss)>0.0001:
        #     last_loss = loss
        #     early_stop=0
        # else:
        #     last_loss = loss
        #     early_stop+=1
        # if early_stop>40:
        #     lr = 0
        #     for para in opt.param_groups:
        #         para["lr"] = para["lr"]/2
        #         lr = para["lr"]
        #     print(f"lr to {lr}")
        #     early_stop=0
        
        # if early_stop>15:
            # break
    # torch.cuda.empty_cache()
    # ik = nn.DataParallel(IK_solver()).to("cuda")  
    # out = ik(torch.as_tensor(joint3d).to("cuda"))
    # # out = torch.concatenate([trans.reshape(-1,3),pose_aa.reshape(-1,72)],dim=-1).detach()
    # pose_aa_gt_flat = out[...,3:69].cpu().numpy().reshape(-1,22,3)
    # trans_gt_flat =out[...,:3].cpu().numpy()  
    # hand = np.array([[0.0,0.0,0.0]]).repeat(pose_aa_gt_flat.shape[0],0)
    # pose_aa_gt_flat = np.concatenate([pose_aa_gt_flat,hand[:,None,:],hand[:,None,:]],axis=1)
    # root = trans_gt_flat.reshape(-1,600,3)
    # pose_aa = pose_aa_gt_flat.reshape(-1,600,24,3)
    # root = np.stack(root,axis=0)
    # pose_aa = np.stack(pose_aa,axis=0)
    # root = root.reshape(40,-1,3)
    # pose_aa = pose_aa.reshape(40,-1,24,3)
    gender = torch.zeros(11)
    get_results_from_rot(pose_aa,root,gender,path)
