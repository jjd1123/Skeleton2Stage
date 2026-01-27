import copy
import os
import pickle
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from p_tqdm import p_map
from pytorch3d.transforms import (axis_angle_to_quaternion,
                                  quaternion_to_axis_angle)
from tqdm import tqdm
import math

from dataset.quaternion import ax_from_6v, quat_slerp
from vis import skeleton_render

from .utils import extract, make_beta_schedule

def gaussian_smooth_1d(x, sigma=1.2):
    # x: [B,T]
    radius = int(math.ceil(3 * sigma))
    t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    xx = F.pad(x.unsqueeze(1), (radius, radius), mode="reflect")
    return F.conv1d(xx, k.view(1, 1, -1)).squeeze(1)

@torch.no_grad()
def smooth_height_by_contact_delta(full_pos, full_pose, static_label,
                                  joint_ids=(7,8,10,11),
                                  up_axis=2,
                                  sigma=1.2,
                                  dmax=0.05,
                                  eps_floor=0.0):
    """
    full_pos:  [B,T,3]  (CPU tensor也可以)
    full_pose: [B,T,24,3]
    static_label: [B,T-1,J] bool  (你代码里的 static_label)
    只修 full_pos[...,up_axis]，用接触段(静止段)把脚贴到 floor=eps_floor，并对修正量做高斯平滑。
    """
    B, T, _ = full_pos.shape
    device = full_pos.device

    # 1) 把 static_label 对齐到 [B,T]，并汇总到单通道 contact01
    #    你的是 (B, T-1, J)，代表 t->t+1 之间脚静止
    contact01 = static_label.any(dim=-1).float()         # [B, T-1]
    contact01 = torch.cat([contact01[:, :1], contact01], dim=1)  # [B, T]

    # 2) 每帧脚底高度（取 joint_ids 的最小高度）
    ids = torch.as_tensor(joint_ids, device=device, dtype=torch.long)
    foot_h = full_pose[:, :, ids, up_axis].min(dim=-1).values     # [B,T]

    foot_h_new = gaussian_smooth_1d(foot_h,sigma=1.2)
    full_pos[...,2] = full_pos[...,2] + contact01 * (- foot_h + foot_h_new)


    return full_pos


def identity(t, *args, **kwargs):
    return t

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        horizon,
        repr_dim,
        smpl,
        n_timestep=1000,
        schedule="linear",
        loss_type="l1",
        clip_denoised=True,
        predict_epsilon=True,
        guidance_weight=3,
        use_p2=False,
        cond_drop_prob=0.2,
    ):
        super().__init__()
        self.horizon = horizon
        self.transition_dim = repr_dim
        self.model = model
        self.ema = EMA(0.99)
        self.master_model = copy.deepcopy(self.model)
        self.origin_model = None#copy.deepcopy(self.model)

        self.cond_drop_prob = cond_drop_prob

        # make a SMPL instance for FK module
        self.smpl = smpl

        betas = torch.Tensor(
            make_beta_schedule(schedule=schedule, n_timestep=n_timestep)
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timestep = int(n_timestep)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.guidance_weight = guidance_weight

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # p2 weighting
        self.p2_loss_weight_k = 1
        self.p2_loss_weight_gamma = 0.5 if use_p2 else 0
        self.register_buffer(
            "p2_loss_weight",
            (self.p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
            ** -self.p2_loss_weight_gamma,
        )

        ## get loss coefficients and initialize objective
        self.loss_fn = F.mse_loss if loss_type == "l2" else F.l1_loss

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise
    
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    @torch.no_grad()
    def origin_model_predictions(self, x, cond, t, weight=None, clip_x_start = False):
        weight = weight if weight is not None else self.guidance_weight
        model_output = self.origin_model.guided_forward(x, cond, t, weight)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity
        
        if isinstance(model_output,tuple):
            model_output,_ = model_output

        x_start = model_output
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start
    
    def model_predictions(self, x, cond, t, weight=None, clip_x_start = False,return_kl=False):
        weight = weight if weight is not None else self.guidance_weight
        model_output = self.model.guided_forward(x, cond, t, weight)
        if return_kl:
            model_output_kl = self.origin_model.guided_forward(x, cond, t, weight)
            if isinstance(model_output_kl,tuple):
                model_output_kl,_ = model_output_kl
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity
        
        if isinstance(model_output,tuple):
            model_output,_ = model_output
        x_start = model_output
        x_start = maybe_clip(x_start)

        pred_noise = self.predict_noise_from_start(x, t, x_start)
        
        if return_kl:
            x_start_kl = maybe_clip(model_output_kl)
            # pred_noise_kl = self.predict_noise_from_start(x, t, x_start_kl)
            kl_loss = (x_start-x_start_kl)**2
            return pred_noise, x_start, kl_loss
        
        return pred_noise, x_start

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        # guidance clipping
        if t[0] > 1.0 * self.n_timestep:
            weight = min(self.guidance_weight, 0)
        elif t[0] < 0.1 * self.n_timestep:
            weight = min(self.guidance_weight, 1)
        else:
            weight = self.guidance_weight

        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.model.guided_forward(x, cond, t, weight)
        )

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    @torch.no_grad()
    def p_sample(self, x, cond, t):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, cond=cond, t=t
        )
        noise = torch.randn_like(model_mean)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(
            b, *((1,) * (len(noise.shape) - 1))
        )
        x_out = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return x_out, x_start

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        cond,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
    ):
        device = self.betas.device

        # default to diffusion over whole timescale
        start_point = self.n_timestep if start_point is None else start_point
        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = cond.to(device)

        if return_diffusion:
            diffusion = [x]

        for i in tqdm(reversed(range(0, start_point))):
            # fill with i
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x, _ = self.p_sample(x, cond, timesteps)

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x
        
    def ema_1d(self, x, alpha=0.85):
        y = torch.zeros_like(x)
        y[:, 0] = x[:, 0]
        for t in range(1, x.size(1)):
            y[:, t] = alpha * y[:, t-1] + (1 - alpha) * x[:, t]
        return y

    def hysteresis_binarize(self, prob, tau_on=0.8, tau_off=0.7, min_on=2, min_off=2):
        B, T = prob.shape
        out = torch.zeros((B, T), device=prob.device, dtype=torch.float32)
        for b in range(B):
            cur = 0
            run = 0
            for t in range(T):
                p = float(prob[b, t])
                if cur == 1:
                    if p < tau_off and run >= min_on:
                        cur = 0; run = 1
                    else:
                        run += 1
                else:
                    if p > tau_on and run >= min_off:
                        cur = 1; run = 1
                    else:
                        run += 1
                out[b, t] = float(cur)
        return out


    def foot_ground_guidance(self, x, t, normalizer, n_guide_steps=10, scale=0.5):
        model_variance = torch.exp(extract(
            self.posterior_log_variance_clipped, t, x.shape
        ))
        # print(model_variance)
        loss, grad = self.gradients(x, normalizer)
        # print(loss)

        for _ in range(n_guide_steps):
            x = x - scale * model_variance * grad
            loss, grad = self.gradients(x, normalizer)
            # print(loss)

        return x.detach()

    def gradients(self, x, normalizer, contact_ema_alpha=0.85,
              tau_on=0.8, tau_off=0.7,
              min_on=2, min_off=2,
              lam_float=1.0, lam_pen=1.0):
        b, s, _ = x.shape
        with torch.enable_grad():
            x.requires_grad_(True)
            x_1 = normalizer.unnormalize(x)
            if x_1.shape[2] == 156:
                contact = x_1[...,147:151]
                pos = x_1[...,:3]
                q = x_1[...,3:147].reshape(b,s,24,6)
            else:
                contact = x_1[...,:4]
                pos = x_1[...,4:7]
                q = x_1[...,7:].reshape(b,s,24,6)
            # contact = torch.any(contact>0.95,dim=-1)
            #######################################################################
            # CONTACT SMOOTH                                                      #
            #######################################################################
            contact_prob = contact.max(dim=-1).values  # [B,T]                  #
            contact_prob_s = self.ema_1d(contact_prob, alpha=contact_ema_alpha) #
            contact = self.hysteresis_binarize(                                 #
                contact_prob_s, tau_on=tau_on, tau_off=tau_off,                 #
                min_on=min_on, min_off=min_off                                  #
            )  # [B,T] 0/1 float                                                #
            #######################################################################
            # contact = torch.ones_like(contact)
            q = ax_from_6v(q).reshape(b,s,24,3)
            joints = self.smpl.forward(q,pos)
            # output = self.smpl_model(body_pose=q[...,3:].reshape(-1,69),global_orient=q[...,:3].reshape(-1,3),transl=pos.reshape(-1,3))
            # joints = output.joints[:,:24]
            # import ipdb; ipdb.set_trace()
            joints = joints.reshape(b,s,24,3)
            mean_floor = joints[...,2].min(dim=-1)[0].mean(dim=-1).mean(dim=-1).detach()
            # print(mean_floor)
            if_penetration = joints[...,2].min(dim=-1)[0] < mean_floor[...,None,None]
            # import ipdb; ipdb.set_trace()
            if_float = (joints[...,2].min(dim=-1)[0] > mean_floor[...,None,None])* contact
            # mask = torch.bitwise_or(joints[...,2].min(dim=-1)[0] < mean_floor[...,None], joints[...,2].min(dim=-1)[0] > (mean_floor[...,None] + 0.05))
            loss = ((if_penetration.detach()+if_float.detach())*(joints[...,2].min(dim=-1)[0]-mean_floor[...,None])**2).sum()
            grad = torch.autograd.grad([loss], [x])[0]
            # 0-3 contact label; 4-6 trans; 7-151 6d rotation
            # grad[:,:,7:7+7*6]=0
            # grad[:,:,7+9*6:]=0
            if x_1.shape[2] == 156:
                grad[:,:,3:]=0
                grad[:,:,:2]=0
                # pass
            else:    
                grad[:,:,7:]=0
                grad[:,:,:6]=0

            # print(grad.shape)
            x.detach()
        return loss, grad

    # def foot_ground_guidance(self, x, t, normalizer, n_guide_steps=10, scale=0.5):
    #     model_variance = torch.exp(extract(
    #         self.posterior_log_variance_clipped, t, x.shape
    #     ))
    #     # print(model_variance)
    #     loss, grad = self.gradients(x, normalizer)
    #     print(loss)

    #     for _ in range(n_guide_steps):
    #         x = x - scale * model_variance * grad
    #         # scale = scale
    #         loss, grad = self.gradients(x, normalizer)
    #         print(loss)

    #     return x.detach()

    # def gradients(self, x, normalizer):
    #     b, s, _ = x.shape
    #     with torch.enable_grad():
    #         x.requires_grad_(True)
    #         x_1 = normalizer.unnormalize(x)
    #         if x_1.shape[2] == 156:
    #             contact = x_1[...,147:151]
    #             pos = x_1[...,:3]
    #             q = x_1[...,3:147].reshape(b,s,24,6)
    #         else:
    #             contact = x_1[...,:4]
    #             pos = x_1[...,4:7]
    #             q = x_1[...,7:].reshape(b,s,24,6)
    #         contact = torch.any(contact>0.95,dim=-1)
    #         # contact = torch.ones_like(contact)
    #         q = ax_from_6v(q).reshape(b,s,24,3)
    #         joints = self.smpl.forward(q,pos)
    #         # output = self.smpl_model(body_pose=q[...,3:].reshape(-1,69),global_orient=q[...,:3].reshape(-1,3),transl=pos.reshape(-1,3))
    #         # joints = output.joints[:,:24]
    #         # import ipdb; ipdb.set_trace()
    #         joints = joints.reshape(b,s,24,3)
    #         mean_floor = joints[...,2].min(dim=-1)[0].mean(dim=-1).mean(dim=-1).detach()
    #         # print(mean_floor)
    #         if_penetration = joints[...,2].min(dim=-1)[0] < mean_floor[...,None,None]
    #         # import ipdb; ipdb.set_trace()
    #         if_float = (joints[...,2].min(dim=-1)[0] > mean_floor[...,None,None])* contact
    #         # mask = torch.bitwise_or(joints[...,2].min(dim=-1)[0] < mean_floor[...,None], joints[...,2].min(dim=-1)[0] > (mean_floor[...,None] + 0.05))
    #         loss = ((if_penetration.detach()+if_float.detach())*(joints[...,2].min(dim=-1)[0]-mean_floor[...,None])**2).sum()
    #         grad = torch.autograd.grad([loss], [x])[0]
    #         # 0-3 contact label; 4-6 trans; 7-151 6d rotation
    #         # grad[:,:,7:7+7*6]=0
    #         # grad[:,:,7+9*6:]=0
    #         if x_1.shape[2] == 156:
    #             grad[:,:,3:]=0
    #             grad[:,:,:2]=0
    #             # pass
    #         else:    
    #             grad[:,:,7:]=0
    #             grad[:,:,:6]=0

    #         # print(grad.shape)
    #         x.detach()
    #     return loss, grad
    
    def save_img(self,poses,step,normalizer):
        import matplotlib.pyplot as plt

        from matplotlib.colors import ListedColormap
        
        samples = normalizer.unnormalize(poses)
        if samples.shape[2] == 151:
            sample_contact, samples = torch.split(
                samples, (4, samples.shape[2] - 4), dim=2
            )
        else:
            sample_contact = None
        # do the FK all at once
        b, s, c = samples.shape
        pos = samples[:, :, :3].to("cuda")  # np.zeros((sample.shape[0], 3))
        q = samples[:, :, 3:].reshape(b, s, 24, 6)
        # go 6d to ax
        q = ax_from_6v(q).to("cuda")
            # squeeze the batch dimension away and render
        poses = self.smpl.forward(q, pos).detach().cpu().numpy()
        
        # 假设 poses 和 smpl_parents 已定义
        smpl_parents = [
        -1,
        0,
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        9,
        9,
        12,
        13,
        14,
        16,
        17,
        18,
        19,
        20,
        21
    ]
        poses = poses[0,[0,74,149]]
        poses = poses - poses[:, 0:1, :] + np.array([[0.8,0,1]])#np.array([[[0.2,0,0.8]]*24,[[0.8, 0, 1]]*24,[[1.4, 0, 1.2]]*24])
        num_steps = poses.shape[0]
        tmp = poses.copy()
        for j in range(3):
            poses = tmp[j]
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")

            point = np.array([0, 0, 1])
            normal = np.array([0, 0, 1])
            d = -point.dot(normal)
            xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 2), np.linspace(-1.5, 1.5, 2))
            z = (-normal[0] * xx - normal[1] * yy - d) * 1.0 / normal[2]

            # 绘制平面
            # ax.plot_surface(xx, yy, z, zorder=-11, cmap=cm.twilight)

            # 创建线条，初始状态为空数据
            lines = [ax.plot([], [], [], zorder=10, linewidth=1.5)[0] for _ in smpl_parents]
            scat = [ax.scatter([], [], [], c=[], zorder=10, s=0, cmap=ListedColormap(["r", "g", "b"])) for _ in range(4)]

            # 绘制骨架的一帧 (假设 frame_index 是我们要保存的帧索引)
            frame_index = 0
            frame_data = poses[frame_index]
            print(len(lines))
            print(len(smpl_parents))
            # for i in range(3):
                # frame_data = poses[i]
            frame_data = poses.reshape(-1, 3)
            for i, (line, parent) in enumerate(zip(lines, smpl_parents)):
                if parent != -1:
                    start = parent
                    end = i
                else:
                    continue
                print(start, end)
                line.set_data([frame_data[start, 0], frame_data[end, 0]], 
                            [frame_data[start, 1], frame_data[end, 1]])
                line.set_3d_properties([frame_data[start, 2], frame_data[end, 2]])

        # 填充数据到 scat
            for scatter, i in zip(scat, range(frame_data.shape[0])):
                scatter._offsets3d = (frame_data[i:i+1, 0], frame_data[i:i+1, 1], frame_data[i:i+1, 2])
                scatter.set_array(np.array([i]))
            # 这里需要填充数据到 lines 和 scat
            # 例如：
            # for line, (start, end) in zip(lines, smpl_parents):
            #     line.set_data([...])
            #     line.set_3d_properties([...])
            # for scatter, data in zip(scat, some_data_list):
            #     scatter._offsets3d = ([...], [...], [...])
            #     scatter.set_array([...])

            # 设置透明背景并保存图片
            ax.set_axis_off()
            fig.patch.set_alpha(0.0)
            ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

            # 保存图片
            plt.savefig(f'./image/frame_{step}_{j}.png', transparent=True)
    
    @torch.no_grad()
    def ddim_sample(self, shape, cond, **kwargs):
        if "return_logprob" in kwargs.keys():
            time_step = []
            log_probs = []
            latent = []
            kl_reward = []
            x_starts = []
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 1

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device = device)
        cond = cond.to(device)
        if "return_logprob" in kwargs.keys(): 
            # x_teacher = x.clone()
            # x_start_teacher = None
            # x_origin = x
            latent.append(x.detach().clone().cpu())
        x_start = None
        # self.save_img(x[[0]],1000,kwargs["normalizer"])
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step',disable="return_logprob" in kwargs.keys()):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(x, cond, time_cond, clip_x_start = self.clip_denoised)
            # pred_noise1, x_start1, *_ = self.model_predictions(x, cond, time_cond, clip_x_start = self.clip_denoised)
            # if (pred_noise1-pred_noise).sum() != 0 or (x_start-x_start1).sum() != 0:
            #     print("oh no")
            if "return_logprob" in kwargs.keys(): 
                x_starts.append(x_start)
                if self.origin_model!=None:
                    pred_noise_kl,x_start_kl, *_ = self.origin_model_predictions(x, cond, time_cond, clip_x_start = self.clip_denoised)
                # pred_noise_teacher,x_start_teacher,*_ = self.origin_model_predictions(x_teacher, cond, time_cond, clip_x_start = self.clip_denoised)
                # print(f"noise:{torch.sum((pred_noise-pred_noise_kl)**2)}")
                # print(f"start:{torch.sum((x_start-x_start_kl)**2)}")
                time_step.append(time)
            
            alpha = self.alphas_cumprod[time]
            if time_next>=0:
                alpha_next = self.alphas_cumprod[time_next]
            else:
                # if "return_logprob" in kwargs.keys():
                #     alpha_next = alpha
                # else:
                # print(alpha)
                alpha_next = torch.tensor(1.0).to(device)#(alpha+torch.tensor(1.0).to(device))/2#to compute logprob(x_0|x_1)
            # print(alpha.shape)
            # print(alpha_next.shape)
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
                  
            # self.save_img(x[[0]],time_next,kwargs["normalizer"])
                  
            # if time_next < 0:
            #     x = x_start

            ####compute log_prob
            if "return_logprob" in kwargs.keys(): 
                # x_origin = x_start_origin * alpha_next.sqrt() + \
                #             c * pred_noise_origin + \
                #             sigma * noise
                # x_teacher = x_start_teacher * alpha_next.sqrt() + \
                #   c * pred_noise_teacher + \
                #   sigma * noise
                prev_sample_mean = alpha_next.sqrt()*x_start+c*pred_noise
                prev_sample = prev_sample_mean + sigma*noise
                sigma = torch.clip(sigma,min=1e-6)
                log_prob = (-((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (sigma**2))
                    - torch.log(sigma)
                    - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi,device=device))))
                log_prob = log_prob.mean(dim=tuple(range(1,log_prob.ndim)))   
                if self.origin_model!=None:             
                    prev_sample_mean_kl = alpha_next.sqrt()*x_start_kl+c*pred_noise_kl
                    # prev_sample_kl = prev_sample_mean_kl+sigma*noise
                    log_prob_kl = (-((prev_sample.detach() - prev_sample_mean_kl) ** 2) / (2 * (sigma**2))
                        - torch.log(sigma)
                        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi,device=device))))
                    log_prob_kl = log_prob_kl.mean(dim=tuple(range(1,log_prob_kl.ndim)))
                    # log_prob = torch.where(torch.isnan(log_prob),
                    #                             torch.full_like(log_prob,0.0,device=device),
                    #                             log_prob)
                    # log_prob_kl = torch.where(torch.isnan(log_prob_kl),
                    #                             torch.full_like(log_prob_kl,0.0,device=device),
                                                # log_prob_kl)
                    if torch.isinf(log_prob_kl.sum()) or torch.isnan(log_prob_kl.sum()):
                        print(f"meet inf or nan at {time}")
                        print(log_prob)
                        print(log_prob_kl)                    
                log_probs.append(
                    log_prob.detach().clone().cpu()
                )
                latent.append(x.detach().clone().cpu())
                if self.origin_model!=None:
                    kl_reward.append(log_prob-log_prob_kl)
                    # kl_reward.append(log_prob)
                
        if "return_logprob" in kwargs.keys():
            if self.origin_model!=None:
                return (torch.tensor(time_step,dtype=torch.long),torch.stack(latent,dim=1),cond.detach().clone().cpu()
                        ,torch.stack(log_probs,dim=1),x.detach(),torch.where(torch.isinf(torch.stack(kl_reward,dim=1)), 
                        torch.full_like(torch.stack(kl_reward,dim=1), 0),torch.stack(kl_reward,dim=1)).mean(dim=1).detach(),torch.stack(x_starts,dim=1))#,x_teacher.detach())#,x_origin)
            else:
                return (torch.tensor(time_step,dtype=torch.long),torch.stack(latent,dim=1),cond.detach().clone().cpu()
                        ,torch.stack(log_probs,dim=1),x.detach(),torch.stack(x_starts,dim=1))#,x_teacher.detach())#,x_origin)
        return x
    
    @torch.no_grad()
    def long_ddim_sample(self, shape, cond, **kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 1
        
        motion = kwargs["motion"]#.to(device)
        
        if batch == 1:
            return self.ddim_sample(shape, cond,**kwargs)

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        weights = np.clip(np.linspace(0, self.guidance_weight * 2, sampling_timesteps), None, self.guidance_weight)
        # weights[:25] = 0
        # weights[:] = self.guidance_weight
        time_pairs = list(zip(times[:-1], times[1:], weights)) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device = device)
        cond = cond.to(device)
        
        assert batch > 1
        assert x.shape[1] % 2 == 0
        half = x.shape[1] // 2

        x_start = None
        # print(time_pairs)
        for time, time_next, weight in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            # print(x.shape,cond.shape)
            pred_noise, x_start, *_ = self.model_predictions(x, cond, time_cond, weight=weight, clip_x_start = self.clip_denoised) 

            if time in [19]: #TODO: test the best time sequence
                # scale_t = [500.0] * 20 + [50.0] * 20 + [0.5] * 8 + [0.5] * 2
                # scale_t = scale_t[time//20-1]
                # step_t = [5] * 10 + [5] * 10 + [5] * 20 + [10] * 8 + [10] * 2
                # step_t = step_t[time//20-1]
                scale_t = 2000.0
                step_t = 10
                # print(time)
                # import ipdb;ipdb.set_trace()
                x_start = self.foot_ground_guidance(x_start,time_cond,kwargs["normalizer"],n_guide_steps=step_t,scale=scale_t)

            if time_next < 0:
                x = x_start
                continue

            # if kwargs["player"]!=None and time in [39,59,79]:
            #     x_start = kwargs["player"].phys_proj_edge(kwargs["normalizer"],1,x_start)

            
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            
            if time > 0:
                # x[0,:motion.shape[1]] = self.q_sample(motion,t=torch.tensor([time_next],device=device))
                # the first half of each sequence is the second half of the previous one
                x[1:, :half] = x[:-1, half:]
            # print(time)
        return x

    @torch.no_grad()
    def inpaint_loop(
        self,
        shape,
        cond,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
    ):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = cond.to(device)
        if return_diffusion:
            diffusion = [x]

        mask = constraint["mask"].to(device)  # batch x horizon x channels
        value = constraint["value"].to(device)  # batch x horizon x channels

        start_point = self.n_timestep if start_point is None else start_point
        for i in tqdm(reversed(range(0, start_point))):
            # fill with i
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)

            # sample x from step i to step i-1
            x, _ = self.p_sample(x, cond, timesteps)
            # enforce constraint between each denoising step
            value_ = self.q_sample(value, timesteps - 1) if (i > 0) else x
            x = value_ * mask + (1.0 - mask) * x

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x

    @torch.no_grad()
    def long_inpaint_loop(
        self,
        shape,
        cond,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
    ):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = cond.to(device)
        if return_diffusion:
            diffusion = [x]

        assert x.shape[1] % 2 == 0
        if batch_size == 1:
            # there's no continuation to do, just do normal
            return self.p_sample_loop(
                shape,
                cond,
                noise=noise,
                constraint=constraint,
                return_diffusion=return_diffusion,
                start_point=start_point,
            )
        assert batch_size > 1
        half = x.shape[1] // 2

        start_point = self.n_timestep if start_point is None else start_point
        for i in tqdm(reversed(range(0, start_point))):
            # fill with i
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)

            # sample x from step i to step i-1
            x, _ = self.p_sample(x, cond, timesteps)
            # enforce constraint between each denoising step
            if i > 0:
                # the first half of each sequence is the second half of the previous one
                x[1:, :half] = x[:-1, half:] 

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x

    @torch.no_grad()
    def conditional_sample(
        self, shape, cond, constraint=None, *args, horizon=None, **kwargs
    ):
        """
            conditions : [ (time, state), ... ]
        """
        device = self.betas.device
        horizon = horizon or self.horizon

        return self.p_sample_loop(shape, cond, *args, **kwargs)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # reconstruct
        x_recon = self.model(x_noisy, cond, t, cond_drop_prob=self.cond_drop_prob)
        assert noise.shape == x_recon.shape

        model_out = x_recon
        if self.predict_epsilon:
            target = noise
        else:
            target = x_start

        # full reconstruction loss
        loss = self.loss_fn(model_out, target, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)

        # split off contact from the rest
        model_contact, model_out = torch.split(
            model_out, (4, model_out.shape[2] - 4), dim=2
        )
        target_contact, target = torch.split(target, (4, target.shape[2] - 4), dim=2)

        # velocity loss
        target_v = target[:, 1:] - target[:, :-1]
        model_out_v = model_out[:, 1:] - model_out[:, :-1]
        v_loss = self.loss_fn(model_out_v, target_v, reduction="none")
        v_loss = reduce(v_loss, "b ... -> b (...)", "mean")
        v_loss = v_loss * extract(self.p2_loss_weight, t, v_loss.shape)

        # FK loss
        b, s, c = model_out.shape
        # unnormalize
        # model_out = self.normalizer.unnormalize(model_out)
        # target = self.normalizer.unnormalize(target)
        # X, Q
        model_x = model_out[:, :, :3]
        model_q = ax_from_6v(model_out[:, :, 3:].reshape(b, s, -1, 6))
        target_x = target[:, :, :3]
        target_q = ax_from_6v(target[:, :, 3:].reshape(b, s, -1, 6))

        # perform FK
        model_xp = self.smpl.forward(model_q, model_x)
        target_xp = self.smpl.forward(target_q, target_x)

        fk_loss = self.loss_fn(model_xp, target_xp, reduction="none")
        fk_loss = reduce(fk_loss, "b ... -> b (...)", "mean")
        fk_loss = fk_loss * extract(self.p2_loss_weight, t, fk_loss.shape)

        # foot skate loss
        foot_idx = [7, 8, 10, 11]

        # find static indices consistent with model's own predictions
        static_idx = model_contact > 0.95  # N x S x 4
        model_feet = model_xp[:, :, foot_idx]  # foot positions (N, S, 4, 3)
        model_foot_v = torch.zeros_like(model_feet)
        model_foot_v[:, :-1] = (
            model_feet[:, 1:, :, :] - model_feet[:, :-1, :, :]
        )  # (N, S-1, 4, 3)
        model_foot_v[~static_idx] = 0
        foot_loss = self.loss_fn(
            model_foot_v, torch.zeros_like(model_foot_v), reduction="none"
        )
        foot_loss = reduce(foot_loss, "b ... -> b (...)", "mean")

        losses = (
            0.636 * loss.mean(),
            2.964 * v_loss.mean(),
            0.646 * fk_loss.mean(),
            10.942 * foot_loss.mean(),
        )
        return sum(losses), losses

    def loss(self, x, cond, t_override=None):
        batch_size = len(x)
        if t_override is None:
            t = torch.randint(0, self.n_timestep, (batch_size,), device=x.device).long()
        else:
            t = torch.full((batch_size,), t_override, device=x.device).long()
        return self.p_losses(x, cond, t)

    def forward(self, x, cond, t_override=None):
        return self.loss(x, cond, t_override)

    def partial_denoise(self, x, cond, t):
        x_noisy = self.noise_to_t(x, t)
        return self.p_sample_loop(x.shape, cond, noise=x_noisy, start_point=t)

    def noise_to_t(self, x, timestep):
        batch_size = len(x)
        t = torch.full((batch_size,), timestep, device=x.device).long()
        return self.q_sample(x, t) if timestep > 0 else x

    def compute_freezing(self,recon_x,normalizer):
        n_joints = 24
        b, constant_length = recon_x.shape[:2]
        sample = normalizer.unnormalize(recon_x)
        trans = sample[...,4:7].reshape(b,3,constant_length//3,3) #b*l*3
        pose_6d = sample[...,7:].reshape(b,3,constant_length//3,n_joints,6)
        # sample_x = normalizer.unnormalize(x)
        # trans_x = sample_x[...,4:7].reshape(b,3,constant_length//3,3) #b*l*3
        # pose_6d_x = sample_x[...,7:].reshape(b,3,constant_length//3,n_joints,6)
        trans_freeze = ((trans[...,1:,:]-trans[...,:-1,:])**2).mean(dim=(-1,-2))
        # trans_x_freeze = ((trans_x[...,1:,:]-trans_x[...,:-1,:])**2).mean(dim=(-1,-2))
        pose_6d_freeze = ((pose_6d[:,:,1:]-pose_6d[:,:,:-1])**2).mean(dim=(-1,-2,-3))
        # pose_6d_x_freeze = ((pose_6d_x[:,:,1:]-pose_6d_x[:,:,:-1])**2).mean(dim=(-1,-2,-3))
        # trans_rate = (trans_freeze/trans_x_freeze).mean(dim=1)
        # pose_rate = (pose_6d_freeze/pose_6d_x_freeze).mean(dim=1)
        # rate = torch.where((trans_rate+pose_rate)/2>1,1.0,(trans_rate+pose_rate)/2)
        rate = trans_freeze.mean(dim=1)+pose_6d_freeze.mean(dim=1)
        return rate
    
    def render_sample(
        self,
        shape,
        cond,
        normalizer,
        epoch,
        render_out,
        fk_out=None,
        name=None,
        sound=True,
        mode="normal",
        noise=None,
        constraint=None,
        sound_folder="ood_sliced",
        start_point=None,
        render=True,
        i = 0,
        player=None,
        motion = None
    ):
        if isinstance(shape, tuple):
            if mode == "inpaint":
                func_class = self.inpaint_loop
            elif mode == "normal":
                func_class = self.ddim_sample
            elif mode == "long":
                func_class = self.long_ddim_sample
            else:
                assert False, "Unrecognized inference mode"
            samples = (
                func_class(
                    shape,
                    cond,
                    noise=noise,
                    constraint=constraint,
                    start_point=start_point,
                    normalizer = normalizer,
                    motion = motion,
                    player = player
                )
                .detach()
                .cpu()
            )
        else:
            samples = shape
        if player != None:
            # pass
            samples = player.phys_proj_edge(normalizer,1,samples)
        # print(self.compute_freezing(samples,normalizer))
        # self.save_img(samples,-20,normalizer)
        # from get_obj import get_results_from_rot
        samples = normalizer.unnormalize(samples)

        if samples.shape[2] == 151:
            sample_contact, samples = torch.split(
                samples, (4, samples.shape[2] - 4), dim=2
            )
        elif samples.shape[2] == 156:
            samples, sample_contact = torch.split(
                samples, (147, samples.shape[2] - 147), dim=2
            )
        else:
            sample_contact = None
        # do the FK all at once
        b, s, c = samples.shape
        pos = samples[:, :, :3].to(cond.device)  # np.zeros((sample.shape[0], 3))
        q = samples[:, :, 3:].reshape(b, s, 24, 6)
        # go 6d to ax
        q = ax_from_6v(q).to(cond.device)

        # gender = torch.zeros(11)
        # get_results_from_rot(q.detach().cpu().numpy(),pos.detach().cpu(),gender,"demo_obj")

        if mode == "long":
            b, s, c1, c2 = q.shape
            assert s % 2 == 0
            half = s // 2
            if b > 1:
                # if long mode, stitch position using linear interp

                fade_out = torch.ones((1, s, 1)).to(pos.device)
                fade_in = torch.ones((1, s, 1)).to(pos.device)
                fade_out[:, half:, :] = torch.linspace(1, 0, half)[None, :, None].to(
                    pos.device
                )
                fade_in[:, :half, :] = torch.linspace(0, 1, half)[None, :, None].to(
                    pos.device
                )

                pos[:-1] *= fade_out
                pos[1:] *= fade_in

                full_pos = torch.zeros((s + half * (b - 1), 3)).to(pos.device)
                idx = 0
                for pos_slice in pos:
                    full_pos[idx : idx + s] += pos_slice
                    idx += half

                # stitch joint angles with slerp
                slerp_weight = torch.linspace(0, 1, half)[None, :, None].to(pos.device)

                left, right = q[:-1, half:], q[1:, :half]
                # convert to quat
                left, right = (
                    axis_angle_to_quaternion(left),
                    axis_angle_to_quaternion(right),
                )
                merged = quat_slerp(left, right, slerp_weight)  # (b-1) x half x ...
                # convert back
                merged = quaternion_to_axis_angle(merged)

                full_q = torch.zeros((s + half * (b - 1), c1, c2)).to(pos.device)
                full_q[:half] += q[0, :half]
                idx = half
                for q_slice in merged:
                    full_q[idx : idx + half] += q_slice
                    idx += half
                full_q[idx : idx + half] += q[-1, half:]

                # unsqueeze for fk
                full_pos = full_pos.unsqueeze(0)
                full_q = full_q.unsqueeze(0)
            else:
                full_pos = pos
                full_q = q
            full_pose = (
                self.smpl.forward(full_q, full_pos).detach().cpu()#.numpy()
            )  # b, s, 24, 3
            full_pos = full_pos.detach().cpu()
            full_q = full_q.detach().cpu()
            joint_ids = [7,8,10,11]
            pred_j3d_static = full_pose[:, :, joint_ids]  # (B, L, J, 3)
            pred_j_disp = pred_j3d_static[:, 1:] - pred_j3d_static[:, :-1]  # (B, L-1, J, 3)
            static_conf_logits = torch.zeros((pred_j_disp.shape[0],pred_j_disp.shape[1],pred_j_disp.shape[2]))
            # print(sample_contact.shape)
            for tmp_i, contact in enumerate(sample_contact):
                # import ipdb;ipdb.set_trace()
                # print(idx)
                if tmp_i == sample_contact.shape[0]-1:
                    static_conf_logits[:,tmp_i*75:tmp_i*75+149] = contact[:149]
                    break
                static_conf_logits[:,tmp_i*75:tmp_i*75+150] = contact
            static_label = static_conf_logits > 0.95  # (B, L-1, J)
            static_label_sumJ = static_label.sum(-1, keepdim=True)  # (B, L-1, 1)
            static_label_sumJ = torch.clamp_min(static_label_sumJ, 1)  # replace 0 with 1
            pred_disp_sumJ = (pred_j_disp * static_label[..., None]).sum(-2)  # (B, L-1, 3)
            pred_disp = pred_disp_sumJ / static_label_sumJ  # (B, L-1, 3)
            pred_disp[:, :, 2] = 0  # do not modify z

            # Overwrite results (for-loop)
            for tmp_i in range(1, full_pose.shape[1]):
                full_pos[:, tmp_i:] -= pred_disp[:, [tmp_i - 1]]
                full_pose[:, tmp_i:] -= pred_disp[:, [tmp_i - 1], None]

            # Put the sequence on the ground by -min(y), this does not consider foot height, for o3d vis
            ground_y = full_pose[..., 2].flatten(-2).mean(dim=-1)[0]  # (B,)  Minimum y value
            full_pos[..., 2] -= ground_y
            full_pose[...,0,2] -= ground_y
            ######################################################################################
            # SMOOTH GRAVITY DIRECTION FOR JUMP JITTERING
            ######################################################################################
            full_pos = smooth_height_by_contact_delta(     
                full_pos=full_pos,                         
                full_pose=full_pose,                       
                static_label=static_label,  # 你上面算出来的  
                joint_ids=joint_ids,        # [7,8,10,11]  
                up_axis=2,                                 
                sigma=1.2,                  # 30fps可以从1.0~1.5试
                dmax=0.06,                  # 每帧最大修正(米)，按你数据调
                eps_floor=0.0               # 想留一点不穿地余量可设0.005
            )

            # 如果你要保存 full_pose 也一致，建议更新 full_pose（需要重算FK更准确）
            # 这里最稳的是再跑一次 forward:
            full_pose = self.smpl.forward(full_q.cuda(), full_pos.cuda()).detach().cpu()
            ######################################################################################
            # squeeze the batch dimension away and render
            # skeleton_render(
            #     full_pose[0],
            #     epoch=f"{epoch}",
            #     out=render_out,
            #     name=name,
            #     sound=sound,
            #     stitch=True,
            #     sound_folder=sound_folder,
            #     render=render,
            #     i = i
            # )
            if fk_out is not None:
                outname = f'{epoch}_{"_".join(os.path.splitext(os.path.basename(name[0]))[0].split("_")[:-1])}.pkl'
                Path(fk_out).mkdir(parents=True, exist_ok=True)
                pickle.dump(
                    {
                        "smpl_poses": full_q.squeeze(0).reshape((-1, 72)).cpu().numpy(),
                        "smpl_trans": full_pos.squeeze(0).cpu().numpy(),
                        "full_pose": full_pose[0].numpy(),
                    },
                    open(os.path.join(fk_out, str(i)+outname), "wb"),
                )
            return

        poses = self.smpl.forward(q, pos).detach().cpu().numpy()
        sample_contact = (
            sample_contact.detach().cpu().numpy()
            if sample_contact is not None
            else None
        )

        def inner(xx):
            num, pose = xx
            filename = name[num] if name is not None else None
            contact = sample_contact[num] if sample_contact is not None else None
            skeleton_render(
                pose,
                epoch=f"e{epoch}_b{num}",
                out=render_out,
                name=filename,
                sound=sound,
                contact=contact,
            )

        p_map(inner, enumerate(poses))

        if fk_out is not None and mode != "long":
            Path(fk_out).mkdir(parents=True, exist_ok=True)
            for num, (qq, pos_, filename, pose) in enumerate(zip(q, pos, name, poses)):
                path = os.path.normpath(filename)
                pathparts = path.split(os.sep)
                pathparts[-1] = pathparts[-1].replace("npy", "wav")
                # path is like "data/train/features/name"
                pathparts[2] = "wav_sliced"
                audioname = os.path.join(*pathparts)
                outname = f"{epoch}_{num}_{pathparts[-1][:-4]}.pkl"
                pickle.dump(
                    {
                        "smpl_poses": qq.reshape((-1, 72)).cpu().numpy(),
                        "smpl_trans": pos_.cpu().numpy(),
                        "full_pose": pose,
                    },
                    open(f"{fk_out}/{outname}", "wb"),
                )
