import sys
import math
import torch
from torch import nn, einsum
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from . import loss

import nibabel as nib

from explore_noisy import *




def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac): # sv407WARNING - completely unused function - is part of make_beta_schedule
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3): # sv407WARNING - completely unused function - is part of set_new_noise_schedule
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False): # sv407WARNING - completely unused function
    def repeat_noise(): return torch.randn(
        (1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))

    def noise(): return torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        diffusion_module, # deformation_module,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None
    ):
        super().__init__()
        self.channels = channels
        self.denoise_fn = diffusion_module
        #self.field_fn = deformation_module
        self.conditional = conditional
        self.loss_type = loss_type
        if schedule_opt is not None: # sv407
            pass

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()
        self.loss_ncc = loss.crossCorrelation3D(1, kernel=(9, 9, 9)).to(device)
        self.loss_reg = loss.gradientLoss("l2").to(device)


    def set_new_noise_schedule(self, schedule_opt, device):  
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start, t): 
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise): 
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t): 
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None): 
        with torch.no_grad():
            score = self.denoise_fn(torch.cat([condition_x, x], dim=1), t)

        
        # predict x_0 (not x_(t-1), but true x_0)
        x_recon = self.predict_start_from_noise(x, t=t, noise=score)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    

        

    def p_sample_loop_ddpm(self, x_in, nsample, continous=False,savename=None):
        
        # define device 
        device = self.betas.device
        
        # grab the image to denoise 
        S = x_in[:, :1]
        
        # create the first image -> pure noise -> ALTERNATIVELY add only a little bit of noise... 
        S_i = torch.randn_like(S)
        
        # set the condition to zeros or one - as a test. 
        # S = torch.zeros_like(S)
        
        # define a vector of Ts from highest to lowest 
        self.num_timesteps = 2000 # i think the gradients must explore - or something like this - when t is very high... (and outside of range of trained t)
        pbar = tqdm(list(range(self.num_timesteps))[::-1])
        
        # save every n steps 
        save_every=500
        save_finer_after=500
        
        for idx in pbar:
            
            # generate a vector of ts based on current value of t 
            t = torch.full((S.shape[0],), idx, device=device, dtype=torch.long)
                
            # make prediction using the model 
            model_mean, posterior_variance, posterior_log_variance = self.p_mean_variance(x=S_i,t=t, clip_denoised=True, condition_x=S)
            
            # generate noise 
            noise = torch.randn_like(S)
            
            # nonzero mask - as long as t != 0 
            if idx == 0:
                nonzero_mask = torch.zeros_like(S)
            else:
                nonzero_mask = torch.ones_like(S)
            
            # predicted mean of noise + scaled down noise variance 
            S_i = model_mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance) * noise
            
            if idx==save_finer_after:
                save_every = 50
            if savename is not None and idx%save_every==0:
                myimage = S_i[0,0,:,:,:].permute(1,2,0).detach().cpu().numpy()
                
                newsavename=savename.replace("_denoised.nii.gz", f"_t{idx}_denoised.nii.gz")
                imo = nib.Nifti1Image(myimage,affine=np.eye(4))
                nib.save(imo, newsavename)
                
                print(f"Saving image at t={idx} to {newsavename}") 
                
            
        # return the final value of S_i
        return S_i

    def p_sample_loop(self, x_in, nsample, continous=False):
        device = self.betas.device
        S = x_in[:, :1]
        T = x_in[:, 1:]
        x_0 = x_in[:, 1:] # sv407WARNING - this does not make any sense - we are feeding the same data (not noisy 'T') to the network twice 
                            # during training we feed noised version of T, and here we just feed 'T' twice - it makes zero sense...
        b, c, d, h, w = S.shape

        with torch.no_grad():
            t = torch.full((b,), 0, device=device, dtype=torch.long) # sv407WARNING - this does not make sense - we are basically telling the network t is equal to zero (i.e. n noise!) why?? the whole markovian step is lost here....
            score = self.denoise_fn(torch.cat([S, T, x_0], dim=1), t) # sv407WARNING - why is our first step of registration - adding zero noise?? so we NEVER use ddpm in inferrence? why?? 
                            # dim=1 - refers to how we MERGE the files together - we merge them across CHANNEL dimension. Because dim=0 -> batch, dim=1 -> channel, dim=2 -> slices, dim=3&4 -> inplane dimensions

            gamma = np.linspace(0, 1, nsample) # sv407WARNING - this is where we set the number of times we will iterate through markovian chain..with the same T (??? doesnt make any sense!)
                                                # in short - nS relates to how many interpolation steps we want between deformations...
            b, c, d, h, w = x_0.shape
            flow_stack = torch.zeros([1, 3, d, h, w], device=device) # sv407 - our initial deformation estimate is nothing (just zeros) -> this is why nS of 1 or 0 gets met flow field of zero...... needs to be at least a list of values... 
            code_stack = score  # sv407WARNING - our first noise estimate is the estimate of noise from completely noise free images... WHY??? 
            defm_stack = S  # sv407 - we start from S... and then add deformed images that will turn into T 


            # sv407WARNING - basically - at inferrence we are not using the markovian nature of DDPM at all - we just do single pass through it to estimate some score... that we dont even use in the end ... 
            # so what was the point of using DDPM in the first place in training? to estimate some estimate of the score - that we pass through in T steps? Ok... but why? 
            for i in (gamma):
                print('-------- Deform with gamma=%.3f' % i)
                code_i = score * (i)  # sv407WARNING - it is basically trying to compute a linearly spaced smooth continuum between two images by multiplying by some fractional estimate between 0 (target) and 1 (source) images ... 
                S_phi_i, phi_i = self.field_fn(torch.cat([S, code_i], dim=1)) # sv407 - here is where they pass it via voxelmorph... [0] is output and [1] is the estimated field
                code_stack = torch.cat([code_stack, code_i], dim=0) # sv407 - we add new score estimate to prev estimate * multiplied by some ratio... 
                defm_stack = torch.cat([defm_stack, S_phi_i], dim=0) # sv407 - we add new estimate of deformed image to prev estimate (in a list)
                flow_stack = torch.cat([flow_stack, phi_i], dim=0) # sv407 - we add new estimate of deformation field to prev estimate (in a list) 

        if continous:
            return code_stack, defm_stack, flow_stack
        else:
            return code_stack[-1], S_phi_i, flow_stack # if not continuous - we just return the LAST known image (which is basically registration in one step)

    @torch.no_grad()
    def ddm_inference(self, x_in, nsample, continous=False,inference_type='DDM',savename=None): # sv407 - this is how inferrence happens... i.e. how we compute multiple steps... 
        if inference_type=='DDM':
            return self.p_sample_loop(x_in, nsample, continous) # sv407 - p_sample is the iterative backwards process, while q_sample is the noise adding (forwards) process
        elif inference_type == 'DDPM': 
            return self.p_sample_loop_ddpm(x_in, nsample, continous,savename=savename) # sv407 - p_sample is the iterative backwards process, while q_sample is the noise adding (forwards) process
        elif inference_type == 'explore_noise':
            self.explore_noise_schedule_visually(x_in, savename=savename)
        elif inference_type == 'explore_noise_png':
            self.explore_noise_schedule_visually_png(x_in, savename=savename)
        else: 
            sys.exit('WRONG inference type')

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        # fix gama
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )

    def p_losses(self, x_in, loss_lambda, noise=None):
        [b, c, d, h, w] = x_in['S'].shape
        t = torch.randint(0, self.num_timesteps, (b,), device=x_in['S'].device).long()  
        noise = default(noise, lambda: torch.randn_like(x_in['S'])) 
        S_i = self.q_sample(x_start=x_in['S'], t=t, noise=noise) 
        # from IPython import embed; embed()
        noise_pred = self.denoise_fn(torch.cat([x_in['S'],S_i], dim=1), t)

        l_pix = self.loss_func(noise, noise_pred) 

        l_pix = l_pix.sum() / int(b * c * d * h * w)
        
        return noise_pred, l_pix

    def forward(self, x, loss_lambda, *args, **kwargs):
        return self.p_losses(x, loss_lambda, *args, **kwargs)
