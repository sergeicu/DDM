import sys
import os 
import math
import torch
from torch import nn, einsum
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from . import loss

import nibabel as nib

from model.ddpm_modules.explore_noisy import *




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


def get_odd_even(data,select='odd', **kwargs):
    newdata = torch.zeros_like(data)
    assert newdata.ndim == 5
    if select=='odd':
        newdata[:,:,0::2,:,:] = data[:,:,0::2,:,:]
    else:
        newdata[:,:,1::2,:,:] = data[:,:,1::2,:,:]
        
    return newdata

def frobenius_norm(input_tensor):
    # substitutes torch.linalg.norm which is not available in earlier torch version 
    return torch.sqrt(torch.sum(input_tensor ** 2))

def grad_and_value(x_prev, x_0_hat, measurement, **kwargs):
    difference = measurement - get_odd_even(x_0_hat, select='odd', **kwargs) # get diff 
    norm = frobenius_norm(difference) # get L2 norm between measured image (odd-even) and predicted image (also within odd-even framework)
    norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0] # calculate gradient w.r.t. to L2 norm to be able to go into that direction
    
    return norm_grad, norm
             
                
def measurement_cond_fn(x_t, measurement, x_prev,x_0_hat, **kwargs):
    scale = 0.5 
    
    norm_grad, norm = grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
    x_t -= norm_grad * scale # we move predicted x_t (after odd_even) 
                             # towards the direction that will minimize the loss between it and the measured image
    
   
    return x_t, norm
    
                
            


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
            if self.conditional:
                sys.exit('not implemented for dps_duo')
                score = self.denoise_fn(torch.cat([condition_x, x], dim=1), t)
            else:
                # we are passing x because it already contains a torch.cat result of both vectors 
                score = self.denoise_fn(x, t) # returns an output with single dim ... is this correct? 

        
        # predict x_0 (not x_(t-1), but true x_0)
        x1 = x[:,0:1, :,:,:]
        x_recon1 = self.predict_start_from_noise(x1, t=t, noise=score)
        
        x2 = x[:,1:2, :,:,:]
        x_recon2 = self.predict_start_from_noise(x2, t=t, noise=score)

        if clip_denoised:
            x_recon1.clamp_(-1., 1.)
            x_recon2.clamp_(-1., 1.)

        model_mean1, posterior_variance1, posterior_log_variance1 = self.q_posterior(
            x_start=x_recon1, x_t=x1, t=t)
        
        model_mean2, posterior_variance2, posterior_log_variance2 = self.q_posterior(
            x_start=x_recon2, x_t=x2, t=t)     
           
        return (model_mean1,model_mean2), (posterior_variance1,posterior_variance2), (posterior_log_variance1,posterior_log_variance2),(x_recon1,x_recon2)

    def p_sample_loop_ddpm(self, x_in, nsample, continous=False,savename=None):
        
        clip_denoised=False # since we do not clip our values on prediction - we should not clip values during inference!
        
        
        # define device 
        device = self.betas.device
        
        # grab the image to denoise 
        S = x_in[:, 0:1]    
        # T = x_in[:, 1:2]  
        T = x_in[:, 0:1]      
        
        
        # create a little bit of noise and go backwards 100 times -> track every saved output 
        # create the first image -> pure noise -> ALTERNATIVELY add only a little bit of noise... 
        #S_i = torch.randn_like(S)
        #T_i = torch.randn_like(S)
        # step=100
        # self.num_timesteps = step 
        step=self.num_timesteps-1
        t = torch.full((S.shape[0],), step, device=device, dtype=torch.long)
        noise = torch.randn_like(S)
        S_i = self.q_sample(S, t=t, noise=noise) 
        T_i = self.q_sample(T, t=t, noise=noise) 
        
                
        
        
        save_noisy=True
        if save_noisy:
            myimage = S_i[0,0,:,:,:].permute(1,2,0).detach().cpu().numpy()
            newsavename=savename.replace("_denoised.nii.gz", f"_S_i_t{step}.nii.gz")
            imo = nib.Nifti1Image(myimage,affine=np.eye(4))
            nib.save(imo, newsavename)     

            myimage = T_i[0,0,:,:,:].permute(1,2,0).detach().cpu().numpy()
            newsavename=savename.replace("_denoised.nii.gz", f"_T_i_t{step}.nii.gz")
            imo = nib.Nifti1Image(myimage,affine=np.eye(4))
            nib.save(imo, newsavename)    
            
            myimage = S[0,0,:,:,:].permute(1,2,0).detach().cpu().numpy()
            newsavename=savename.replace("_denoised.nii.gz", f"_S.nii.gz")
            imo = nib.Nifti1Image(myimage,affine=np.eye(4))
            nib.save(imo, newsavename)     

            myimage = T[0,0,:,:,:].permute(1,2,0).detach().cpu().numpy()
            newsavename=savename.replace("_denoised.nii.gz", f"_T.nii.gz")
            imo = nib.Nifti1Image(myimage,affine=np.eye(4))
            nib.save(imo, newsavename)     
            
            print(f"Saved to: {os.path.dirname(newsavename)}")
             
        # define a vector of Ts from highest to lowest 
        pbar = tqdm(list(range(self.num_timesteps))[::-1])
        
        # save every n steps 
        save_every=100
        save_finer_after=100
        
        dps=True
        if dps: 
            
            # Forward measurement model (Ax + n)
            sigma = 0.05 # 5% noise (we are adding it to other image because it is necessary?)
            y1 = get_odd_even(S,select='odd') # -> i.e. select odd lines 
            y2 = get_odd_even(S,select='even') # -> i.e. select even lines             
            
            
            #y_n = y + torch.randn_like(S, device=device) * sigma  # add noise based on given variance to this image (just as a start...)
            
            # ADD NOISE TO EVEN LINES (THAT ARE EMPTY)
            # lets just add noise where the slices are zero
            y_n1 = y1.clone()
            y_n1[:,:,1::2,:,:] = y1[:,:,1::2,:,:] + torch.randn_like(S[:,:,1::2,:,:], device=device) * sigma  # add noise based on given variance to this image (just as a start...)

            # ADD NOISE TO ODD LINES  (THAT ARE EMPTY)
            y_n2 = y2.clone()
            y_n2[:,:,0::2,:,:] = y2[:,:,0::2,:,:] + torch.randn_like(S[:,:,0::2,:,:], device=device) * sigma  # add noise based on given variance to this image (just as a start...)

            
            if savename:
                myimage = S[0,0,:,:,:].permute(1,2,0).detach().cpu().numpy()
                newsavename=savename.replace("_denoised.nii.gz", f"_input.nii.gz")
                imo = nib.Nifti1Image(myimage,affine=np.eye(4))
                nib.save(imo, newsavename)     
                
                myimage = y1[0,0,:,:,:].permute(1,2,0).detach().cpu().numpy()
                newsavename=savename.replace("_denoised.nii.gz", f"_y1.nii.gz")
                imo = nib.Nifti1Image(myimage,affine=np.eye(4))
                nib.save(imo, newsavename)                            

                myimage = y_n1[0,0,:,:,:].permute(1,2,0).detach().cpu().numpy()
                newsavename=savename.replace("_denoised.nii.gz", f"_y_n1.nii.gz")
                imo = nib.Nifti1Image(myimage,affine=np.eye(4))
                nib.save(imo, newsavename)             

                myimage = y2[0,0,:,:,:].permute(1,2,0).detach().cpu().numpy()
                newsavename=savename.replace("_denoised.nii.gz", f"_y2.nii.gz")
                imo = nib.Nifti1Image(myimage,affine=np.eye(4))
                nib.save(imo, newsavename)                            

                myimage = y_n2[0,0,:,:,:].permute(1,2,0).detach().cpu().numpy()
                newsavename=savename.replace("_denoised.nii.gz", f"_y_n2.nii.gz")
                imo = nib.Nifti1Image(myimage,affine=np.eye(4))
                nib.save(imo, newsavename)             

                
                print(f"Original files saved to: {os.path.dirname(newsavename)}")               
        else:
            y_n1=S
            y_n2=S

        
        repeat_every_step = 1 # set to 1 get default

        for idx in pbar:
            for i in range(0,repeat_every_step):     # repeat N times        
                # required for DPS 
                S_i = S_i.requires_grad_()    
                T_i = T_i.requires_grad_()            
                
                # generate a vector of ts based on current value of t 
                t = torch.full((S.shape[0],), idx, device=device, dtype=torch.long)
                
                # make prediction using the model 
                condition = (y_n1,y_n2)
                input = torch.cat([S_i, T_i], dim=1)
                model_mean, posterior_variance, posterior_log_variance,x_recon = self.p_mean_variance(x=input,t=t, clip_denoised=clip_denoised, condition_x=condition)
                
                # unfold returned values 
                model_mean1, model_mean2 = model_mean
                posterior_log_variance1, posterior_log_variance2 = posterior_log_variance
                x_recon1, x_recon2 = x_recon
                
                # generate noise 
                noise = torch.randn_like(S)
                
                # predicted mean of noise + scaled down noise variance 
                if idx !=0:
                    out1 = model_mean1 + torch.exp(0.5 * posterior_log_variance1) * noise
                    out2 = model_mean2 + torch.exp(0.5 * posterior_log_variance2) * noise
                else:
                    out1 = model_mean1
                    out2 = model_mean2
                    

                ############
                # DPS part 
                ############
                
                # q_sample on y_n -> noisy_measurement 
                
                # diff operator -> calculate diff between odd-even image... -> but i dont get it because we already removed lines no? 
                # AH! i finally understand - where we do forward operator - y=operator.forwad(ref_img) -> noiser(y) -> this simulators the data!!! 
                # we can do this on images in real time or we can do it BEFORE we feed the images - in the dataloader... lol ... 
                # ... 
                # the only thing i dont understand is why we do the noiser part?? 
                
                if dps: 

                    
                    # generate identical noise to add to both images 
                    noise_ = torch.randn_like(S)
                    
                    # add noise to y_n properly according to timestep
                    noisy_measurement1 = self.q_sample(y_n1, t=t,noise=noise_)  
                    
                    S_i, distance1 = measurement_cond_fn(x_t=out1,
                                            measurement=y_n1,
                                            noisy_measurement=noisy_measurement1,
                                            x_prev=S_i,
                                            x_0_hat=x_recon1)     
                    S_i = S_i.detach()  
                    
                    # add noise to y_n properly according to timestep
                    noisy_measurement2 = self.q_sample(y_n2, t=t,noise=noise_)   
                    
                    T_i, distance2 = measurement_cond_fn(x_t=out2,
                                            measurement=y_n2,
                                            noisy_measurement=noisy_measurement2,
                                            x_prev=T_i,
                                            x_0_hat=x_recon2)     
                    T_i = T_i.detach()                  
                    pbar.set_postfix({'distance1': distance1.item()}, refresh=False)
                    pbar.set_postfix({'distance2': distance2.item()}, refresh=False)
                    
                else:
                    S_i = out1.detach()   
                    T_i = out2.detach()     
                    
                    
                # if idx == 99: 
                #     from IPython import embed; embed()
                # else:
                #     sys.exit('exited')
                    
                    
                    
                # save images 
                if idx==save_finer_after:
                    save_every = 10
                if savename is not None and idx%save_every==0:
                    myimage1 = out1[0,0,:,:,:].permute(1,2,0).detach().cpu().numpy()
                    myimage2 = out2[0,0,:,:,:].permute(1,2,0).detach().cpu().numpy()
                    
                    newsavename=savename.replace("_denoised.nii.gz", f"_t{idx}_denoised1.nii.gz")
                    imo = nib.Nifti1Image(myimage1,affine=np.eye(4))
                    nib.save(imo, newsavename)

                    newsavename=savename.replace("_denoised.nii.gz", f"_t{idx}_denoised2.nii.gz")
                    imo = nib.Nifti1Image(myimage2,affine=np.eye(4))
                    nib.save(imo, newsavename)

                    
                    print(f"Saving image at t={idx} to {newsavename}") 
                    
                                    
                # uncommented for now 
                ########################################
                # REPEAT AGAIN!!! - 10 times for every steps in last 50 steps 
                ########################################
                
                # if idx<50:
                    
                #     for ii in range(0,10):
                #         # required for DPS 
                #         S_i = S_i.requires_grad_()            
                        
                #         # generate a vector of ts based on current value of t 
                #         t = torch.full((S.shape[0],), idx, device=device, dtype=torch.long)
                            
                #         # make prediction using the model 
                #         condition = y_n
                #         model_mean, posterior_variance, posterior_log_variance,x_recon = self.p_mean_variance(x=S_i,t=t, clip_denoised=clip_denoised, condition_x=condition)
                        
                #         # generate noise 
                #         noise = torch.randn_like(S)
                        
                #         # predicted mean of noise + scaled down noise variance 
                #         if idx !=0:
                #             out = model_mean + torch.exp(0.5 * posterior_log_variance) * noise
                #         else:
                #             out = model_mean
                            

                #         ############
                #         # DPS part 
                #         ############
                        
                #         # q_sample on y_n -> noisy_measurement 
                        
                #         # diff operator -> calculate diff between odd-even image... -> but i dont get it because we already removed lines no? 
                #         # AH! i finally understand - where we do forward operator - y=operator.forwad(ref_img) -> noiser(y) -> this simulators the data!!! 
                #         # we can do this on images in real time or we can do it BEFORE we feed the images - in the dataloader... lol ... 
                #         # ... 
                #         # the only thing i dont understand is why we do the noiser part?? 
                        
                #         if dps: 

                            
                #             # add noise to y_n properly according to timestep
                #             noisy_measurement = self.q_sample(y_n, t=t)  
                            
                #             S_i, distance = measurement_cond_fn(x_t=out,
                #                                     measurement=y_n,
                #                                     noisy_measurement=noisy_measurement,
                #                                     x_prev=S_i,
                #                                     x_0_hat=x_recon)     
                #             S_i = S_i.detach()  
                #             pbar.set_postfix({'distance': distance.item()}, refresh=False)
                            
                #         else:
                #             S_i = out.detach()     
                            
                            
                            
                #         # save images 
                #         if idx==save_finer_after:
                #             save_every = 100
                #         if savename is not None and idx%save_every==0:
                #             myimage = out[0,0,:,:,:].permute(1,2,0).detach().cpu().numpy()
                            
                #             newsavename=savename.replace("_denoised.nii.gz", f"_t{idx}_denoised.nii.gz")
                #             imo = nib.Nifti1Image(myimage,affine=np.eye(4))
                #             nib.save(imo, newsavename)
                            
                #             print(f"Saving image at t={idx} to {newsavename}") 
                            
                                        
                
                
        # return the final value of S_i
        return S_i,T_i                 



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

    # @torch.no_grad()
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
        T_i = self.q_sample(x_start=x_in['T'], t=t, noise=noise) 
        if self.conditional: 
            sys.exit('Not implemented for two inputs')
            noise_pred = self.denoise_fn(torch.cat([x_in['S'],S_i], dim=1), t)
        else:
            noise_pred = self.denoise_fn(torch.cat([S_i, T_i], dim=1), t)

        l_pix = self.loss_func(noise, noise_pred) 

        l_pix = l_pix.sum() / int(b * c * d * h * w)
        
        return noise_pred, l_pix

    def forward(self, x, loss_lambda, *args, **kwargs):
        return self.p_losses(x, loss_lambda, *args, **kwargs)
