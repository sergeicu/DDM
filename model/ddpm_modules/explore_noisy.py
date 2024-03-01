import torch 
import matplotlib.pyplot as plt 
import numpy as np 
from tqdm import tqdm
import sys 
import nibabel as nib 

def calculate_snr(self, signal_image, noise_image):
    """
    Calculate the SNR (Signal-to-Noise Ratio) in decibels for the central 50x50 region of a 128x128 image.

    Parameters:
    - signal_image: PyTorch tensor of the original image.
    - noise_image: PyTorch tensor of the noise image.

    Returns:
    - SNR in decibels as a float.
    """
    
    # Extract the central 50x50 part of the images
    center_region_start = (128 - 50) // 2
    assert signal_image.ndim == 5 
    sl=signal_image.shape[2]//2
    signal_center = signal_image[:, :, sl-3:sl+3, center_region_start:center_region_start+50, center_region_start:center_region_start+50]
    noise_center = noise_image[:, :, sl-3:sl+3, center_region_start:center_region_start+50, center_region_start:center_region_start+50]
    
    # Calculate the mean square value (power) of the signal and noise
    signal_power = torch.mean(signal_center ** 2)
    noise_power = torch.mean(noise_center ** 2)
    
    # Calculate SNR in linear scale and convert to decibels
    snr_linear = signal_power / noise_power
    snr_db = 10 * torch.log10(snr_linear)
    
    return snr_linear.item(), snr_db.item()  # Return SNR value as a Python float
    
def plot_snr_decrease(self,snr_array, file_path='snr_decrease.png', title=""):
    plt.figure(figsize=(10, 6))
    plt.plot(snr_array, label='SNR over Time')
    plt.xlabel('Time Step')
    plt.ylabel('SNR')
    plt.title(f'Schedule: {title}')
    plt.legend()
    plt.savefig(file_path)
    plt.close()   
    
def plot_noise(self, filepath='noise_decrease.png', title=""):
    plt.figure(figsize=(10, 6))

    plt.plot(self.alphas_cumprod.cpu() ** 0.5, label=r"${\sqrt{\bar{\alpha}_t}}$")
    plt.plot((1 - self.alphas_cumprod.cpu()) ** 0.5, label=r"$\sqrt{(1 - \bar{\alpha}_t)}$")
    plt.legend(fontsize="x-large")
    plt.xlabel('Time Step')
    plt.ylabel('Rel Noise')
    plt.title(f'Schedule: {title}')
    plt.legend()
    plt.savefig(filepath)
    plt.close()   

    

def explore_noise_schedule_visually(self, x_in, savename=None):
    """These are helper functions that save the images at different noise levels"""
    
    
    device = self.betas.device

    
    # grab the image to denoise 
    S = x_in[:, :1]
    
    
    ################
    # Schedules
    ################        
    # default 
    default_schedule = {
        "schedule": "linear",
        "n_timestep": 2000,
        "linear_start": 1e-6,
        "linear_end": 1e-2
    }
    
    # 1e-4 -> 2e-2 -> from openai -> line 31 here - https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/gaussian_diffusion.py#L18
            
    # quad, linear, cosine, jsd 
    cos_schedule = {
        "schedule": "cosine",
        "n_timestep": 1000,
        "linear_start": 1e-6,
        "linear_end": 1e-2
    }
    
    gd_schedule = {
        "schedule": "linear",
        "n_timestep": 1000,
        "linear_start": 1e-4,
        "linear_end": 1e-2
    }
    
    cos_schedule2 = {
        "schedule": "cosine",
        "n_timestep": 500,
        "linear_start": 1e-6,
        "linear_end": 1e-2
    }        
    
    cos_schedule3 = {
        "schedule": "cosine",
        "n_timestep": 500,
        "linear_start": 1e-6,
        "linear_end": 1e-1
    }                
    
    cos_schedule4 = {
        "schedule": "cosine",
        "n_timestep": 1000,
        "linear_start": 1e-6,
        "linear_end": 5e-2
    }                
    
    cos_schedule5 = {
        "schedule": "cosine",
        "n_timestep": 1000,
        "linear_start": 1e-6,
        "linear_end": 5e-3
    }                        


    # nothing changes much on cosine changes...
    cos_schedule6 = {
        "schedule": "cosine",
        "n_timestep": 1000,
        "linear_start": 1e-6,
        "linear_end": 5e-4
    }                
    
    # good bad need to add more noise initially
    linear_schedule2 = {
        "schedule": "linear",
        "n_timestep": 1000,
        "linear_start": 1e-6,
        "linear_end": 2e-4
    }                                        

    # very bad - most image is noise 
    linear_schedule3 = {
        "schedule": "linear",
        "n_timestep": 1000,
        "linear_start": 1e-6,
        "linear_end": 2e-2
    }     
    
    # good bad need to add more noise initially - same as before 
    linear_schedule4 = {
        "schedule": "linear",
        "n_timestep": 1000,
        "linear_start": 1e-8,
        "linear_end": 2e-4
    }                   
    
    # very bad - most image is noise - same as before 
    linear_schedule5 = {
        "schedule": "linear",
        "n_timestep": 1000,
        "linear_start": 1e-6,
        "linear_end": 2e-3
    }                                        

    # BEST ONE!!!! FOR my ACDC images ... 
    linear_schedule6_5 = {
        "schedule": "linear",
        "n_timestep": 1000,
        "linear_start": 1e-6,
        "linear_end": 2e-3 
    }                                        

                                            
    # better - almost there! 
    linear_schedule6 = {
        "schedule": "linear",
        "n_timestep": 1000,
        "linear_start": 1e-6,
        "linear_end": 8e-4
    }                                        


    # BEST SO FAR -> only difference is num of timesteps is 2000->1000 and linear_end is 1e-4->1e-
    linear_schedule7 = {
        "schedule": "linear",
        "n_timestep": 1000,
        "linear_start": 1e-6,
        "linear_end": 1e-3
    }                  
    
    # better - almost there! 
    cos_schedule7 = {
        "schedule": "cosine",
        "n_timestep": 1000,
        "linear_start": 1e-6,
        "linear_end": 1e-3
    }                          
                            
                                                    
    #schedule=linear_schedule7
    # schedule=linear_schedule6
    schedule=linear_schedule6_5
    self.set_new_noise_schedule(schedule,device)
            
    # save every n steps 
    save_every=100
    
    from IPython import embed; embed()
    
    # define a vector of Ts from highest to lowest 
    num_timesteps = schedule['n_timestep']
    pbar = tqdm(list(range(num_timesteps))[::-1])        

    images = []
    snrs = []
    snrs_db = []
    for idx in pbar:
        
        # generate a vector of ts based on current value of t 
        t = torch.full((S.shape[0],), idx, device=device, dtype=torch.long)
        
        noise = torch.randn_like(S)
        S_i = self.q_sample(x_start=S, t=t, noise=noise) 
        
        # snr, snr_db = self.calculate_snr(S_i, noise)
        snr, snr_db = self.calculate_snr(S_i, noise)
        snrs.append(snr)
        snrs_db.append(snr_db)
            
        if savename is not None and idx%save_every==0:
            myimage = S_i[0,0,:,:,:].permute(1,2,0).detach().cpu().numpy()
            
            min_val, max_val = myimage.min(), myimage.max()
            normalized_image_array = (myimage - min_val) / (max_val - min_val)
            
            # saving normalized 
            images.append(normalized_image_array)
    
    
    # plot snr                 
    sched = f"{schedule['schedule']}_st_{schedule['linear_start']}_end_{schedule['linear_end']}_T_{num_timesteps}"
    sched = sched.replace(".", "_").replace("-", "_")             
    self.plot_snr_decrease(snrs, savename.replace(".nii.gz", f"_snr_" +sched+ ".png"), title=sched)        
    self.plot_snr_decrease(snrs_db, savename.replace(".nii.gz", f"_snr_db_" +sched+ ".png"),title=sched)        
    
    # plot noise schedule as is 
    self.plot_noise(savename.replace(".nii.gz", "_noisesched_" +sched+ ".png"),title=sched)

    # save final image
    final_image = np.moveaxis(np.array(images),0,-1)
    newsavename=savename.replace("_denoised.nii.gz", f"_t_every_{save_every}_{sched}.nii.gz")
    imo = nib.Nifti1Image(final_image,affine=np.eye(4))
    nib.save(imo, newsavename)

            
    print(f"Saving image at t={idx} to {newsavename}") 
            
    sys.exit("finished - exiting")
            
def explore_noise_schedule_visually_png(self, x_in, savename=None):
    """These are helper functions that save the images at different noise levels"""
    
    
    impath='/fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/diffusion/experiments/s20240209_oecorr/DDM/data/00003.png'
    from PIL import Image 
    
    im = np.array(Image.open(impath))
    rescaled_image_array = (im / 127.5) - 1

    device=self.betas.device
    imt=torch.Tensor(rescaled_image_array).to(device)
    

    
    ################
    # Schedules
    ################        
    # default 
    default_schedule = {
        "schedule": "linear",
        "n_timestep": 2000,
        "linear_start": 1e-6,
        "linear_end": 1e-2
    }
    
    gd_schedule = {
        "schedule": "linear",
        "n_timestep": 1000,
        "linear_start": 1e-4,
        "linear_end": 1e-2
    }
    

    
    # 1e-4 -> 2e-2 -> from openai -> line 31 here - https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/gaussian_diffusion.py#L18
            
    # quad, linear, cosine, jsd 
    cos_schedule = {
        "schedule": "cosine",
        "n_timestep": 1000,
        "linear_start": 1e-6,
        "linear_end": 1e-2
    }
    
    cos_schedule2 = {
        "schedule": "cosine",
        "n_timestep": 500,
        "linear_start": 1e-6,
        "linear_end": 1e-2
    }        
    
    cos_schedule3 = {
        "schedule": "cosine",
        "n_timestep": 500,
        "linear_start": 1e-6,
        "linear_end": 1e-1
    }                
    
    cos_schedule4 = {
        "schedule": "cosine",
        "n_timestep": 1000,
        "linear_start": 1e-6,
        "linear_end": 5e-2
    }                
    
    cos_schedule5 = {
        "schedule": "cosine",
        "n_timestep": 1000,
        "linear_start": 1e-6,
        "linear_end": 5e-3
    }                        


    # nothing changes much on cosine changes...
    cos_schedule6 = {
        "schedule": "cosine",
        "n_timestep": 1000,
        "linear_start": 1e-6,
        "linear_end": 5e-4
    }                
    
    # good bad need to add more noise initially
    linear_schedule2 = {
        "schedule": "linear",
        "n_timestep": 1000,
        "linear_start": 1e-6,
        "linear_end": 2e-4
    }                                        

    # very bad - most image is noise 
    linear_schedule3 = {
        "schedule": "linear",
        "n_timestep": 1000,
        "linear_start": 1e-6,
        "linear_end": 2e-2
    }     
    
    # good bad need to add more noise initially - same as before 
    linear_schedule4 = {
        "schedule": "linear",
        "n_timestep": 1000,
        "linear_start": 1e-8,
        "linear_end": 2e-4
    }                   
    
    # very bad - most image is noise - same as before 
    linear_schedule5 = {
        "schedule": "linear",
        "n_timestep": 1000,
        "linear_start": 1e-6,
        "linear_end": 2e-3
    }                                        
                                            
    # better - almost there! 
    linear_schedule6 = {
        "schedule": "linear",
        "n_timestep": 1000,
        "linear_start": 1e-6,
        "linear_end": 8e-4
    }                                        


    # BEST SO FAR -> only difference is num of timesteps is 2000->1000 and linear_end is 1e-4->1e-
    linear_schedule7 = {
        "schedule": "linear",
        "n_timestep": 1000,
        "linear_start": 1e-6,
        "linear_end": 1e-3
    }                  
    
    # better - almost there! 
    cos_schedule7 = {
        "schedule": "cosine",
        "n_timestep": 1000,
        "linear_start": 1e-6,
        "linear_end": 1e-3
    }                          
                            
                    
    from IPython import embed; embed()
                                                    
    schedule=gd_schedule
    self.set_new_noise_schedule(schedule,device)
            
    # save every n steps 
    save_every=100
    
    
    # define a vector of Ts from highest to lowest 
    num_timesteps = schedule['n_timestep']
    pbar = tqdm(list(range(num_timesteps))[::-1])        
    
    # plot snr                 
    sched = f"{schedule['schedule']}_st_{schedule['linear_start']}_end_{schedule['linear_end']}_T_{num_timesteps}"
    sched = sched.replace(".", "_").replace("-", "_")             
    
    for idx in pbar:
        
    
        # generate a vector of ts based on current value of t 
        t = torch.full((1,), idx, device=self.betas.device, dtype=torch.long)
        
        noise = torch.randn_like(imt).to(device)
        S_i = self.q_sample(x_start=imt, t=t, noise=noise) 
            
        if savename is not None and idx%save_every==0:
            myimage = S_i.detach().cpu().numpy()
            
            min_val, max_val = myimage.min(), myimage.max()
            normalized_image_array = (myimage - min_val) / (max_val - min_val)
            
            scaled_image_array = (normalized_image_array * 255).astype(np.uint8)
            
            newsavename = savename.replace("patient101_iso", "im_example_")
            newsavename=newsavename.replace("_denoised.nii.gz", f"_{sched}_t{idx}.png")
            

            image_to_save = Image.fromarray(scaled_image_array)
            image_to_save.save(newsavename)   
            
            print(f"Saving image at t={idx} to {newsavename}")              


            
    
    
    # plot noise schedule as is 
    self.plot_noise(savename.replace(".nii.gz", "_noisesched_" +sched+ ".png"),title=sched)


            
    
            
    sys.exit("finished - exiting")
            