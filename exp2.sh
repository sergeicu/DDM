cd ~/w/code/diffusion/experiments/s20240209_oecorr/DDM
conda activate diffusers 
source ../DiffuseMorph/venv/bin/activate 


cd ~/w/code/diffusion/experiments/s20240209_oecorr/DDM
conda activate diffusers 
source venv_e2/bin/activate 




####################################################################
# Train DDPM only 
####################################################################

# train ddpm only 
git checkout ddpm_only 
python3 DDM_train.py -p train -c config/DDM_train_b8_ddpm_only.json

# train with UPDATED noise schedule  
    1A. images with condition
        python3 DDM_train.py -p train -c config/DDM_train_ddpm_wcond_1s.json # astral-cloud  -> experiments/DDM_train_240301_172419/checkpoint_b8_data_ED_ES/
        python3 DDM_train.py -p train -c config/DDM_train_ddpm_wcond.json # likely-salad ! -> experiments/DDM_train_240301_172720/checkpoint_b8_data_ED_ES/
    1B. images without condition: 
        python3 DDM_train.py -p train -c config/DDM_train_ddpm_nocond.json # gallant sponge ! -> experiments/DDM_train_240301_180405/checkpoint_b8_data_ED_ES/
        python3 DDM_train.py -p train -c config/DDM_train_ddpm_nocond_1s.json # stellar-dragon -> experiments/DDM_train_240301_190357/checkpoint_b8_data_ED_ES/
    2A. ES-ED images without condition... 

####################################################################
# Test DDPM only 
####################################################################

# test ddpm only 
git checkout ddpm_only 
python3 DDM_test.py -p test -c config/DDM_test_b8_ddpm_only.json 


####################################################################
# Test DPS only 
####################################################################


# test dps with conditional model (DDPM only trained)
git checkout dps_only
python3 DDM_test.py -p test -c config/DDM_test_b8_ddpm_only.json 

# test dps with no conditional model (DDPM trained with updated schedule)
git checkout dps_only
python3 DDM_test.py -p test -c config/DDM_test_ddpm_nocond.json



# TODO NEXT 
# train with UPDATED noise schedule for 
    # 1. 1 image with and without condition 
    # 2. all images with and without condition 
    # 3. ES-ED images without condition... 


# meanwhile todo tasks: 
    # 1. Integrate DPS loss function into DDPM only testing 
    # 2. Integrate DPS into DiffuseMorph with correct noise schedules etc 
    # 3. Build my own inferrence schedule with DiffuseMorph architecture 
    # 4. Build my own architecture (no additional VM network for computign the field map)

# Test with: 
    # 1. Odd-even slices 
    # 2. Simply interpolated slices... 






# start on e2 
e2 
srun -A crl -p crl-gpu -t 14:00:00 --qos=crl --gres=gpu:NVIDIA_A40:1 -n 8 --mem 128GB --pty /bin/bash 
srun -A crl -p crl-gpu -t 14:00:00 --qos=crl --gres=gpu:NVIDIA_A40:1  --pty /bin/bash 
cd ~/w/code/diffusion/experiments/s20240209_oecorr/DDM
conda activate diffusers 
source venv/bin/activate 



conda install -p /home/ch215616/w/miniconda2/envs/DDM pytorch nibabel numpy matplotlib scikit-learn scipy einops ipython