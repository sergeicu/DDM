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
    # with dps 
    experiments/DDM_test_240302_004621/results/

    # without dps 
    experiments/DDM_test_240302_004721/results/

# test dps by ONLY adding noise to bits which are zero...
    # with dps, only adding noise to slices which are zero
    experiments/DDM_test_240302_005655/results/
    # -> better 

# let's try by starting from idx 500
    experiments/DDM_test_240302_010255/results
    # -> TERRIBLE. very noisy. 


# let's try to load a later network? 
    experiments/DDM_test_240302_010752/results/
    # loading after 12000 steps instead of 8000 ...(600 epochs instead of 400)
    # possibly better - but not TOO much...


# let's try to load the network that was trained on one image... 
    python3 DDM_test.py -p test -c config/DDM_test_ddpm_nocond_1.json
    experiments/DDM_test_240302_011625/results/
    # loading after 12000 steps instead of 8000 ...(600 epochs instead of 400)
    # possibly better - but not TOO much...

    # load the network from correct patient (001.mat) -> we changed the dataloader to point to that patient always
    experiments/DDM_test_240302_012329/results 
        # results are much better!! 

    # load the network and test on another patient (010.mat) -> should be completely random 
    experiments/DDM_test_240302_012705/checkpoint_b8_data_ED_ES
        # results are very good also! interesting!!! 
            # i wish I had a 3D network that is actually good - it would be so easy to do DPS with it :( 

    # let's iterate 100 times more over the network 
        # 100 at the end (at t=0) -> extra 100 steps
        experiments/DDM_test_240302_013631/results/
            # >> 100 extra at the end is MUCH better than the whole schedule, but it is superceded by the other ones below 

        # 5 times for every time (aka 5000 steps)
        experiments/DDM_test_240302_013252/results/
            # >> similar performance as the one below (10 times per step in last 50 steps)

        # 10 times for every step in the last 50 steps -> extra 500 steps 
        experiments/DDM_test_240302_013756/results/


        # >>> ALL OF THESE ARE MUCH BETTER THEN ANY PREVIOUS STEPS!!!!

    


# let's try to train a tandem 3D network - where both even and odd slices are loaded... with single or two different noises!!! 

# 

# also need to train a better 3D network that actually learns something WITHOUT conditionals lol ... (these DiffuseMorph networks are totally worthless)



# BTW - training 3D model failed - maybe i can learn to predict not from pure noise but from say 'noisy'? image? 


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