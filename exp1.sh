cd ~/w/code/diffusion/experiments/s20240209_oecorr/DDM
conda activate diffusers 
source ../DiffuseMorph/venv/bin/activate 

# train 
python3 DDM_train.py -p train -c config/DDM_train_b8.json
python3 DDM_train.py -p train -c config/DDM_train_b8_motion_oe.json
python3 DDM_train.py -p train -c config/DDM_train_b8_nomotion_oe.json


# test 
python3 DDM_test.py -p test -c config/DDM_test_b8.json 
python3 DDM_test.py -p test -c config/DDM_test_b8_nomotion_oe.json
python3 DDM_test.py -p test -c config/DDM_test_b8_motion_oe.json


# things to investigate: 
    in model.py
        test - this is where actual inferrence is defined (is called ). which is called from train.py 
    in diffusion.py - heavily modify 
        p_sample_loop
        p_losses
    in diffusion.py - never used functions (because we never use DDPM like we should have )
        set_new_noise_schedule
        q_mean_variance
        p_mean_variance

