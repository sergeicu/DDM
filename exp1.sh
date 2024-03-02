cd ~/w/code/diffusion/experiments/s20240209_oecorr/DDM
conda activate diffusers 
source ../DiffuseMorph/venv/bin/activate 

# train 
python3 DDM_train.py -p train -c config/DDM_train_b8.json
python3 DDM_train.py -p train -c config/DDM_train_b8_motion_oe.json
python3 DDM_train.py -p train -c config/DDM_train_b8_nomotion_oe.json
python3 DDM_train.py -p train -c config/DDM_train_b8_motion_nooe.json


# test - round 1 
python3 DDM_test.py -p test -c config/DDM_test_b8.json # experiments/DDM_test_240221_114339/results
python3 DDM_test.py -p test -c config/DDM_test_b8_motion_oe.json # experiments/DDM_test_240221_133806/results
python3 DDM_test.py -p test -c config/DDM_test_b8_nomotion_oe.json # experiments/DDM_test_240221_133834/results
python3 DDM_test.py -p test -c config/DDM_test_b8_motion_nooe.json # experiments/DDM_test_240221_134055/results


# test - round 2 (later weights ) (need to check results and compare)
python3 DDM_test.py -p test -c config/DDM_test_b8_motion_oe.json # experiments/DDM_test_240221_151953/results
python3 DDM_test.py -p test -c config/DDM_test_b8_nomotion_oe.json # experiments/DDM_test_240221_151956/results
python3 DDM_test.py -p test -c config/DDM_test_b8_motion_nooe.json # experiments/DDM_test_240221_152036/results


# train v2 - for very long time 
git checkout main
python3 DDM_train.py -p train -c config/DDM_train_b8.json
python3 DDM_train.py -p train -c config/DDM_train_b8_motion_nooe.json

# train v2 - with infinite deformations (not the same over each epoch)
git checkout onthefly_deform
python3 DDM_train.py -p train -c config/DDM_train_b8_motion_nooe.json


# train ddpm only 
git checkout ddpm_only 
python3 DDM_train.py -p train -c config/DDM_train_b8_ddpm_only.json


# test ddpm only 
git checkout ddpm_only 
python3 DDM_test.py -p test -c config/DDM_test_b8_ddpm_only.json 


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


# train with UPDATED noise schedule for 
    1A. images with condition

        python3 DDM_train.py -p train -c config/DDM_train_ddpm_wcond_1s.json # astral-cloud 
        python3 DDM_train.py -p train -c config/DDM_train_ddpm_wcond.json # likely-salad 
    1B. images without condition: 
        python3 DDM_train.py -p train -c config/DDM_train_ddpm_nocond.json # gallant sponge 
        python3 DDM_train.py -p train -c config/DDM_train_ddpm_nocond_1s.json
    2A. ES-ED images without condition... 
        


# start on e2 
e2 
srun -A crl -p crl-gpu -t 14:00:00 --qos=crl --gres=gpu:NVIDIA_A40:1 -n 8 --mem 128GB --pty /bin/bash 
srun -A crl -p crl-gpu -t 14:00:00 --qos=crl --gres=gpu:NVIDIA_A40:1  --pty /bin/bash 
cd ~/w/code/diffusion/experiments/s20240209_oecorr/DDM
conda activate diffusers 
source venv/bin/activate 
