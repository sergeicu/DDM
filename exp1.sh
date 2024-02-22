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


# start on e2 
e2 
srun -A crl -p crl-gpu -t 14:00:00 --qos=crl --gres=gpu:NVIDIA_A40:1 -n 8 --mem 128GB --pty /bin/bash 
srun -A crl -p crl-gpu -t 14:00:00 --qos=crl --gres=gpu:NVIDIA_A40:1  --pty /bin/bash 
cd ~/w/code/diffusion/experiments/s20240209_oecorr/DDM
conda activate diffusers 
source venv/bin/activate 
