cd ~/w/code/diffusion/experiments/s20240209_oecorr/DDM
conda activate diffusers 
source ../DiffuseMorph/venv/bin/activate 

# train 
python3 DDM_train.py -p train -c config/DDM_train_b8.json
python3 DDM_train.py -p train -c config/DDM_train_b8_motion_oe.json
python3 DDM_train.py -p train -c config/DDM_train_b8_nomotion_oe.json
python3 DDM_train.py -p train -c config/DDM_train_b8_motion_nooe.json


# test 
python3 DDM_test.py -p test -c config/DDM_test_b8.json # experiments/DDM_test_240221_114339/results
python3 DDM_test.py -p test -c config/DDM_test_b8_motion_oe.json # experiments/DDM_test_240221_133806/results
python3 DDM_test.py -p test -c config/DDM_test_b8_nomotion_oe.json # experiments/DDM_test_240221_133834/results
python3 DDM_test.py -p test -c config/DDM_test_b8_motion_nooe.json # experiments/DDM_test_240221_134055/results


# start on e2 
e2 
srun -A crl -p crl-gpu -t 14:00:00 --qos=crl --gres=gpu:NVIDIA_A40:1 -n 8 --mem 128GB --pty /bin/bash 
srun -A crl -p crl-gpu -t 14:00:00 --qos=crl --gres=gpu:NVIDIA_A40:1  --pty /bin/bash 
cd ~/w/code/diffusion/experiments/s20240209_oecorr/DDM
conda activate diffusers 
source venv/bin/activate 
