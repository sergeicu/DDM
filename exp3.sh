# train DPSduo and DPS_morph


##############
# Activate 
##############
cd ~/w/code/diffusion/experiments/s20240209_oecorr/DDM
conda activate diffusers 
source ../DiffuseMorph/venv/bin/activate 


cd ~/w/code/diffusion/experiments/s20240209_oecorr/DDM
conda activate diffusers 
source venv_e2/bin/activate 


##############
# Train 
##############

# train dpsduo 1 - same noise - all subjects - single batch 
git checkout dps_duo 
python3 DDM_train.py -p train -c config/DPSduo_train_v1.json --nowandb


# train dpsduo 1 - different noise... - all subjects - single batch 
git checkout dps_duo 
python3 DDM_train.py -p train -c config/DPSduo_train_v1.json --nowandb




