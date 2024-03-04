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
python3 DDM_train.py -p train -c config/DPSduo_train_v1.json  # comic_brook_40  # experiments/DPSduo_train_240302_031722


# train dpsduo 1 - different noise... - all subjects - single batch 
git checkout dps_duo_2losses
python3 DDM_train.py -p train -c config/DPSduo_train_2losses.json --nowandb

# train dps_morph -> 


##############
# Test 
##############

# test dpsduo 1 - same noise - all subjects - single batch 
python3 DDM_test.py -p test -c config/DPSduo_test_v1.json  # weights are here: experiments/DPSduo_train_240302_031722
    
    # dps without repetitions 
    experiments/DPSduo_test_240302_155956/..

    # dps repeated 5 times for every step 
    experiments/DPSduo_test_240302_160533/results

    # no dps
    experiments/DPSduo_test_240302_164015/results

    # see other results in: 
    log_s20240302_v2.sh


# dpsduo1 - v2 - with updated weights after 2000 epochs 
python3 DDM_test.py -p test -c config/DPSduo_test_v2.json 



# what was the error? it was something to do with... noise schedule?? 
