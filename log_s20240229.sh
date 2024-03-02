find out where is the train DDPM_only model for DDM 
    - check config files -> configs/DDM_train_b8_ddpm_only.json
        trained models are: 
        DDM_train_240222_012445/checkpoint_b8_data_ED_ES/ -> I50000 -> i suspect that this is the ddpm testing...
        DDM_train_240222_011330/checkpoint_b8_data_ED_ES/ -> only I500 ... not sure -> could be synthmorph on the fly... that broke...
        DDM_train_240222_003248/checkpoint_b8_data_ED_ES/ -> only I200 ... not sure -> could be synthmorph on the fly... that broke...
        DDM_train_240221_193618/checkpoint_b8_data_ES_motion_noOE_v2 -> 34000 epochs -> this is ED-ES model trained to the end 
        DDM_train_240221_130315/ -> only 4420 -> this and the next 3 models are: ED-ES model + ED+synthmorph + ED+odd-even? 
        DDM_train_240221_124951/ -> only 2720 
        DDM_train_240221_124950/ -> only 2720 (repeated)

        # check sizes 
        for fldr in */; 
        do 
            echo $fldr
            ls -lat $fldr/checkpoint_*/ | head -n 3
        done 



        - experiments/DDM_test_240223_161223/results/ -> this is me doing inferrence on model that was trained on: 
            .oe_motion_noOE_mov.nii.gz... -> motion model but no odd/even (deformed synthetically)
        - experiments/DDM_test_240223_161148/results/ -> this is me doing inferrence on model that was trained on: 
            patient101_iso_moved.nii.gz -> original model (ED/ES)
        - experiments/DDM_test_240223_160501/results/ -> this is a totally empty folder -> failed ddpm_only test 

    - 

find out which branch i used to train it (lol) - ddpm_only or single thing... ->> DDPM_only 
    - it works on single image - im feeding single image and I get the predicted noise output ... 
        - now i need to add a correct schduler to it... 


# which files are we changing now: 
    I need to change: 
        model.py -> def test(self, continous=False):
            -> calls to inferrence 
            -> provides code, deformed image and field (from DDPM + VM model)

            -> we should change it to run: 
                -> 
        diffusion.py -> 



SUMMARY OF EXPERIMENTS USING 3D DDPM_only: 

    - the network produces very noisy images after 2000 iterations 
        - it is very likely that the amount of noise added to these images with 2000 noise steps at current beta is too much 
        - we need to reduce the amount of noise added... 
            - produce a set of images to regularize the amount of noise being added...
            - we should have 1000 steps... 
            - and we should have noise destructed slowly to pure noise at 1000 already.. (or close to)


        - to do this - lets calculate SNR as we add noise and plot it as a linear plot - 
            - use this code example - 
            https://chat.openai.com/share/816fbc7b-b00e-41f7-94ba-959ca331e4cd

            

    - adding zeros or ones only when training the network - completely destroys the produced image 
        normal - 2000 iters - saving finer after 500 
            experiments/DDM_test_240229_183309/results/patient101_iso_t1000_denoised.nii.gz

        ones - 2000 iters - saving finer after 500 
            experiments/DDM_test_240229_183352/results/patient101_iso_t1500_denoised.nii.gz

        zeros - 2000 iters - saving finer after 500 
            experiments/DDM_test_240229_183420/results/patient101_iso_t1000_denoised.nii.gz		