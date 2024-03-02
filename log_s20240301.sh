biggest changes are in: 
    - model.py and diffusion.py 
        - if i would need to revert to original model this is where a lot of changes are made 
    - some changes are made in train.py and networks.py (removed deformation module for example)




task right now: 
    - build an algorithm that outputs image at different noise levels of T for me to view... 

    


git checkout f9b39164556386ad5b794c0469917fff50b10623
python3 DDM_test.py -p test -c config/DDM_test_b8_ddpm_only.json 

    experiments/DDM_test_240301_192316/results/ -> do not add noise at idx = 0 (different way)


    experiments/DDM_test_240301_193033/results/ -> - > extra_denoise = True
    experiments/DDM_test_240301_193626/results/ -> saves every iteration as well (at the end)

    experiments/DDM_test_240301_193149/results/ - > clip_denoised = false ->>>> WORKS !!! 





1. Where is the forward model defined in DPS? 

2. Where is the forward model called in DPS? 
    - p_sample_loop 





# lets test these properly 
    WITH CONDITION 
        python3 DDM_train.py -p train -c config/DDM_train_ddpm_wcond_1s.json # astral-cloud  -> experiments/DDM_train_240301_172419/checkpoint_b8_data_ED_ES/
        python3 DDM_train.py -p train -c config/DDM_train_ddpm_wcond.json # likely-salad ! -> experiments/DDM_train_240301_172720/checkpoint_b8_data_ED_ES/
    WITHOUT CONDITION
        python3 DDM_train.py -p train -c config/DDM_train_ddpm_nocond.json # gallant sponge ! -> experiments/DDM_train_240301_180405/checkpoint_b8_data_ED_ES/
        python3 DDM_train.py -p train -c config/DDM_train_ddpm_nocond_1s.json # stellar-dragon -> experiments/DDM_train_240301_190357/checkpoint_b8_data_ED_ES/


    # 1. python3 DDM_train.py -p train -c config/DDM_train_ddpm_nocond.json # gallant sponge ! -> experiments/DDM_train_240301_180405/checkpoint_b8_data_ED_ES/

    DDM_test_ddpm_nocond.json

