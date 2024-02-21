import shutil 
import os 
import glob

d='/fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/diffusion/experiments/s20240209_oecorr/DDM/data/acdc/'

for folder in ['training', 'testing']:
    os.chdir(d+folder)
    patients = sorted(glob.glob('patient*/'))
    for c, patient in enumerate(patients): 
        print(f"{patient}")
        
        mats = glob.glob(patient + "mat_*")
        for m in mats: 
            shutil.rmtree(m)
        