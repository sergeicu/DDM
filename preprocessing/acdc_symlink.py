# todo: 
# do not multiply the affine - it should stay the same - not increase in size... 
# need to downsample by 2x in x,y plane to reach 128x128 pixels...


import glob 
import os 

d='/fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/diffusion/experiments/s20240209_oecorr/DDM/data/acdc/'

savedir_r = d + 'data_ED_ES/'
os.makedirs(savedir_r, exist_ok=True)



from IPython import embed; 


split = [76, 10, 10]

# for folder in ['training', 'testing']:
for folder in ['training', 'testing']:
    os.chdir(d+folder)
    patients = sorted(glob.glob('patient*/'))
    for c, patient in enumerate(patients): 
        
        savematdir = patient + "mat/"
        savename_iso = savematdir + patient.replace("/", "_iso.mat")
        
        assert os.path.exists(savename_iso), embed()
        
        # # create dir
        # if c <76: 
        #     savedir = d + "train/"
        # elif c >= 76 and c < 86:
        #     savedir = d + "val/"
        # elif c >= 86: 
        #     savedir = d + "test/"
            
        os.makedirs(savedir_r + folder.replace("ing", "/"), exist_ok=True)
        newfilename = savedir_r + folder.replace("ing", "/") + os.path.basename(savename_iso)
        if os.path.islink(newfilename): 
            os.unlink(newfilename)
        if not os.path.exists(newfilename) and not os.path.islink(newfilename):
            os.symlink(d+folder+"/"+savename_iso, newfilename)





            
        
            
        
        

        