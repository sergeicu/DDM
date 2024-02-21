# link files 


# currently done 
    # python preprocessing/acdc_symlink.py data_ES_motion_noOE mat_oe_motion_noOE
    # python preprocessing/acdc_symlink.py data_ES_motion_OE mat_oe_motion_OE
    # python preprocessing/acdc_symlink.py data_ES_nomotion_OE mat_oe_nomotion_OE
    # python preprocessing/acdc_symlink.py data_ES_nomotion_noOE mat_oe_nomotion_noOE
    # python preprocessing/acdc_symlink.py data_ED_ES mat



import glob 
import os 
import sys 
from IPython import embed

savedir_target = sys.argv[1] # 'data_ED_ES' -> 
savedir_source = sys.argv[2] # 'mat' -> /fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/diffusion/experiments/s20240209_oecorr/DDM/data/acdc/
assert len(sys.argv) ==3 


d='/fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/diffusion/experiments/s20240209_oecorr/DDM/data/acdc/'
savedir_r = d + savedir_target + "/" #'data_ED_ES/'
os.makedirs(savedir_r, exist_ok=True)

for folder in ['training', 'testing']:
    os.chdir(d+folder)
    patients = sorted(glob.glob('patient*/'))
    for c, patient in enumerate(patients): 
        
        savematdir = patient + savedir_source + "/"
        assert os.path.exists(savematdir)
        matfiles = glob.glob(savematdir + "*.mat")
        assert len(matfiles) ==1, embed()
        savename_iso = matfiles[0]
            
        os.makedirs(savedir_r + folder.replace("ing", "/"), exist_ok=True)
        newfilename = savedir_r + folder.replace("ing", "/") + os.path.basename(savename_iso)
        if os.path.islink(newfilename): 
            os.unlink(newfilename)
        if not os.path.exists(newfilename) and not os.path.islink(newfilename):
            os.symlink(d+folder+"/"+savename_iso, newfilename)





            
        
            
        
        

        