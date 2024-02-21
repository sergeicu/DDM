# link files - relative symlink


# currently done 
    # python preprocessing/acdc_symlink2.py data_ES_motion_noOE mat_oe_motion_noOE
    # python preprocessing/acdc_symlink2.py data_ES_motion_OE mat_oe_motion_OE
    # python preprocessing/acdc_symlink2.py data_ES_nomotion_OE mat_oe_nomotion_OE
    # python preprocessing/acdc_symlink2.py data_ES_nomotion_noOE mat_oe_nomotion_noOE
    # python preprocessing/acdc_symlink2.py data_ED_ES mat



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

def create_relative_symlink(target, link_name):
    # Get the absolute path of the target and link location
    abs_target = os.path.abspath(target)
    abs_link_name = os.path.abspath(link_name)
    
    # Compute relative path to the target from the link location
    rel_path = os.path.relpath(abs_target, os.path.dirname(abs_link_name))
    
    # Create symbolic link using relative path
    os.symlink(rel_path, link_name)

for folder in ['training', 'testing']:
    os.chdir(d+folder)
    patients = sorted(glob.glob('patient*/'))
    for c, patient in enumerate(patients): 
        
        savematdir = patient + savedir_source + "/"
        assert os.path.exists(savematdir)
        matfiles = glob.glob(savematdir + "*.mat")
        if len(matfiles)==2: 
            matfiles = [i for i in matfiles if '_og.mat' not in i]
        assert len(matfiles) ==1, embed()
        savename_iso = matfiles[0]
            
        os.makedirs(savedir_r + folder.replace("ing", "/"), exist_ok=True)
        newfilename = savedir_r + folder.replace("ing", "/") + os.path.basename(savename_iso)
        if os.path.islink(newfilename): 
            os.unlink(newfilename)
        if not os.path.exists(newfilename) and not os.path.islink(newfilename):
            #os.symlink(d+folder+"/"+savename_iso, newfilename)
            create_relative_symlink(d+folder+"/"+savename_iso, newfilename)





            
        
            
        
        

        