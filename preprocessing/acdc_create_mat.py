# todo: 
# do not multiply the affine - it should stay the same - not increase in size... 
# need to downsample by 2x in x,y plane to reach 128x128 pixels...


import glob 
import os 
import copy
import scipy.io as sio
import nibabel as nb 
import numpy as np 
from scipy.ndimage import zoom
import shutil


d_partial='/fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/diffusion/experiments/s20240209_oecorr/DDM/data/acdc/'


for fold in ["training/", "testing/"]:
    d=d_partial+fold
    os.chdir(d)

    patients = sorted(glob.glob('patient*/'))
    for c, patient in enumerate(patients): 
        
        savematdir = patient + "mat/"
        if os.path.exists(savematdir):
            shutil.rmtree(savematdir)
        os.makedirs(savematdir, exist_ok=True)
        
        # remove previously created files
        files_to_remove = glob.glob(patient + "*.mat")
        for f in files_to_remove:
            os.remove(f)
        
        
        
        print(patient)
        
        # grab files 
        segfiles = sorted(glob.glob(patient + "*frame*gt.nii.gz")) 
        assert len(segfiles)==2
        files=[i.replace("_gt", "") for i in segfiles]
        assert len(files)==2
        
        
        # read niftis and save into .mat 
        mdict = {}    
        imo = nb.load(files[0])
        mdict['label_ED'] = nb.load(segfiles[0]).get_fdata()
        mdict['label_ES'] = nb.load(segfiles[1]).get_fdata()
        mdict['image_ED'] = nb.load(files[0]).get_fdata()
        mdict['image_ES'] = nb.load(files[1]).get_fdata()   
        assert  mdict['label_ED'].shape == mdict['label_ES'].shape == mdict['image_ED'].shape == mdict['image_ES'].shape
        
        # interpolate the data
        x,y,z = imo.shape 

        # verbose 
        # print(f"{x,y,z}")    
        # print(f"{imo.header['pixdim'][1:4]}")
        for k,v in mdict.items():        
            newimo = nb.Nifti1Image(mdict[k], affine=imo.affine, header=imo.header)
            savename = savematdir + k + "_og.nii.gz"
            nb.save(newimo, savename)    
        
        # ##############################
        # # interpolate 6x in z axis (i.e. 10 / 36 = 1.66mm, while the x/y is 1.46mm )
        # ##############################    
        # # linear / bspline by 3x 
        # target_shape = (x, y, z*6)
        # zoom_factors = np.array(target_shape) / np.array((x, y, z))
        # mdict2_6x = {}  
        # for k,v in mdict.items():        
        #     mdict2_6x[k] = zoom(v, zoom_factors, order=1)
            
        #     # deal with segmentations 
        #     if 'label' in k:
        #         segments = np.unique(v)
        #         mdict2_6x[k] = np.round(mdict2_6x[k],0)
        #         mdict2_6x[k][mdict2_6x[k]<0] = 0 
        #         mdict2_6x[k][mdict2_6x[k]>segments[-1]] = segments[-1]
        #         segments_new = np.abs(np.unique(mdict2_6x[k]))
        #         assert np.all(segments==segments_new)
                
                
        #     # build new affine and save the image 
        #     new_affine = np.copy(imo.affine)
        #     #np.fill_diagonal(new_affine, np.append(zoom_factors, [1]))         # update affine to reflect new size
        #     new_header = copy.deepcopy(imo.header)
        #     new_header['pixdim'][3] = new_header['pixdim'][3]/6 # update 3rd pixel dimension to be isotropic
        #     newimo = nb.Nifti1Image(mdict2_6x[k], affine=new_affine, header=new_header)
        #     savename = savematdir + k + "_div_z_by_6.nii.gz"
        #     nb.save(newimo, savename)            
            
        # ##############################
        # # interpolate to isotropic in z (i.e. everything is 1.46mm...)
        # ##############################
        # # linear / bspline to isotropic 
        # voxel_spacing=imo.header['pixdim'][1:4]
        # assert voxel_spacing[0] == voxel_spacing[1]
        # target_voxel_size = voxel_spacing[0]
        # zoom_factors = voxel_spacing / target_voxel_size
        # mdict2_iso = {}  
        # order = 3
        # for k,v in mdict.items():
        #     mdict2_iso[k] = zoom(v, zoom_factors, order=order)  # order=3 for cubic interpolation
            
        #     if 'label' in k:
        #         segments = np.unique(v)
        #         mdict2_iso[k] = np.round(mdict2_iso[k],0)
        #         mdict2_iso[k][mdict2_iso[k]<0] = 0 
        #         mdict2_iso[k][mdict2_iso[k]>segments[-1]] = segments[-1]
        #         segments_new = np.abs(np.unique(mdict2_iso[k]))
        #         assert np.all(segments==segments_new)
            
        #     # build new affine and save the image 
        #     new_affine = np.copy(imo.affine)
        #     #np.fill_diagonal(new_affine, np.append(zoom_factors, [1]))         # update affine to reflect new size
        #     new_header = copy.deepcopy(imo.header)
        #     new_header['pixdim'][3] = new_header['pixdim'][2] # update 3rd pixel dimension to be isotropic
        #     newimo = nb.Nifti1Image(mdict2_iso[k], affine=new_affine, header=new_header)
        #     savename = savematdir + k + "_iso.nii.gz"
        #     nb.save(newimo, savename)



        ##############################
        # interpolate to isotropic in z (i.e. everything is 2.5 mm...)
        ##############################
        # linear / bspline to isotropic 
        voxel_spacing=imo.header['pixdim'][1:4]
        assert voxel_spacing[0] == voxel_spacing[1]
        target_voxel_size = 2.5 
        zoom_factors = voxel_spacing / target_voxel_size
        mdict2_iso = {}  
        for k,v in mdict.items():
            mdict2_iso[k] = zoom(v, zoom_factors, order=3)  # order=3 for cubic interpolation
            
            if 'label' in k:
                segments = np.unique(v)
                mdict2_iso[k] = np.round(mdict2_iso[k],0)
                mdict2_iso[k][mdict2_iso[k]<0] = 0 
                mdict2_iso[k][mdict2_iso[k]>segments[-1]] = segments[-1]
                segments_new = np.abs(np.unique(mdict2_iso[k]))
                assert np.all(segments==segments_new)
                
            # remove negatives 
            #mdict2_iso[k][mdict2_iso[k]<0] = 0
            
            # build new affine and save the image 
            new_affine = np.copy(imo.affine)
            #np.fill_diagonal(new_affine, np.append(zoom_factors, [1]))         # update affine to reflect new size
            new_header = copy.deepcopy(imo.header)
            # new_header['pixdim'][3] = new_header['pixdim'][2] # update 3rd pixel dimension to be isotropic
            new_header['pixdim'][1:4] = np.array([2.5, 2.5, 2.5])
            newimo = nb.Nifti1Image(mdict2_iso[k], affine=new_affine, header=new_header)
            savename = savematdir + k + "_iso.nii.gz"
            nb.save(newimo, savename)
            
            
            

        # ##############################
        # # pad with zeros in z by 3x - i.e. 
        # ##############################    

        # new_shape = x,y,z*3
        # mdict2_pad = {}  
        # for k,v in mdict.items():
        #     mdict2_pad[k] = np.zeros(new_shape)
        #     mdict2_pad[k][:, :, 0::3] = v        
            
            
        #     # build new affine and save the image 
        #     # new_affine = np.copy(imo.affine)
        #     # np.fill_diagonal(new_affine, np.append(new_shape, [1]))         # update affine to reflect new size
        #     # new_header = copy.deepcopy(imo.header)
        #     # newimo = nb.Nifti1Image(mdict2_pad[k], affine=new_affine, header=new_header)
        #     newimo = nb.Nifti1Image(mdict2_pad[k], affine=imo.affine, header=imo.header)
        #     savename = savematdir + k + "_pad.nii.gz"
        #     nb.save(newimo, savename)
        
        # ##############################
        # # save auxiliary info
        # ##############################          
        # # load auxiliary info 
        # mdict_aux = {}    
        # mdict_aux['header'] = nb.load(files[0]).header
        # mdict_aux['affine'] = nb.load(files[0]).affine
        
        
        
        ##############################
        # SAVE .mat files
        ##############################    
        # savefiles 
        savename_og = savematdir + patient.replace("/", "_og.mat")
        sio.savemat(savename_og, mdict)
        # savename_header = savematdir + patient.replace("/", "_header.mat")
        # sio.savemat(savename_header, mdict_aux)

        savename_iso = savematdir + patient.replace("/", "_iso.mat")
        sio.savemat(savename_iso, mdict2_iso)
        
        print(f"{mdict['label_ED'].shape}->{mdict2_iso['label_ED'].shape}")
        
        # savename_pad = savematdir + patient.replace("/", "_pad.mat")
        # sio.savemat(savename_pad, mdict2_pad)
        
        # savename_6x = savematdir + patient.replace("/", "_6x.mat")
        # sio.savemat(savename_6x, mdict2_6x)    
        
        
        
            
        # reference data - uncomment only for debug 
        # dataPath='../../acdc_example/ACDC_dataset/train/data_ED_ES/patient_train0.mat'
        # dataPath='../../acdc_example/ACDC_dataset/test/data_ED_ES/patient_test0.mat'
        # data_ = sio.loadmat(dataPath)
        # from IPython import embed; embed()
        # break 
        
        
        # i can resample to 1,1,1 or to 1.5 by 1.5 by 1.5? 
        
        
        
        
        
        
        
