# still to do - produced segmentations are not actually segmentations... but the same disturbed images. 


import argparse
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Create the parser
parser = argparse.ArgumentParser(description="Process the boolean flags.")

# Add arguments with default True
parser.add_argument("--motion", type=str2bool, default=True, help="Set motion (default: True)")
parser.add_argument("--oddeven", type=str2bool, default=True, help="Set oe (default: True)")
args = parser.parse_args()
motion = args.motion
oe = args.oddeven


if motion:
    prefix = "_motion"
else:
    prefix="_nomotion"
if oe: 
    prefix = prefix + "_OE"
else:
    prefix = prefix + "_noOE"    
    
    
import glob 
import os 
import shutil

import scipy.io as sio
import numpy as np 
import nibabel as nb 

import copy




import torch
from torchio import RandomElasticDeformation,ElasticDeformation
import torchio as tio

from torchio.data.subject import Subject
class ConsistentRandomElasticDeformation:
    def __init__(self, num_control_points=7, max_displacement=7.5, locked_borders=2, image_interpolation='linear', label_interpolation='nearest'):
        # Initialize parameters
        self.num_control_points = num_control_points
        self.max_displacement = max_displacement
        self.locked_borders = locked_borders
        self.image_interpolation = image_interpolation
        self.label_interpolation = label_interpolation

        # Generate a consistent random deformation field upon initialization
        self.deformation_field = self._generate_deformation_field()

    def _generate_deformation_field(self):
        # Use RandomElasticDeformation to generate a deformation field.
        # This step requires extracting the relevant logic from RandomElasticDeformation
        # or refactoring its implementation to expose such functionality.
        random_elastic_deformation = RandomElasticDeformation(
            num_control_points=self.num_control_points,
            max_displacement=self.max_displacement,
            locked_borders=self.locked_borders,
            image_interpolation=self.image_interpolation,
            label_interpolation=self.label_interpolation,
        )
        deformation_field = random_elastic_deformation.get_params(
            num_control_points=self.num_control_points,
            max_displacement=self.max_displacement,
            num_locked_borders=self.locked_borders
        )
        return deformation_field

    def apply_deformation(self, subject):
        # Use the generated deformation field to deform the given image.
        # This might require adapting or directly calling ElasticDeformation with the deformation field.
        elastic_deformation = ElasticDeformation(
            control_points=self.deformation_field,
            max_displacement=self.max_displacement,
            image_interpolation=self.image_interpolation,
            label_interpolation=self.label_interpolation,
        )
        transformed_subject = elastic_deformation.apply_transform(subject)
        return transformed_subject




def apply_deformation_to_tensors(image_tensor, label_tensor, deformation_transform):
    # Assuming image_tensor and label_tensor are 3D torch tensors [C, H, W, D]

    # Wrap tensors in a format expected by the transformation classes (e.g., TorchIO Subject)
    image = tio.ScalarImage(tensor=image_tensor)  # Use ScalarImage for the image
    label = tio.LabelMap(tensor=label_tensor)     # Use LabelMap for the segmentation label

    subject = tio.Subject(image=image, label=label)

    # Apply the deformation
    transformed_subject = deformation_transform.apply_deformation(subject)

    # Extract the deformed image and label tensors
    deformed_image_tensor = transformed_subject['image'].tensor
    deformed_label_tensor = transformed_subject['label'].tensor

    # Retrieve the deformation field (assuming it's stored and accessible as a tensor)
    deformation_field_tensor = torch.tensor(deformation_transform.deformation_field)

    return deformed_image_tensor, deformed_label_tensor, deformation_field_tensor


d='/fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/diffusion/experiments/s20240209_oecorr/DDM/data/acdc/'



k_load='ES'
k1='ES'
k2='ED'
savedir_r = d + "data_"+k_load + prefix+"/"
if os.path.exists(savedir_r):
    shutil.rmtree(savedir_r)        
os.makedirs(savedir_r, exist_ok=True)


from IPython import embed; 




for folder in ['training', 'testing']:
    os.chdir(d+folder)
    patients = sorted(glob.glob('patient*/'))
    for c, patient in enumerate(patients): 
        print(f"{patient}")
        
        savematdir = patient + "mat/"
        savename_iso = savematdir + patient.replace("/", "_iso.mat")
        assert os.path.exists(savename_iso), embed()
        
        # load nifti ref 
        dirname = os.path.dirname(savename_iso) + "/"
        imo = nb.load(dirname + "label_"+k_load+"_iso.nii.gz")
        
        newdirname = dirname.replace("/mat/", "/mat_oe"+prefix+"/")
        if os.path.exists(newdirname):
            shutil.rmtree(newdirname)
        os.makedirs(newdirname,exist_ok=True)
        
        # load file 
        mdict = sio.loadmat(savename_iso)
        x,y,z = mdict['image_' +k_load].shape
        
        # split odd-even 
        
        
        # create array of zeros 
        mdict2 = {}
        mdict2['image_' +k1] = np.zeros((x,y,z))
        mdict2['label_' +k1] = np.zeros((x,y,z))
        mdict2['image_' +k2] = np.zeros((x,y,z))
        mdict2['label_' +k2] = np.zeros((x,y,z))

        # fill the even ones 
        if oe:
            mdict2['image_' +k2][:,:,0::2] = copy.deepcopy(mdict['image_' +k_load][:,:,0::2])
            mdict2['label_' +k2][:,:,0::2] = copy.deepcopy(mdict['label_' +k_load][:,:,0::2])
        else: 
            mdict2['image_' +k2] = copy.deepcopy(mdict['image_' +k_load])
            mdict2['label_' +k2] = copy.deepcopy(mdict['label_' +k_load])
                    
        # create a copy of the array and multiply by some deformation field 
        deformed_image = copy.deepcopy(mdict['image_' +k_load])
        deformed_label = copy.deepcopy(mdict['label_' +k_load])
        labels = np.unique(deformed_label)
        
        # turn into tensor 
        deformed_image_t = torch.Tensor(deformed_image).unsqueeze(0)
        deformed_label_t = torch.Tensor(deformed_label).unsqueeze(0)
        
        # generate random deformation field... 
        if motion: 
            # https://torchio.readthedocs.io/transforms/augmentation.html
            deformation_transform = ConsistentRandomElasticDeformation(num_control_points=(7,7,7), max_displacement=(7.5,7.5,7.5), locked_borders=2)
            deformed_image_t, deformed_label_t, deformation_field = apply_deformation_to_tensors(deformed_image_t, deformed_label_t, deformation_transform)
        else: 
            deformation_field = torch.zeros_like(deformed_image_t)
            
        deformed_image = deformed_image_t.numpy()[0,...]
        deformed_label = deformed_label_t.numpy()[0,...]
        deformation_field = deformation_field.numpy()[0,...]
        
        newlabels = np.unique(deformed_label)
        assert np.all(labels==newlabels)
        
        # copy to slice 
        if oe:
            mdict2['image_' +k1][:,:,1::2] = deformed_image[:,:,1::2] # odd 
            mdict2['label_' +k1][:,:,1::2] = deformed_label[:,:,1::2] # odd 
        else: 
            mdict2['image_' +k1] = deformed_image # odd 
            mdict2['label_' +k1] = deformed_label # odd 
                    

                
        newimo = nb.Nifti1Image(deformed_image, affine=imo.affine, header=imo.header)
        savename = newdirname + 'image_' +k_load + "_iso_" + "full"+"_tr"+".nii.gz"
        nb.save(newimo, savename)    
        
        newimo = nb.Nifti1Image(mdict['image_' +k1], affine=imo.affine, header=imo.header)
        savename = newdirname + 'image_' +k_load + "_iso_" + "full"+".nii.gz"
        nb.save(newimo, savename)            

        newimo = nb.Nifti1Image(mdict2['image_' +k2], affine=imo.affine, header=imo.header)
        savename = newdirname + 'image_' +k_load + "_iso_" + "_even"+".nii.gz"
        nb.save(newimo, savename)    
        
        newimo = nb.Nifti1Image(mdict2['image_' +k1], affine=imo.affine, header=imo.header)
        savename = newdirname + 'image_' +k_load + "_iso_" + "odd"+"_tr"+".nii.gz"
        nb.save(newimo, savename)            

        odd_og = np.zeros_like(mdict['image_' +k_load])
        if oe:
            odd_og[:,:,1::2] = copy.deepcopy(mdict['image_' +k_load][:,:,1::2])
        else:
            odd_og = copy.deepcopy(mdict['image_' +k_load])
        newimo = nb.Nifti1Image(odd_og, affine=imo.affine, header=imo.header)
        savename = newdirname + 'image_' +k_load + "_iso_" + "_odd"+".nii.gz"
        nb.save(newimo, savename)            

        newimo = nb.Nifti1Image(deformation_field, affine=imo.affine, header=imo.header)
        savename = newdirname + 'field_' +k_load + "_iso_" + "_tr"+ "_full"+".nii.gz"
        nb.save(newimo, savename)                    
        






        newimo = nb.Nifti1Image(deformed_label, affine=imo.affine, header=imo.header)
        savename = newdirname + 'label_' +k_load + "_iso_" + "full"+"_tr"+".nii.gz"
        nb.save(newimo, savename)    
        
        newimo = nb.Nifti1Image(mdict['label_' +k1], affine=imo.affine, header=imo.header)
        savename = newdirname + 'label_' +k_load + "_iso_" + "full"+".nii.gz"
        nb.save(newimo, savename)            

        newimo = nb.Nifti1Image(mdict2['label_' +k2], affine=imo.affine, header=imo.header)
        savename = newdirname + 'label_' +k_load + "_iso_" + "_even"+".nii.gz"
        nb.save(newimo, savename)    
        
        newimo = nb.Nifti1Image(mdict2['label_' +k1], affine=imo.affine, header=imo.header)
        savename = newdirname + 'label_' +k_load + "_iso_" + "odd"+"_tr"+".nii.gz"
        nb.save(newimo, savename)            

        odd_og = np.zeros_like(mdict['label_' +k_load])
        if oe:
            odd_og[:,:,1::2] = copy.deepcopy(mdict['label_' +k_load][:,:,1::2])
        else:
            odd_og = copy.deepcopy(mdict['label_' +k_load])
        newimo = nb.Nifti1Image(odd_og, affine=imo.affine, header=imo.header)
        savename = newdirname + 'label_' +k_load + "_iso_" + "_odd"+".nii.gz"
        nb.save(newimo, savename)            
    
    
        # save file 
        newmatname = newdirname + os.path.basename(savename_iso).replace(".mat", "_oe"+prefix+".mat")
        sio.savemat(newmatname, mdict2)    
                    
        os.makedirs(savedir_r + folder.replace("ing", "/"), exist_ok=True)
        newfilename = savedir_r + folder.replace("ing", "/") + os.path.basename(savename_iso)
        if os.path.islink(newfilename): 
            os.unlink(newfilename)
        if not os.path.exists(newfilename) and not os.path.islink(newfilename):
            os.symlink(d+folder+"/"+savename_iso, newfilename)
            
            




            
        
            
        
        

        