from torch.utils.data import Dataset
import data.util as Util
import os
import numpy as np
import scipy.io as sio
import time


import numpy as np 

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



def deformdata(deformed_image,deformed_label,num_control_points=(7,7,7), max_displacement=(7.5,7.5,7.5), locked_borders=2):
    
    labels = np.unique(deformed_label)
    
    deformed_image_t = torch.Tensor(deformed_image).unsqueeze(0)
    deformed_label_t = torch.Tensor(deformed_label).unsqueeze(0)
    
    # https://torchio.readthedocs.io/transforms/augmentation.html
    deformation_transform = ConsistentRandomElasticDeformation(num_control_points=num_control_points, max_displacement=max_displacement,locked_borders=locked_borders)
    deformed_image_t, deformed_label_t, deformation_field = apply_deformation_to_tensors(deformed_image_t, deformed_label_t, deformation_transform)
    
    deformed_image = deformed_image_t.numpy()[0,...]
    deformed_label = deformed_label_t.numpy()[0,...]
    deformation_field = deformation_field.numpy()[0,...]
    
    
    newlabels = np.unique(deformed_label)
    assert np.all(labels==newlabels)
    

    # newimo = nb.Nifti1Image(deformation_field, affine=imo.affine, header=imo.header)
    # savename = newdirname + 'field_' +k_load + "_iso_" + "_tr"+ "_full"+".nii.gz"
    # nb.save(newimo, savename)                    

    return deformed_image,deformed_label

class ACDCDataset(Dataset):
    def __init__(self, dataroot, split='train'):
        self.split = split
        self.imageNum = []
        self.dataroot = dataroot

        datapath = os.path.join(dataroot, split)
        dataFiles = sorted(os.listdir(datapath))
        for isub, dataName in enumerate(dataFiles):
            self.imageNum.append(os.path.join(datapath, dataName))

        self.data_len = len(self.imageNum)
        self.fineSize = [128, 128, 32]

    def __len__(self):
        return self.data_len

    def random_deformation(self, images):
        pass
        
    
    def __getitem__(self, index):
        dataPath = self.imageNum[index]
        data_ = sio.loadmat(dataPath)
        dataA = data_['image_ED']
        label_dataA = data_['label_ED']
        
        start_time = time.time()
        dataB,label_dataB = deformdata(dataA, label_dataA)
        end_time = time.time()
        # print(f"Function1 took {end_time - start_time} seconds.")

        
        assert dataA.shape == dataB.shape 
        assert label_dataA.shape == label_dataB.shape 
        
        # save deformation 
        
        # if self.split == 'test':
        #     dataName = dataPath.split('/')[-1]
        #     data_ = sio.loadmat(os.path.join(self.dataroot, 'data_ED2ES', dataName))
        #     dataW = data_['image']
        #     nsample = dataW.shape[-1]
        # else:
        nsample = 7 # the average number of samples used in inferrence only 

        dataA -= dataA.min()
        dataA /= dataA.std()
        dataA -= dataA.min()
        dataA /= dataA.max()

        dataB -= dataB.min()
        dataB /= dataB.std()
        dataB -= dataB.min()
        dataB /= dataB.max()

        nh, nw, nd = dataA.shape
        desired_h, desired_w, desired_d = self.fineSize  # fineSize now includes depth


        pad_h = max(0, desired_h - nh)
        pad_w = max(0, desired_w - nw)
        pad_d = max(0, desired_d - nd)


        # Apply padding evenly on both sides, with any extra padding added to the bottom/right
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_front = pad_d // 2
        pad_back = pad_d - pad_front        
        
        # Pad the images and labels if necessary
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            dataA = np.pad(dataA, ((pad_top, pad_bottom), (pad_left, pad_right), (pad_front, pad_back)), 'constant', constant_values=0)
            label_dataA = np.pad(label_dataA, ((pad_top, pad_bottom), (pad_left, pad_right), (pad_front, pad_back)), 'constant', constant_values=0)
            dataB = np.pad(dataB, ((pad_top, pad_bottom), (pad_left, pad_right), (pad_front, pad_back)), 'constant', constant_values=0)
            label_dataB = np.pad(label_dataB, ((pad_top, pad_bottom), (pad_left, pad_right), (pad_front, pad_back)), 'constant', constant_values=0)
        
        # Recalculate new heights and widths after padding
        nh, nw, nd = dataA.shape
        
        # Calculate start points for cropping (automatically handles indivisibility by using integer division)
        sh = max(0, (nh - desired_h) // 2)
        sw = max(0, (nw - desired_w) // 2)
        sd = max(0, (nd - desired_d) // 2)

        # Crop to the desired size for height, width, and depth
        dataA = dataA[sh:sh + desired_h, sw:sw + desired_w, sd:sd + desired_d]
        label_dataA = label_dataA[sh:sh + desired_h, sw:sw + desired_w, sd:sd + desired_d]
        dataB = dataB[sh:sh + desired_h, sw:sw + desired_w, sd:sd + desired_d]
        label_dataB = label_dataB[sh:sh + desired_h, sw:sw + desired_w, sd:sd + desired_d]        
        

        [data, label] = Util.transform_augment([dataA, dataB], split=self.split, min_max=(-1, 1))

        return {'S': data, 'T': label, 'SL': label_dataA, 'TL': label_dataB, 'nS':nsample, 'P':dataPath, 'Index': index}
