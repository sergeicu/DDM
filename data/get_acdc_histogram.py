"""Get histogram of acdc - required for good training"""

from scipy.io import loadmat

root='/fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/diffusion/experiments/s20240209_oecorr/DDM/data/acdc/data_ED_ES/train/'

import glob
import numpy as np 
import nibabel as nb 

files = glob.glob(root+"*.mat")
assert files
ims = []

fineSize = [128, 128, 32]

def pad(dataA, dataB, fineSize):

    nh, nw, nd = dataA.shape
    desired_h, desired_w, desired_d = fineSize  # fineSize now includes depth


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
        dataB = np.pad(dataB, ((pad_top, pad_bottom), (pad_left, pad_right), (pad_front, pad_back)), 'constant', constant_values=0)
    
    # Recalculate new heights and widths after padding
    nh, nw, nd = dataA.shape
    
    # Calculate start points for cropping (automatically handles indivisibility by using integer division)
    sh = max(0, (nh - desired_h) // 2)
    sw = max(0, (nw - desired_w) // 2)
    sd = max(0, (nd - desired_d) // 2)

    # Crop to the desired size for height, width, and depth
    dataA = dataA[sh:sh + desired_h, sw:sw + desired_w, sd:sd + desired_d]
    dataB = dataB[sh:sh + desired_h, sw:sw + desired_w, sd:sd + desired_d]
    return dataA, dataB    

for i,f in enumerate(files):
    
    ff = loadmat(f)
    im1o=ff['image_ES']
    im2o=ff['image_ED']


    im1,im2=pad(im1o, im2o, fineSize)
    
    
    ims.append(im1)
    ims.append(im2)
    
    print(i)    
    if i >50: 
    # if i >5: 
        break 
    
    
from IPython import embed; embed()
ims = np.moveaxis(np.array(ims), 0,-1)
ims_ave = np.mean(ims, axis=-1)

nb.save(nb.Nifti1Image(ims,affine=np.eye(4)),"data/acdc_104_volumes.nii.gz" )


nb.save(nb.Nifti1Image(ims_ave,affine=np.eye(4)),"data/acdc_104_volumes_averaged.nii.gz" )


nb.save(nb.Nifti1Image(im1o,affine=np.eye(4)),"data/acdc_1_volume.nii.gz" )



import matplotlib.pyplot as plt




# Flatten the image data to get a 1D array of all pixel values
flat_image_data = ims.flatten()

# Plotting the histogram of the flattened image data
plt.hist(flat_image_data, bins=256, color='skyblue', alpha=0.7)
plt.title('Pixel Intensity Distribution')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
# plt.show()

plt.savefig('data/histogram_all.png')




clipped_image_data = np.clip(ims, 0, 500)


nb.save(nb.Nifti1Image(clipped_image_data,affine=np.eye(4)),"data/acdc_104_volumes_clipped.nii.gz" )



