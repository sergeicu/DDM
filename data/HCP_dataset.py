from torch.utils.data import Dataset
import data.util as Util
import os
import numpy as np
import scipy.io as sio
from scipy.ndimage import zoom
import nibabel as nib
import glob 


class HCPDataset(Dataset):
    def __init__(self, dataroot, split='train',resize_method='interpolate'):
        
        self.split = split
        self.imageNum = []
        self.dataroot = dataroot 

        datadirA = os.path.join(dataroot, 'F')
        datadirB = os.path.join(dataroot,  'M')
        dataFiles = sorted(glob.glob(datadirA + "/*.nii.gz"))
        for isub, dataNameF in enumerate(dataFiles):
            dataName = os.path.basename(dataNameF)
            imA = os.path.join(datadirA, dataName)
            imB = os.path.join(datadirB, dataName)
            self.imageNum.append([imA,imB])

        self.data_len = len(self.imageNum)
        self.fineSize = [128, 128, 32]
        self.resize_method = resize_method

    def __len__(self):
        return self.data_len
    
    def _pad_image(self,image,expected_shape=(128,128,32)):
        
        # interpolate the image
        assert image.ndim == 3 
        zoom_factors = [expected_shape[0] / image.shape[0],  expected_shape[1] / image.shape[1],expected_shape[2] / image.shape[2]]

        # Downsample
        out = zoom(image, zoom=zoom_factors, order=3)  # Using order=3 for cubic interpolation

            
        return out 
    

    def __getitem__(self, index):
        pathA,pathB = self.imageNum[index]
        dataA = nib.load(pathA).get_fdata()
        dataB = nib.load(pathB).get_fdata()
        label_dataA = ""
        label_dataB = ""
                
        # data_ = sio.loadmat(dataPath)
        # dataA = data_['image_ED']
        # dataB = data_['image_ES']
        # label_dataA = data_['label_ED']
        # label_dataB = data_['label_ES']

        # if self.split == 'test':
        #     dataName = dataA.split('/')[-1]
        #     data_ = sio.loadmat(os.path.join(self.dataroot, self.split, 'data_ED2ES', dataName))
        #     dataW = data_['image']
        #     nsample = dataW.shape[-1]
        # else:
        #     nsample = 0
        
        nsample = 1        


        # normalize 
        dataA -= dataA.min()
        dataA /= dataA.std()
        dataA -= dataA.min()
        dataA /= dataA.max()

        dataB -= dataB.min()
        dataB /= dataB.std()
        dataB -= dataB.min()
        dataB /= dataB.max()


        # cropping 
        if self.resize_method == 'crop':
            nh, nw, nd = dataA.shape
            sh = int((nh - self.fineSize[0]) / 2)
            sw = int((nw - self.fineSize[1]) / 2)
            dataA = dataA[sh:sh + self.fineSize[0], sw:sw + self.fineSize[1]]
            dataB = dataB[sh:sh + self.fineSize[0], sw:sw + self.fineSize[1]]
            label_dataA = label_dataA[sh:sh + self.fineSize[0], sw:sw + self.fineSize[1]]
            label_dataB = label_dataB[sh:sh + self.fineSize[0], sw:sw + self.fineSize[1]]

            if nd >= 32:
                sd = int((nd - self.fineSize[2]) / 2)
                dataA = dataA[..., sd:sd + self.fineSize[2]]
                dataB = dataB[..., sd:sd + self.fineSize[2]]
                label_dataA = label_dataA[..., sd:sd + self.fineSize[2]]
                label_dataB = label_dataB[..., sd:sd + self.fineSize[2]]
            else:
                sd = int((self.fineSize[2] - nd) / 2)
                dataA_ = np.zeros(self.fineSize)
                dataB_ = np.zeros(self.fineSize)
                dataA_[:, :, sd:sd + nd] = dataA
                dataB_[:, :, sd:sd + nd] = dataB
                label_dataA_ = np.zeros(self.fineSize)
                label_dataB_ = np.zeros(self.fineSize)
                label_dataA_[:, :, sd:sd + nd] = label_dataA
                label_dataB_[:, :, sd:sd + nd] = label_dataB
                dataA, dataB = dataA_, dataB_
                label_dataA, label_dataB = label_dataA_, label_dataB_
        else: 
            # interpolate instead to 128x128x32 as expected 
            dataA = self._pad_image(dataA,expected_shape=self.fineSize)
            dataB = self._pad_image(dataB,expected_shape=self.fineSize)
            
            # dataA[dataA<0] = 0
            # dataB[dataB<0] = 0


        [data, label] = Util.transform_augment([dataA, dataB], split=self.split, min_max=(-1, 1))

        #return {'M': data, 'F': label, 'MS': label_dataA, 'FS': label_dataB, 'nS':nsample, 'P':pathA, 'Index': index}
        return {'S': data, 'T': label, 'SL': label_dataA, 'TL': label_dataB, 'nS':nsample, 'P':pathA, 'Index': index}
