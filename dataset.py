import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import nibabel as nib
import random
from copy import deepcopy

class traindata(Dataset):
    def __init__(self, data_root,scale=4):
        super().__init__()
        self.data_root = data_root
        self.trainlist=os.listdir(self.data_root)
        self.scale=scale
        random.shuffle(self.trainlist)
        self.trainlist = self.trainlist#[::2]
        sample = dict()
        
    def volume2frame(self,volume):
        volume=volume.transpose(2,0,1)
        return volume
        
    def degrade(self,hr_data):
       
        if self.scale ==4:
            imgfft = np.fft.fft2(hr_data)
            imgfft=np.fft.fftshift(imgfft)
            imgfft = imgfft[:,90 : 150, 90 : 150 ]
            imgfft=np.fft.ifftshift(imgfft)
            imgifft = np.fft.ifft2(imgfft)
            img_out = abs(imgifft)
            
        if self.scale ==3:
            imgfft = np.fft.fft2(hr_data)
            imgfft=np.fft.fftshift(imgfft)
            imgfft = imgfft[:,80 : 160, 80 : 160 ]
            imgfft=np.fft.ifftshift(imgfft)
            imgifft = np.fft.ifft2(imgfft)
            img_out = abs(imgifft)
            
        if self.scale ==2:
            imgfft = np.fft.fft2(hr_data)
            imgfft=np.fft.fftshift(imgfft)
            imgfft = imgfft[:,60 : 180, 60 : 180 ]
            imgfft=np.fft.ifftshift(imgfft)
            imgifft = np.fft.ifft2(imgfft)
            img_out = abs(imgifft)
        max_d,min_d=max(img_out.reshape(-1)),min(img_out.reshape(-1))
        if (max_d - min_d) !=0 :
            img_out = (img_out-min(img_out.reshape(-1)))/(max_d-min_d)
        return img_out
    
    
    def __getitem__(self, index):
        volumepath = os.path.join(self.data_root, self.trainlist[index])
        volume=nib.load(volumepath)
        volumeIn=np.array([volume.get_fdata()])
        volumeIn_t1=volumeIn[:,:,:,:,0].squeeze()
        volumeIn_t2=volumeIn[:,:,:,:,2].squeeze()
        volumeIn_t1=self.volume2frame(volumeIn_t1)
        volumeIn_t2=self.volume2frame(volumeIn_t2)
        begin = random.randint(10,140)
        volumeIn_t1=volumeIn_t1[begin:begin+2,:,:]
        volumeIn_t2=volumeIn_t2[begin:begin+2,:,:]
       # if random.random() >= 0.5:
       #     volumeIn_t2 = volumeIn_t2[:,::-1,:]
       # if random.random() >= 0.5:
       #     volumeIn_t1 = volumeIn_t1[::-1,:,:]#.copy()
       #     volumeIn_t2 = volumeIn_t2[::-1,:,:]
       # if random.random() >= 0.5:
       #     volumeIn_t1 = volumeIn_t1[:,:,::-1]#.copy()
       #     volumeIn_t2 = volumeIn_t2[:,:,::-1]
        max_d,min_d=volumeIn.reshape(-1).max() , volumeIn.reshape(-1).min()
        if (max_d-min_d) !=0 :
            volumeIn_t1 = (volumeIn_t1 - volumeIn_t1.reshape(-1).min())/(max_d-min_d)
            volumeIn_t2 = (volumeIn_t2 - volumeIn_t1.reshape(-1).min())/(max_d-min_d)
            
        volumeDown_t1=self.degrade(volumeIn_t1)
        volumeDown_t2=self.degrade(volumeIn_t2)
        volumeTar_t1=torch.from_numpy(volumeIn_t1.copy())
        volumeDown_t1=torch.from_numpy(volumeDown_t1.copy())
        volumeTar_t2=torch.from_numpy(volumeIn_t2.copy())
        volumeDown_t2=torch.from_numpy(volumeDown_t2.copy())
        sample = {'lr_t1': volumeDown_t1,'lr_t2': volumeDown_t2, 'hr_t1': volumeTar_t1,'hr_t2': volumeTar_t2, 'filename':self.trainlist[index]}
       # sample = {'lr_t1': volumeDown_t1,'lr_t2': volumeDown_t2, 'hr_t1': volumeTar_t1,'hr_t2': volumeTar_t2, 't2_re': t2_re,'filename':self.trainlist[index]}
        return sample

    def __len__(self):
        return len(self.trainlist)

class testdata(Dataset):
    def __init__(self, data_root,scale=4):
        super().__init__()
        self.data_root = data_root
        self.testlist=os.listdir(self.data_root)
        self.scale=scale
        self.testlist = self.testlist#[::2]
        sample = dict()
    
    def volume2frame(self,volume):
        volume=volume.transpose(2,0,1)
        return volume
    
    def degrade(self,hr_data,):
        if self.scale ==4:
            imgfft = np.fft.fft2(hr_data)
            imgfft=np.fft.fftshift(imgfft)   
            imgfft = imgfft[:,90 : 150, 90 : 150 ]
            imgfft=np.fft.ifftshift(imgfft)
            imgifft = np.fft.ifft2(imgfft)
            img_out = abs(imgifft)
            
        if self.scale ==3:
            imgfft = np.fft.fft2(hr_data)
            imgfft=np.fft.fftshift(imgfft)
            imgfft = imgfft[:,80 : 160, 80 : 160 ]
            imgfft=np.fft.ifftshift(imgfft)
            imgifft = np.fft.ifft2(imgfft)
            img_out = abs(imgifft)
            
        if self.scale ==2:
            imgfft = np.fft.fft2(hr_data)
            imgfft=np.fft.fftshift(imgfft)
            imgfft = imgfft[:,60 : 180, 60 : 180 ]
            imgfft=np.fft.ifftshift(imgfft)
            imgifft = np.fft.ifft2(imgfft)
            img_out = abs(imgifft)
        img_out = (img_out-min(img_out.reshape(-1)))/(max(img_out.reshape(-1))-min(img_out.reshape(-1)))
        return img_out
    
    
    def __getitem__(self, index):
        volumepath = os.path.join(self.data_root, self.testlist[index])
        volume=nib.load(volumepath)
        volumeIn=np.array([volume.get_fdata()])
        volumeIn_t1=volumeIn[:,:,:,:,0].squeeze()
        volumeIn_t2=volumeIn[:,:,:,:,2].squeeze()
        volumeIn_t1=self.volume2frame(volumeIn_t1)
        volumeIn_t2=self.volume2frame(volumeIn_t2)
            
        max_d,min_d=volumeIn.reshape(-1).max() , volumeIn.reshape(-1).min()
        if (max_d-min_d) !=0 :
            volumeIn_t1 = (volumeIn_t1 - volumeIn_t1.reshape(-1).min())/(max_d-min_d)
            volumeIn_t2 = (volumeIn_t2 - volumeIn_t1.reshape(-1).min())/(max_d-min_d)
            
        volumeDown_t1=self.degrade(volumeIn_t1)
        volumeDown_t2=self.degrade(volumeIn_t2)
        volumeTar_t1=torch.from_numpy(volumeIn_t1)
        volumeDown_t1=torch.from_numpy(volumeDown_t1)
        volumeTar_t2=torch.from_numpy(volumeIn_t2)
        volumeDown_t2=torch.from_numpy(volumeDown_t2)
        
        sample = {'lr_t1': volumeDown_t1,'lr_t2': volumeDown_t2, 'hr_t1': volumeTar_t1,'hr_t2': volumeTar_t2, 'filename':self.testlist[index]}
        #print(max(volumeDown_t2.reshape(-1)))
        #print(max(volumeTar_t2.reshape(-1)))
        return sample

    def __len__(self):
        return len(self.testlist)
        
        
####################################################################################################################################
class ourstraindata(Dataset):
    def __init__(self, data_root,scale=4):
        super().__init__()
        self.data_root = data_root
        self.trainlist=os.listdir(self.data_root)
        self.scale=scale
        random.shuffle(self.trainlist)
        self.trainlist = self.trainlist#[::2]
        sample = dict()
    
    def volume2frame(self,volume):
        volume=volume.transpose(2,0,1)
        return volume
        
    def degrade(self,hr_data):
        if self.scale ==4:
            imgfft = np.fft.fft2(hr_data)
            imgfft=np.fft.fftshift(imgfft)
            imgfft = imgfft[:,90 : 150, 90 : 150 ]
            imgfft=np.fft.ifftshift(imgfft)
            imgifft = np.fft.ifft2(imgfft)
            img_out = abs(imgifft)
            
        if self.scale ==3:
            imgfft = np.fft.fft2(hr_data)
            imgfft=np.fft.fftshift(imgfft)
            imgfft = imgfft[:,80 : 160, 80 : 160 ]
            imgfft=np.fft.ifftshift(imgfft)
            imgifft = np.fft.ifft2(imgfft)
            img_out = abs(imgifft)
            
        if self.scale ==2:
            imgfft = np.fft.fft2(hr_data)
            imgfft=np.fft.fftshift(imgfft)
            imgfft = imgfft[:,60 : 180, 60 : 180 ]
            imgfft=np.fft.ifftshift(imgfft)
            imgifft = np.fft.ifft2(imgfft)
            img_out = abs(imgifft)
        if max(img_out.reshape(-1))-min(img_out.reshape(-1)) !=0:
            img_out = (img_out-min(img_out.reshape(-1)))/(max(img_out.reshape(-1))-min(img_out.reshape(-1)))
        return img_out
    

    
    def __getitem__(self, index):
        volumepath = os.path.join(self.data_root, self.trainlist[index])
        volume=nib.load(volumepath)
        volumeIn=np.array([volume.get_fdata()])
        volumeIn_t1=volumeIn[:,:,:,:,0].squeeze()
        volumeIn_t2=volumeIn[:,:,:,:,2].squeeze()
        volumeIn_t1=self.volume2frame(volumeIn_t1)
        volumeIn_t2=self.volume2frame(volumeIn_t2)
        begin = random.randint(10,145)
        volumeIn_t1=volumeIn_t1[begin:begin+4,:,:]
        volumeIn_t2=volumeIn_t2[begin:begin+4,:,:]   
        max_d,min_d=volumeIn.reshape(-1).max() , volumeIn.reshape(-1).min()
        if (max_d-min_d) !=0 :
            volumeIn_t1 = (volumeIn_t1 - volumeIn_t1.reshape(-1).min())/(max_d-min_d)
            volumeIn_t2 = (volumeIn_t2 - volumeIn_t1.reshape(-1).min())/(max_d-min_d)
            
        hr_t1=volumeIn_t1
       # volumespacedown_t2=self.spacedown(volumeIn_t2)
        volumeDown_t2=self.degrade(volumeIn_t2)
        #t2_re=self.decon(volumeIn_t2, scale=self.scale)
        #print(volumeDown.shape)
        hr_t1=torch.from_numpy(hr_t1.copy())
        #volumespacedown_t2=torch.from_numpy(volumespacedown_t2.copy())
        volumeDown_t2=torch.from_numpy(volumeDown_t2.copy())
        volumeIn_t2=torch.from_numpy(volumeIn_t2.copy())
        #t2_re=torch.from_numpy(t2_re.copy())
        sample = {'hr_t1': hr_t1, 'lr_t2': volumeDown_t2,'hr_t2': volumeIn_t2,'filename':self.trainlist[index]}
       # sample = {'lr_t1': volumeDown_t1,'lr_t2': volumeDown_t2, 'hr_t1': volumeTar_t1,'hr_t2': volumeTar_t2, 't2_re': t2_re,'filename':self.trainlist[index]}
        return sample

    def __len__(self):
        return len(self.trainlist)

class ourstestdata(Dataset):
    def __init__(self, data_root,scale=4):
        super().__init__()
        self.data_root = data_root
        self.testlist=os.listdir(self.data_root)
        self.scale=scale
        self.testlist = self.testlist#[::2]
        sample = dict()
        
    def volume2frame(self,volume):
        volume=volume.transpose(2,0,1)
        return volume
        
    def degrade(self,hr_data):
        if self.scale ==4:
            imgfft = np.fft.fft2(hr_data)
            imgfft=np.fft.fftshift(imgfft)
            imgfft = imgfft[:,90 : 150, 90 : 150 ]
            imgfft=np.fft.ifftshift(imgfft)
            imgifft = np.fft.ifft2(imgfft)
            img_out = abs(imgifft)
            
        if self.scale ==3:
            imgfft = np.fft.fft2(hr_data)
            imgfft=np.fft.fftshift(imgfft)
            imgfft = imgfft[:,80 : 160, 80 : 160 ]
            imgfft=np.fft.ifftshift(imgfft)
            imgifft = np.fft.ifft2(imgfft)
            img_out = abs(imgifft)
            
        if self.scale ==2:
            imgfft = np.fft.fft2(hr_data)
            imgfft=np.fft.fftshift(imgfft)
            imgfft = imgfft[:,60 : 180, 60 : 180 ]
            imgfft=np.fft.ifftshift(imgfft)
            imgifft = np.fft.ifft2(imgfft)
            img_out = abs(imgifft)
        if max(img_out.reshape(-1))-min(img_out.reshape(-1)) !=0:
            img_out = (img_out-min(img_out.reshape(-1)))/(max(img_out.reshape(-1))-min(img_out.reshape(-1)))
        return img_out
    

    
    def __getitem__(self, index):
        volumepath = os.path.join(self.data_root, self.testlist[index])
        volume=nib.load(volumepath)
        volumeIn=np.array([volume.get_fdata()])
        volumeIn_t1=volumeIn[:,:,:,:,0].squeeze()
        volumeIn_t2=volumeIn[:,:,:,:,2].squeeze()
        volumeIn_t1=self.volume2frame(volumeIn_t1)
        volumeIn_t2=self.volume2frame(volumeIn_t2)
        #begin = random.randint(0,120)
        #volumeIn_t1=volumeIn_t1[10:140,:,:]
        #volumeIn_t2=volumeIn_t2[10:140,:,:]   
        max_d,min_d=volumeIn.reshape(-1).max() , volumeIn.reshape(-1).min()
        if (max_d-min_d) !=0 :
            volumeIn_t1 = (volumeIn_t1 - volumeIn_t1.reshape(-1).min())/(max_d-min_d)
            volumeIn_t2 = (volumeIn_t2 - volumeIn_t1.reshape(-1).min())/(max_d-min_d)
            
        hr_t1=volumeIn_t1
       # volumespacedown_t2=self.spacedown(volumeIn_t2)
        volumeDown_t2=self.degrade(volumeIn_t2)
        #t2_re=self.decon(volumeIn_t2, scale=self.scale)
        #print(volumeDown.shape)
        hr_t1=torch.from_numpy(hr_t1.copy())
        #volumespacedown_t2=torch.from_numpy(volumespacedown_t2.copy())
        volumeDown_t2=torch.from_numpy(volumeDown_t2.copy())
        volumeIn_t2=torch.from_numpy(volumeIn_t2.copy())
        #t2_re=torch.from_numpy(t2_re.copy())
        sample = {'hr_t1': hr_t1, 'lr_t2': volumeDown_t2,'hr_t2': volumeIn_t2,'filename':self.testlist[index]}
       # sample = {'lr_t1': volumeDown_t1,'lr_t2': volumeDown_t2, 'hr_t1': volumeTar_t1,'hr_t2': volumeTar_t2, 't2_re': t2_re,'filename':self.trainlist[index]}
        return sample

    def __len__(self):
        return len(self.testlist)
        
if __name__ == '__main__':
    traindata = traindata(r'/home/data/Task01_BrainTumour/imagesTr/')
    print("the length of traindata :",len(traindata))
    print(traindata[0]['hr_t1'].shape)
    print(traindata[0]['hr_t2'].shape)
    print(traindata[0]['lr_t1'].shape)
    print(traindata[0]['lr_t2'].shape)
    testdata = testdata(r'/home/data/Task01_BrainTumour/imagesTs/')
    print("the length of testdata :",len(testdata))
    print(testdata[0]['lr_t2'].shape)
    print("the length of testdata :",len(testdata))
    print(testdata[0]['lr_t2'].shape)