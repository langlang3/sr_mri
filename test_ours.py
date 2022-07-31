import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
from config import arguments
args = arguments()
#from net.ours_net.spsr_transfomer.t1_t2_fea import SPSRNet as sr_model
from net.ours_net_4.spsr_transfomer.t1_t2_ada_chuan_downshuffletwoswin import SPSRNet as sr_model
#from net.ours_net_3.spsr_transfomer.t1_t1_crossscale_spsr import SPSRNet as sr_model
#from net.ours_net.spsr_transfomer.t1_t2_spsr import SPSRNet as sr_model
import progressbar
from pytorch_ssim import SSIM as calc_ssim
calc_ssim = calc_ssim()
from torch.utils.data import DataLoader
from dataset import ourstestdata as ourstestdata
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from thop import profile
from thop import clever_format
def calc_psnr_255(img1, img2,max=255.0):
    mse = ((img1-img2)**2).mean()
    return 10. * ((max**2)/(mse)).log10()
def calc_psnr_1(img1, img2,max=1.0):
    mse = ((img1-img2)**2).mean() 
    return 10. * ((max**2)/(mse)).log10()
 
def volume2frame(volume):
    b,g,c,k=volume.shape
    ans_volume = volume.contiguous().view(-1,c,k).unsqueeze(1)
    #print(new_volume.shape)
    return ans_volume 

def strip_prefix(state_dict, prefix='module.'):
    if not all(key.startswith(prefix) for key in state_dict.keys()):
        return state_dict
    stripped_state_dict = {}
    for key in list(state_dict.keys()):
        stripped_state_dict[key.replace(prefix, '',1)] = state_dict.pop(key)
    return stripped_state_dict

class test(object):
     def __init__(self, model, args):  
        self.dataset = ourstestdata(data_root=r'/home/data/Task01_BrainTumour/imagesTs/',scale=4)
        self.model = model().cuda()
        self.model.eval()
        t2=torch.randn(1,1,60,60).cuda()
        t1=torch.randn(1,1,240,240).cuda()
        flops, params = profile(self.model, (t2,t1))
        flops, params = clever_format([flops, params],"%.3f")
        print(flops, params)
        return
        #self.model.load_state_dict(torch.load(r'/home/liangwang/MRI_sr/logs/oursspsrt1t1crossscalex3_best_model.pt'))
        model_name = r'/home/liangwang/mm_mrisr/logs/epoch36x2_best_model.pt'
        self.model.load_state_dict(strip_prefix(torch.load(model_name)))
        dataloader = DataLoader(self.dataset, batch_size=1,
                                shuffle=False, num_workers=4)
        average_psnr = 0
        average_ssim = 0 
        std_psnr = []
        std_ssim = []
      #  mi,ma=100000,-100000
        for i, sample in enumerate(dataloader):
            lr_t2,hr_t2,hr_t1 =  sample['lr_t2'].float().cuda(),sample['hr_t2'].float().cuda(),sample['hr_t1'].float().cuda()
            lr_t2,hr_t2,hr_t1 = volume2frame(lr_t2),volume2frame(hr_t2),volume2frame(hr_t1)
            #input_batch_t2, label_batch_t2 = volume2frame(input_batch_t2),volume2frame(label_batch_t2)
            with torch.no_grad():
                x_out_branch_1, x_out_1, = self.model(lr_t2[0:20,:,:,:],hr_t1[0:20,:,:,:])
                x_out_branch_2, x_out_2, = self.model(lr_t2[20:40,:,:,:],hr_t1[20:40,:,:,:])
                x_out_branch_3, x_out_3, = self.model(lr_t2[40:60,:,:,:],hr_t1[40:60,:,:,:])
                x_out_branch_4, x_out_4, = self.model(lr_t2[60:80,:,:,:],hr_t1[60:80,:,:,:])
                x_out_branch_5, x_out_5, = self.model(lr_t2[80:100,:,:,:],hr_t1[80:100,:,:,:])
                x_out_branch_6, x_out_6, = self.model(lr_t2[100:120,:,:,:],hr_t1[100:120,:,:,:])
                x_out_branch_7, x_out_7, = self.model(lr_t2[120:140,:,:,:],hr_t1[120:140,:,:,:])
                x_out_branch_8, x_out_8, = self.model(lr_t2[140:,:,:,:],hr_t1[140:,:,:,:])
            
            psnr_1 = calc_psnr_1(x_out_1, hr_t2[0:20,:,:,:]).item()
            ssim_1 = calc_ssim(x_out_1, hr_t2[0:20,:,:,:]).item() 
            psnr_2 = calc_psnr_1(x_out_2, hr_t2[20:40,:,:,:]).item()
            ssim_2 = calc_ssim(x_out_2, hr_t2[20:40,:,:,:]).item()
            psnr_3 = calc_psnr_1(x_out_3, hr_t2[40:60,:,:,:]).item()
            ssim_3 = calc_ssim(x_out_3, hr_t2[40:60,:,:,:]).item() 
            psnr_4 = calc_psnr_1(x_out_4, hr_t2[60:80,:,:,:]).item()
            ssim_4= calc_ssim(x_out_4, hr_t2[60:80,:,:,:]).item()
            psnr_5 = calc_psnr_1(x_out_5, hr_t2[80:100,:,:,:]).item()
            ssim_5= calc_ssim(x_out_5, hr_t2[80:100,:,:,:]).item()
            psnr_6 = calc_psnr_1(x_out_6, hr_t2[100:120,:,:,:]).item()
            ssim_6= calc_ssim(x_out_6, hr_t2[100:120,:,:,:]).item()
            psnr_7 = calc_psnr_1(x_out_7, hr_t2[120:140,:,:,:]).item()
            ssim_7= calc_ssim(x_out_7, hr_t2[120:140,:,:,:]).item()
            psnr_8 = calc_psnr_1(x_out_8, hr_t2[140:,:,:,:]).item()
            ssim_8= calc_ssim(x_out_8, hr_t2[140:,:,:,:]).item()
            psnr=(psnr_1+psnr_2+psnr_3+psnr_4+psnr_5+psnr_6+psnr_7+psnr_8)/8.0
            ssim=(ssim_1+ssim_2+ssim_3+ssim_4+ssim_5+ssim_6+ssim_7+ssim_8)/8.0
            average_psnr = average_psnr +psnr
            average_ssim = average_ssim +ssim
            std_psnr.append(psnr) 
            std_ssim.append(ssim)
            print('volume %s,  psnr %.4f, ssim %.4f '\
                  % (sample['filename'], psnr,ssim))
        print('the average psnr and ssim is,  average_psnr: %.4f,average_ssim: %.4f,std_psnr: %.4f,std_ssim: %.4f' \
                  % ( average_psnr/(i+1),average_ssim/(i+1),np.std(std_psnr),np.std(std_ssim)))
        print(model_name)

                
if __name__ == "__main__":
    test(sr_model,  args)
    

                
