import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
from config import arguments
args = arguments()
from loss import MSE_and_SSIM_loss,get_loss_fn
from net.ours_net_2.spsr_transfomer.t1_t2_ada_chuan_downshuffletwoswin import SPSRNet as sr_model
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import progressbar
from torch.autograd import Variable 
from torch.utils.data import DataLoader
from net.ours_net_2.spsr_transfomer.t1_t2_ada_chuan_downshuffletwoswin import Get_gradient_nopadding
from dataset import ourstraindata as traindata 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def save_ckpt(state ,model,save_dir='./logs'):
    save_path = save_dir + '/' +str(model) + '_' + 'best_model.pt'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(state, save_path)
    
def calc_psnr_255(img1, img2,max=255.0):
    mse = ((img1-img2)**2).mean()
    return 10. * ((max**2)/(mse)).log10()
def calc_psnr_1(img1, img2,max=1.0):
    mse = ((img1-img2)**2).mean()
    return 10. * ((max**2)/(mse)).log10()
 
def volume2frame(volume):
    b,g,c,k=volume.shape
    ans_volume = volume.contiguous().view(-1,c,k).unsqueeze(1)
    return ans_volume

class train(object):
     def __init__(self, model, args):  
        self.dataset = traindata(data_root=r'/home/data/Task01_BrainTumour/imagesTr/',scale=2)
        self.model = model().cuda()
        self.model.train()
        self.gradient = Get_gradient_nopadding()
        #self.model.load_state_dict(torch.load(r'/home/liangwang/MRI_sr/logs/ourschuan_best_model.pt'))
       # self.model = torch.nn.DataParallel(model(),device_ids=[0,1]).cuda()
        self.num_epochs = args.__dict__.pop('num_epochs', 400)
        self.batch_size = args.__dict__.pop('batch_size', 2)
        self.transfomer_lr = args.__dict__.pop('ours_lr', 1e-4)
        #self.model = nn.DataParallel(self.model)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.transfomer_lr, weight_decay=1e-4)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        self.loss_fn = MSE_and_SSIM_loss()
        self.loss_mse = torch.nn.MSELoss()
        #self.fine_tune = kwargs.pop('fine_tune', False)
        self.verbose = args.__dict__.pop('verbose', True)
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size,
                                shuffle=True, num_workers=32)

        best_model=30
        self.hist_loss=[]
        self.hist_psnr=[]
        for epoch in range(self.num_epochs):
            num_batchs = len(self.dataset) // self.batch_size
            if self.verbose:
                bar = progressbar.ProgressBar(num_batchs)
                bar.start()
            running_loss = 0
            running_psnr = 0 
            for i, sample in enumerate(dataloader):
                lr_t2,hr_t2,hr_t1 =  sample['lr_t2'].float().cuda(),sample['hr_t2'].float().cuda(),sample['hr_t1'].float().cuda()
                lr_t2,hr_t2,hr_t1 = volume2frame(lr_t2),volume2frame(hr_t2),volume2frame(hr_t1)
                
                #input_batch_t2, label_batch_t2 = volume2frame(input_batch_t2),volume2frame(label_batch_t2)
                self.optimizer.zero_grad()
                x_out_branch,x_out = self.model(lr_t2,hr_t1)
                #print(t1_branch.shape)
                #print(x_out.shape) 
                #print(x_out_branch.shape)
                loss = self.loss_fn(x_out, hr_t2) + 0.5*self.loss_fn(x_out_branch, self.gradient(hr_t2))
                running_loss += loss.item()
                psnr = calc_psnr_1(x_out, hr_t2)
                running_psnr += psnr.item()
                # Backward + update
                loss.backward()
                #nn.utils.clip_grad_norm_(self.model.parameters(), 0.4)
                self.optimizer.step()
    
                if self.verbose: 
                    bar.update(i)
            self.scheduler.step()
            average_loss = running_loss / num_batchs
            self.hist_loss.append(average_loss)
            average_psnr = running_psnr / num_batchs
            self.hist_psnr.append(average_psnr)
            if self.verbose:
                print('Epoch  %5d, loss %.5f , psnr %.4f'\
                      % (epoch, average_loss, average_psnr))
            if abs(average_psnr - best_model)<15:#and average_psnr>best_model:
                save_ckpt(state=self.model.state_dict(),model="sepoch"+str(epoch)+'x2')
                best_model =average_psnr
                
if __name__ == "__main__":
    train(sr_model,  args)
    from test import *
    test(sr_model,  args)
