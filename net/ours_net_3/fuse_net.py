import torch
from torch import nn
from net.ours_net_2.image_branch import RDN as image_branch
from net.ours_net_2.gra_branch import T2Net as fre_branch

class fuse_net(nn.Module):
    def __init__(self,):
        super(fuse_net, self).__init__()
        self.image_branch = image_branch(scale_factor=2)
        self.fre_branch = fre_branch(upscale_factor=2)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(4)
        self.bn3 = nn.BatchNorm2d(1) 
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, t1,t2_lr): 
        t2_sr = self.image_branch(t2_lr)
        t1_srspace,t2_srspace = self.fre_branch(t1,t2_lr)
        t2 = t2_sr + t2_srspace
        out = self.conv1(t2)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act(out)

        return out,t2_srspace