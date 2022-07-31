import torch
from torch import nn
#adaptive instance norm
class SAM(nn.Module):
    def __init__(self, nf=32, use_residual=True, learnable=True):
        super(SAM, self).__init__()

        self.learnable = learnable
        self.norm_layer = nn.InstanceNorm2d(nf, affine=False)
        self.pixel_shuffle = nn.PixelShuffle(3)
        if self.learnable:
            self.conv_noshared = nn.Sequential(nn.Conv2d(nf , nf*9, 3, 1, 1, bias=True),
                                             nn.GELU())
            self.conv_shared = nn.Sequential(nn.Conv2d(nf*2 , nf, 3, 1, 1, bias=True),
                                             nn.GELU())
            self.conv_gamma = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.conv_beta = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.use_residual = use_residual

            # initialization
            self.conv_gamma.weight.data.zero_()
            self.conv_beta.weight.data.zero_()
            self.conv_gamma.bias.data.zero_()
            self.conv_beta.bias.data.zero_()

    def forward(self, lr, ref_ori):#lr (b,32,80,80)  ref (b,32,240,240)
        b,c,h,w = lr.shape
        lr_mean = torch.mean(lr.reshape(b,c,h*w), dim=-1, keepdim=True).reshape(b,c,1,1)
        lr_std = torch.std(lr.reshape(b,c,h*w), dim=-1, keepdim=True).reshape(b,c,1,1)
        lr = self.conv_noshared(lr)
        lr = self.pixel_shuffle(lr)
        ref_normed = self.norm_layer(ref_ori)#b,1152,40,40
        style = self.conv_shared(torch.cat([lr, ref_ori], dim=1))#b,512,60,60
        gamma = self.conv_gamma(style)
        beta = self.conv_beta(style)
        if self.learnable:
            if self.use_residual:
                gamma = gamma + lr_std
                beta = beta + lr_mean
        out = ref_normed * gamma + beta
        return out
        
        
if __name__ == "__main__":
    net=SAM()
    lr=torch.randn(1,32,80,80)
    ref=torch.randn(1,32,240,240)
    out=net(lr,ref)
    print(out.shape)