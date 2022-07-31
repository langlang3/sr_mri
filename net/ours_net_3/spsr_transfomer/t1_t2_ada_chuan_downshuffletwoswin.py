import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import cv2
from net.ours_net_3.spsr_transfomer import block as B
from net.ours_net_3.spsr_transfomer.downshuffle_twoswin import SwinTransformer as gra_transfomer
from net.ours_net_3.spsr_transfomer.one_branch_mutil_head_attention import CrossCMMT as sam
from net.ours_net_3.spsr_transfomer.one_branch_mutil_head_attention import Select as Select
from net.ours_net_3.spsr_transfomer.fea_adaptive import SAM as t1_adaptive

class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0], 
                    [0, 0, 0], 
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0], 
                    [-1, 0, 1], 
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0).cuda()
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0).cuda()
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False)
        
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False)
        
    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim = 1)
        return x

class SPSRNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nf=32, nb=23, gc=8, upscale=3, norm_type=None, \
            act_type='gelu', mode='CNA', upsample_mode='pixelshuffle'):
        super(SPSRNet, self).__init__()

        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1
        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        rb_blocks = [B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)]        
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)
        self.sam1=sam()
        self.sam2=sam()
        self.sam3=sam()
        self.sam4=sam()
        self.conv_f1 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None)
        self.conv_f2 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None)
        self.conv_f3 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None)
        self.conv_f4 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None)
        self.t1_adap_1 = t1_adaptive()
        self.t1_adap_2 = t1_adaptive()
        self.t1_adap_3 = t1_adaptive()
        self.t1_adap_4 = t1_adaptive()
        self.select1 = Select()
        self.select2 = Select()
        self.select3 = Select()
        self.select4 = Select()
        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        
        self.HR_conv0_new = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        self.HR_conv1_new = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),\
            *upsampler, self.HR_conv0_new)
        self.get_g_nopadding = Get_gradient_nopadding()
        self.t2_fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        self.t2_fea_conv_s = B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA')
        self.t1_fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)#32 channels
        self.t1_fea_conv_s = B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA')
        self.b_concat_1 = B.conv_block(2*nf, nf, kernel_size=3, norm_type=None, act_type = None)
        self.b_block_1 = gra_transfomer(dim=32, input_resolution=(80,80), num_heads=4)
        self.b_concat_2 = B.conv_block(2*nf, nf, kernel_size=3, norm_type=None, act_type = None)
        self.b_block_2 = gra_transfomer(dim=32, input_resolution=(80,80), num_heads=4)
        self.t1_concat_2 = B.conv_block(2*nf, nf, kernel_size=3, norm_type=None, act_type = None)
        self.b_concat_3 = B.conv_block(2*nf, nf, kernel_size=3, norm_type=None, act_type = None)
        self.b_block_3 = gra_transfomer(dim=32, input_resolution=(80,80), num_heads=4)
        self.t1_concat_3 = B.conv_block(2*nf, nf, kernel_size=3, norm_type=None, act_type = None)
        self.b_concat_4 = B.conv_block(2*nf, nf, kernel_size=3, norm_type=None, act_type = None)
        self.b_block_4 = gra_transfomer(dim=32, input_resolution=(80,80), num_heads=4)
        self.t1_concat_4 = B.conv_block(2*nf, nf, kernel_size=3, norm_type=None, act_type = None)
        self.b_LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)
        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            b_upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            b_upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        b_HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        b_HR_conv1 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        self.b_module = B.sequential(*b_upsampler, b_HR_conv0, b_HR_conv1)
        self.conv_w = B.conv_block(nf, out_nc, kernel_size=1, norm_type=None, act_type=act_type)
        self.f_concat = B.conv_block(nf*2, nf, kernel_size=3, norm_type=None, act_type=None)
        self.f_block = B.RRDB(nf*2, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA')
        self.f_HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        self.f_HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

    def forward(self, x,t1):    
        x_grad = self.get_g_nopadding(x)
        t1_grad = t1.clone()
        x_b_fea = self.t2_fea_conv(x_grad)#t2 grad branch
        t1_b_fea = self.t1_fea_conv(t1_grad)#t1 grad branch
        x_b_fea = self.t2_fea_conv_s(x_b_fea)
        t1_b_fea = self.t1_fea_conv_s(t1_b_fea)
        x = self.model[0](x)  
        x, block_list = self.model[1](x)
        x_ori = x.clone()
        for i in range(5):
            x = block_list[i](x)
        x_fea1 = x.clone()
        x_fea1_temp = self.conv_f1(x_fea1)
        x_cat_1 = torch.cat([x_b_fea, x_fea1_temp], dim=1)
        x_cat_1 = self.b_concat_1(x_cat_1)#64->32
        x_cat_1 = self.b_block_1(x_cat_1)#32
        t1_b_fea_tmp = self.t1_adap_1(x_cat_1,t1_b_fea)
        t1_cat_1 = self.sam1(x_cat_1,t1_b_fea_tmp)#32
        t1_cat_1 = self.select2(x_fea1_temp,x_cat_1)
        x_cat_1 = x_cat_1 + t1_cat_1
        x = x  + t1_cat_1
        
        for i in range(5):
            x = block_list[i+5](x)
        x_fea2 = x.clone()
        x_fea2_temp = self.conv_f2(x_fea2)
        x_cat_2 = torch.cat([x_cat_1, x_fea2_temp], dim=1)
        x_cat_2 = self.b_concat_2(x_cat_2)#32
        x_cat_2 = self.b_block_2(x_cat_2)#64
        t2_b_fea_tmp = self.t1_adap_2(x_cat_2,t1_b_fea)
        t1_cat_2 = self.sam2(x_cat_2,t2_b_fea_tmp)
        x_cat_2 = x_cat_2 + t1_cat_2
        t1_cat_2 = self.select2(x_fea2_temp,x_cat_2)
        x = x + t1_cat_2
        
        for i in range(5):
            x = block_list[i+10](x)
        x_fea3 = x.clone()
        x_fea3_temp = self.conv_f3(x_fea3)
        x_cat_3 = torch.cat([t1_cat_2, x_fea3_temp], dim=1)
        x_cat_3 = self.b_concat_3(x_cat_3)
        x_cat_3 = self.b_block_3(x_cat_3)
        t3_b_fea_tmp = self.t1_adap_3(t1_cat_2,t1_b_fea)
        t1_cat_3 = self.sam3(x_cat_3,t3_b_fea_tmp)
        x_cat_3 = x_cat_3 + t1_cat_3
        t1_cat_3 = self.select3(x_fea3_temp,x_cat_3)
        x = x + t1_cat_3
        
        for i in range(5):
            x = block_list[i+15](x)
        x_fea4 = x.clone()
        x_fea4_temp = self.conv_f4(x_fea4)
        x_cat_4 = torch.cat([x_cat_3, x_fea4_temp], dim=1)
        x_cat_4 = self.b_concat_4(x_cat_4)
        x_cat_4 = self.b_block_4(x_cat_4)
        t4_b_fea_tmp = self.t1_adap_4(x_cat_4,t1_b_fea)
        t1_cat_4 = self.sam4(x_cat_4,t4_b_fea_tmp)
        x_cat_4 = x_cat_4 + t1_cat_4
        t1_cat_4 = self.select4(x_fea4_temp,x_cat_4)
        x = x + t1_cat_4
        x = block_list[20:](x)
        x = x_ori+x
        x= self.model[2:](x)
        x = self.HR_conv1_new(x)
        x_cat_4 = self.b_LR_conv(x_cat_4) + x_cat_4
        x_branch = self.b_module(x_cat_4) 
        x_out_branch = self.conv_w(x_branch) 
        ########
        x_branch_d = x_branch.clone()
        x_f_cat = torch.cat([x_branch_d, x], dim=1)
        x_f_cat = self.f_block(x_f_cat)
        x_out = self.f_concat(x_f_cat)
        x_out = self.f_HR_conv0(x_out) + x_out
        x_out = self.f_HR_conv1(x_out)
        #########
        return x_out_branch, x_out#,attention,proj_value
if __name__ == "__main__":
    net = SPSRNet(in_nc=1, out_nc=1, nf=32, nb=23,).cuda()
    t2_gra=torch.randn(1,1,80,80).cuda()
    t1_gra=torch.randn(1,1,240,240).cuda()
    a=net(t2_gra,t1_gra)
    print(a[0].shape)
    print(a[1].shape)