import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition_downshuffle(x, window_size):#10
    B, C, H, W = x.shape
    h_interval,w_interval=int(H/window_size),int(W/window_size)
    y=[]
    for i in range(h_interval):
        for j in range(w_interval):
            y.append(x[:,:,i::h_interval,j::w_interval])#fold
    windows = torch.stack(y,1).reshape(B,-1,C*window_size*window_size)
    return windows
    
def window_partition_normal(x, window_size):#1,10/2,10
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(B, -1,C*window_size*window_size)
    return windows

def window_reverse_downshuffle(windows, window_size, H, W):
    B,num,dim = windows.shape
    scale = int(pow((H * W / window_size / window_size),0.5))
    x = windows.reshape(B,num,dim//(window_size*window_size),window_size,window_size).reshape(B,num*dim//(window_size*window_size),window_size,window_size)
    pixshuffle = nn.PixelShuffle(scale)
    x = pixshuffle(x)
    return x
    
def window_reverse_normal(windows, window_size, H, W):
    B,num,dim = windows.shape
    scale = int(pow((H * W / window_size / window_size),0.5))
    x = windows.view(B, num, dim//(window_size*window_size),window_size, window_size)
    x = x.permute(0, 2, 1, 3, 4).reshape(B, dim//(window_size*window_size), H, W)
    return x

class LAM_Module(nn.Module):
    def __init__(self,visual=False): 
        super(LAM_Module, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
        self.visual = visual
    def forward(self,x):
        batchsize, head, num, dim = x.size()
        x = x .permute(0,3,1,2)
        proj_query = x.view(batchsize, dim, -1)  
        proj_key = x.view(batchsize, dim, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key) 
        energy_new = energy-torch.max(energy, -1, keepdim=True)[0].expand_as(energy)
        attention = self.softmax(energy)
        proj_value = x.view(batchsize, dim, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(batchsize, dim, head, num)
        out = self.gamma*out + x
        out = out.permute(0,2,3,1)
        return out

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4,  attn_drop=0., proj_drop=0.,visual=False):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads#200 ,dim is 800
        self.scale = head_dim ** -0.5
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv_other = nn.Linear(dim, dim*2, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.layer_att_other = LAM_Module(visual=visual)
        self.visual = visual
        
    def forward(self, t2_grad, t1):
        B_x, N_x, C_x = t2_grad.shape#1,144,800
        t2_grad_copy = t2_grad.clone()
        q = self.to_q(t2_grad).reshape(B_x, N_x, self.num_heads, C_x//self.num_heads).permute(0, 2, 1, 3) #batch,num_head(4),144,200
        kv_other = self.to_kv_other(t1).reshape(B_x, N_x, 2, self.num_heads, C_x//self.num_heads).permute(2, 0, 3, 1, 4)
        k_other, v_other = kv_other[0], kv_other[1]
        attn_other = (q @ k_other.transpose(-2, -1)) * self.scale
        attn_other = attn_other.softmax(dim=-1)
        x_other = (attn_other @ v_other)
        if not self.visual:
            x_other = self.layer_att_other(x_other).transpose(1, 2).reshape(B_x, N_x, C_x)+q.transpose(1, 2).reshape(B_x, N_x, C_x)
        else:
            x_other,attention,proj_value = self.layer_att_other(x_other)
            x_other = x_other.transpose(1, 2).reshape(B_x, N_x, C_x)+q.transpose(1, 2).reshape(B_x, N_x, C_x)
        x = t2_grad_copy + x_other
        x = self.proj(x)
        if self.visual:
            return x,attention,proj_value
        else:
            return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=5, is_downshuffle=False,
                     mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                     act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.W,self.H = input_resolution[0],input_resolution[1]
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.is_downshuffle = is_downshuffle
        self.window_partition_downshuffle = window_partition_downshuffle
        self.window_partition_normal = window_partition_normal
        self.window_reverse_downshuffle = window_reverse_downshuffle
        self.window_reverse_normal = window_reverse_normal
        if min(input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.window_size = min(input_resolution)

        self.norm1 = norm_layer(self.dim*window_size*window_size)
        self.attn = MultiHeadCrossAttention(dim=self.dim*window_size*window_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(self.dim*window_size*window_size)
        mlp_hidden_dim = int(self.dim*window_size*window_size * mlp_ratio)
        self.mlp = Mlp(in_features=self.dim*window_size*window_size, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.pos_embedding = nn.Parameter(torch.ones(1, (input_resolution[0]//window_size)*(input_resolution[1]//window_size), window_size*window_size*dim))#1,144,800
        self.pos_embedding_simi= nn.Parameter(torch.ones(1, (input_resolution[0]//window_size)*(input_resolution[1]//window_size), window_size*window_size*dim))#1,144,800
        self.con1 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
        self.con4 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
        self.ln11 = nn.LayerNorm([80,80])
        self.ln12 = nn.LayerNorm([80,80])
        self.ln21 = nn.LayerNorm([80,80])
        self.ln22 = nn.LayerNorm([80,80])
        self.t1act = nn.GELU()
    def forward(self, x):
        pici,tongdao,chang,kuang=x.shape
        x_simi = x.clone()
        x = self.ln11(x)
        x = self.con1(x)
        x = self.ln12(x)
        x_simi = self.ln21(x_simi)
        x_simi = self.con4(x_simi)
        x_simi = self.ln22(x_simi)
        x = self.t1act(x)
        x_simi = self.t1act(x_simi)
        if self.is_downshuffle:
            x_windows = window_partition_downshuffle(x, self.window_size)  # nW*B, window_size, window_size, C
            x_windows_simi = window_partition_downshuffle(x_simi, self.window_size)  # nW*B, window_size, window_size, C
        if not self.is_downshuffle:
            x_windows = window_partition_normal(x, self.window_size)  # nW*B, window_size, window_size, C
            x_windows_simi = window_partition_normal(x_simi, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows + self.pos_embedding
        x_windows_simi = x_windows_simi + self.pos_embedding_simi
        # W-MSA/dW-MSA
        x = self.drop_path(self.attn(x_windows,x_windows_simi)) + x_windows  # nW*B, window_size*window_size, C   #mutil attention
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if  self.is_downshuffle:
            x = window_reverse_downshuffle(x, self.window_size, self.H, self.W)  # B H' W' C
        if not self.is_downshuffle:
            x = window_reverse_normal(x, self.window_size, self.H, self.W)  # B H' W' C
        return x
        
class SwinTransformer(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=5, 
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.swin_normal=SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, is_downshuffle=False)
        self.swin_downshuffle=SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, is_downshuffle=True)

    def forward(self, x):
        x = self.swin_normal(x)
        x = self.swin_downshuffle(x)
        
        return x

if __name__ == "__main__":
    net=SwinTransformer(dim=32, input_resolution=(80,80), num_heads=4,)
    a=torch.randn(4,32,80,80)
    out=net(a)
    print(out.shape)
