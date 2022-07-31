import torch
from torch import nn as nn
from einops.layers.torch import Rearrange

class Mlp(nn.Module):
    # two mlp, fc-relu-drop-fc-relu-drop
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

class LAM_Module(nn.Module):
    """ Layer attention module"""
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
       # self.adaptive_fc_other = nn.Linear(dim, 1, bias=False)
       # self.adaptive_act=nn.GELU()
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

class CrossTransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio = 1., attn_drop=0., proj_drop=0.,drop_path = 0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,visual=False):
        super(CrossTransformerEncoderLayer, self).__init__()
        self.x_norm1 = norm_layer(dim)
        self.c_norm1 = norm_layer(dim)
        self.attn = MultiHeadCrossAttention(dim, num_heads, attn_drop, proj_drop,visual=visual)
        self.x_norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=proj_drop)
        self.visual = visual
        self.drop1 = nn.Dropout(drop_path)
        self.drop2 = nn.Dropout(drop_path)

    def forward(self, x, complement):
        x = self.x_norm1(x)
        complement = self.c_norm1(complement)
        if not self.visual:
            x = x + self.drop1(self.attn(x, complement))
        else:
            temp,attention,proj_value = self.attn(x, complement)
            x = x + self.drop1(temp)
        x = x + self.drop2(self.mlp(self.x_norm2(x)))
        if not self.visual:
            return x
        else:
            return x,attention,proj_value



class CrossTransformer(nn.Module):
    def __init__(self, x_dim, num_heads, mlp_ratio =1., attn_drop=0., proj_drop=0., drop_path =0.,visual=False):
        super(CrossTransformer, self).__init__()

        self.cross_att=CrossTransformerEncoderLayer(x_dim, num_heads, mlp_ratio, attn_drop, proj_drop, drop_path,visual=visual)  
        self.visual = visual
  
    def forward(self, x, complement):
        if not self.visual:
            x =  self.cross_att(x, complement) + x
            return x
        else:
             temp,attention,proj_value = self.cross_att(x, complement)
             x = temp+x
             return x,attention,proj_value

# add by YunluYan
INPUT_SIZE = 80
SCALE = 3
INPUT_DIM = 32   # the channel of input
OUTPUT_DIM = 32   # the channel of output
HEAD_HIDDEN_DIM = 32  # the hidden dim of Head
TRANSFORMER_DEPTH = 1  # the depth of the transformer
TRANSFORMER_NUM_HEADS = 4 # the head's num of multi head attention
TRANSFORMER_MLP_RATIO = 2  # the MLP RATIO Of transformer
TRANSFORMER_EMBED_DIM = 4  # the EMBED DIM of transformer
P1 = 5
P2 = 15

class CrossCMMT(nn.Module):

    def __init__(self,visual=False):
        super(CrossCMMT, self).__init__()
        x_patch_dim = HEAD_HIDDEN_DIM * P1 ** 2#800
        x_num_patches = (INPUT_SIZE // P1) ** 2#144
        complement_patch_dim = HEAD_HIDDEN_DIM * P2 ** 2#12800
        complement_num_patches = (INPUT_SIZE * SCALE // P2) ** 2#144
        self.con1 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
        self.con4 = nn.Conv2d(32, 32, kernel_size=3, stride=3, padding=0)
        self.x_patch_embbeding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=P1,
                      p2=P1),
        )#b,144,800
        self.complement_patch_embbeding = nn.Sequential(
            Rearrange('b c (h1 p1) (w1 p2) -> b (h1 w1) (p1 p2 c)', p1=P2,
                      p2=P2),
        )#b,144,12800
        self.t1act = nn.GELU()
        self.x_pos_embedding = nn.Parameter(torch.ones(1, x_num_patches, x_patch_dim))#1,144,800
        self.complement_pos_embedding = nn.Parameter(torch.ones(1, complement_num_patches, complement_patch_dim//(SCALE * SCALE)))#1,144,12800
        self.cross_transformer = CrossTransformer(x_patch_dim,
                                                  TRANSFORMER_NUM_HEADS,
                                                  TRANSFORMER_MLP_RATIO,visual=visual)
        self.p1 = P1
        self.p2 = P2
        self.visual = visual
        self.ln11 = nn.LayerNorm([80,80])
        self.ln12 = nn.LayerNorm([80,80])
        self.ln21 = nn.LayerNorm([240,240])
        self.ln22 = nn.LayerNorm([80,80])
        self.conv_end = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
    def forward(self, x, complement):
        x_copy = x.clone()
        b, _, h, w = x.shape
        _, _, c_h, c_w = complement.shape
        #1*1conv   #4*4conv
        x = self.ln11(x)
        x = self.con1(x)
        x = self.ln12(x)
        complement = self.ln21(complement)
        complement = self.con4(complement)
        complement = self.ln22(complement)
        x = self.t1act(x)
        complement = self.t1act(complement)
        x = self.x_patch_embbeding(x)
        x += self.x_pos_embedding
        complement = self.x_patch_embbeding(complement)#b,144,800#32*20*20
        c_b,c_nums,c_dims = complement.shape
        x = self.t1act(x)
        complement = self.t1act(complement)
        complement += self.complement_pos_embedding
        if not self.visual:
            x = self.cross_transformer(x, complement)
        else:
            x,attention,proj_value = self.cross_transformer(x, complement)
        c = int(x.shape[2] / (self.p1 * self.p1))
        H = int(h / self.p1)
        W = int(w / self.p1)
        x = x.reshape(b, H, W, self.p1, self.p1, c)  # b H W p1 p2 c
        x = x.permute(0, 5, 1, 3, 2, 4)  # b c H p1 W p2
        x = x.reshape(b, -1, h, w, )
        x = x_copy + self.conv_end(x)
        if not self.visual:
            return x
        else:
            return x,attention,proj_value    
                 
class Select(nn.Module):#conv
    def __init__(self,in_nc=32, out_nc=32,kernel_size=1,stride=1,padding=0,visual=False):
        super(Select, self).__init__()
        self.conv = nn.Conv2d(in_nc, 1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act = nn.Sigmoid()
    def forward(self, x, complement):
        x = self.conv(x)
        x = self.act(x)
        out = x * complement
        return out
