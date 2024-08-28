import torch.nn as nn
import torch
import torch.nn.functional as F
import Config as config
from functools import partial
from timm.models.layers import DropPath


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class CCA(nn.Module):
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # channel-wise attention
        avg_pool_x = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d( g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g)/2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out


class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, skip_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.coatt = CCA(F_g=in_channels, F_x=skip_channels)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.coatt(g=up, x=skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return x


class FCUDown(nn.Module):

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = norm_layer(outplanes)
        self.act = act_layer()

        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, x_t):
        x = self.conv_project(x)  # [N, C, H, W]

        x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = self.ln(x)
        x = self.act(x)

        if x_t is not None:
            x_t_avg = self.avgpool(x_t.transpose(1,2)).transpose(1,2)
            x = torch.cat([x_t_avg, x], dim=1)

        return x


class FCUUp(nn.Module):

    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6),):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, _, C = x.shape
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))
        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


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


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvTransBlock(nn.Module):
    def __init__(self, inplanes, outplanes, nb_Conv, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):

        super(ConvTransBlock, self).__init__()
        self.cnn_block = _make_nConv(inplanes, outplanes, nb_Conv)
        self.fusion_block = ConvBatchNorm(outplanes, outplanes)

        self.squeeze_block = FCUDown(inplanes=outplanes, outplanes=embed_dim, dw_stride=dw_stride)

        self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes, up_stride=dw_stride)

        self.trans_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim

    def forward(self, x, x_t):
        x = self.cnn_block(x)

        _, _, H, W = x.shape

        x_st = self.squeeze_block(x, x_t)

        x_t = self.trans_block(x_st + x_t)

        x_t_r = self.expand_block(x_t, H // self.dw_stride, W // self.dw_stride)
        x = self.fusion_block(x + x_t_r)

        return x, x_t


class TACUNet(nn.Module):
    def __init__(self, config,n_channels=3, n_classes=1,img_size=224,vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv=2)
        self.down2 = DownBlock(in_channels*2, in_channels*4, nb_Conv=2)
        self.down3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=2)
        self.down4 = DownBlock(in_channels*8, in_channels*8, nb_Conv=2)

        self.embed_dim = config.trans_embed_dim

        self.trans_patch_conv = nn.Conv2d(in_channels, self.embed_dim, kernel_size=(16,16), stride=(16, 16), padding=0)
        self.trans_1 = Block(dim=self.embed_dim, num_heads=config.trans_num_heads, mlp_ratio=config.trans_mpl_ratio)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.MaxPool2d(2)

        # stage 2
        self.ctb1 = ConvTransBlock(in_channels, in_channels*2, 2, dw_stride=config.patch_sizes[0], embed_dim=self.embed_dim,
                        num_heads=config.trans_num_heads, mlp_ratio=config.trans_mpl_ratio)

        self.ctb2 = ConvTransBlock(in_channels*2, in_channels*4, 2, dw_stride=config.patch_sizes[1],
                                   embed_dim=self.embed_dim,
                                   num_heads=config.trans_num_heads, mlp_ratio=config.trans_mpl_ratio)

        self.ctb3 = ConvTransBlock(in_channels*4, in_channels*8, 2, dw_stride=config.patch_sizes[2],
                                   embed_dim=self.embed_dim,
                                   num_heads=config.trans_num_heads, mlp_ratio=config.trans_mpl_ratio)

        self.ctb4 = ConvTransBlock(in_channels*8, in_channels*8, 2, dw_stride=config.patch_sizes[3],
                                   embed_dim=self.embed_dim,
                                   num_heads=config.trans_num_heads, mlp_ratio=config.trans_mpl_ratio)

        # decoder--------------------------------------------------------------------------------------
        self.up4 = UpBlock_attention(in_channels*8, in_channels*8)
        self.up4_ctb = ConvTransBlock(in_channels * 16, in_channels * 4, 2, dw_stride=config.patch_sizes[2],
                                   embed_dim=self.embed_dim,
                                   num_heads=config.trans_num_heads, mlp_ratio=config.trans_mpl_ratio)

        self.up3 = UpBlock_attention(in_channels*4, in_channels*4)
        self.up3_ctb = ConvTransBlock(in_channels * 8, in_channels * 2, 2, dw_stride=config.patch_sizes[1],
                                   embed_dim=self.embed_dim,
                                   num_heads=config.trans_num_heads, mlp_ratio=config.trans_mpl_ratio)

        self.up2 = UpBlock_attention(in_channels*2, in_channels*2)
        self.up2_ctb = ConvTransBlock(in_channels * 4, in_channels, 2, dw_stride=config.patch_sizes[0],
                                   embed_dim=self.embed_dim,
                                   num_heads=config.trans_num_heads, mlp_ratio=config.trans_mpl_ratio)

        self.up1 = UpBlock_attention(in_channels, in_channels)
        self.last_fuse = _make_nConv(in_channels*2, in_channels, 2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1), stride=(1,1))
        self.last_activation = nn.Sigmoid() # if using BCELoss

    def forward(self, x):
        x = x.float()
        x1 = self.inc(x)

        # init for cnn-transformers
        x_p = self.trans_patch_conv(x1).flatten(2).transpose(1, 2)  # [1, 197, 768]
        x_avg = self.avgpool(x_p.transpose(1, 2)).transpose(1,2)
        x_t_init = self.trans_1(torch.cat([x_p, x_avg], dim=1))  # [1, 197, 768]

        # fuse-1 for cnn-transformers
        x2 = self.maxpool(x1)
        x2_cn, x2_ct = self.ctb1(x2, x_t_init)

        x3 = self.maxpool(x2_cn)
        x3_cn, x3_ct = self.ctb2(x3, x2_ct)

        x4 = self.maxpool(x3_cn)
        x4_cn, x4_ct = self.ctb3(x4, x3_ct)

        x5 = self.maxpool(x4_cn)
        x5_cn, x5_ct = self.ctb4(x5, x4_ct)

        # decoder for cnn-transformer
        x4_up = self.up4(x5_cn, x4_cn)
        x4_up, x4_upct = self.up4_ctb(x4_up, x5_ct)

        x3_up = self.up3(x4_up, x3_cn)
        x3_up, x3_upct = self.up3_ctb(x3_up, x4_upct)

        x2_up = self.up2(x3_up, x2_cn)
        x2_up, x2_upct = self.up2_ctb(x2_up, x3_upct)

        x = self.up1(x2_up, x1)
        x = self.last_fuse(x)

        logits = self.last_activation(self.outc(x))

        return logits

if __name__ == '__main__':
    config_vit = config.get_config_tacunet()
    model = TACUNet(config_vit)
    x = torch.rand((1, 3, 224, 224))
    y = model(x)
    print(y.shape)
