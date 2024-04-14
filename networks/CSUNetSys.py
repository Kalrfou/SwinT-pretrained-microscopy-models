# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import logging
import math
from os.path import join as pjoin
import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
import torch.utils.checkpoint as checkpoint
from einops import rearrange,repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from .common import PatchEmbed
from .swin_encoder import Encoder
from .cnn_encoder import CNN_Encoder
from segmentation_models_pytorch.encoders import get_encoder
import torch.utils.model_zoo as model_zoo
from .CSUnetDecoder import CS_Decoder
logger = logging.getLogger(__name__)

class CSUnetSys(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False,remlp=1,cnn='efficientnet-b4',weights='imagenet', final_upsample="expand_first", **kwargs):
        super().__init__()
        print("CSUnetSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(depths,
        depths_decoder,drop_path_rate,num_classes))
   
        self.patch_norm= patch_norm
        #Embedding Part
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        
        #Swin encoder
        self.encoder=Encoder(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                 embed_dim=embed_dim, depths=depths, depths_decoder=depths_decoder, num_heads=num_heads,
                 window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None,
                 drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False,patch_embed=self.patch_embed, **kwargs)
        #CNN Encoder
        #print(cnn,weights)
        self.cnn= CNN_Encoder(CNN_fam=cnn,pre_train=weights)
        
        self.ch,self.inver=self.cnn.get_channels()
        #print("The chanelles are "+str(self.inver))
         #Build decoder
        self.decoder = CS_Decoder(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                 embed_dim=embed_dim, depths=depths, depths_decoder=depths_decoder, num_heads=num_heads,
                 window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None,
                 drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False,final_upsample="expand_first",chan=self.ch,inver_chan=self.inver,patch_embed=self.patch_embed, **kwargs)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        encoder_cnn=self.cnn(x)
        x, x_downsample = self.encoder(x)
        
        x = self.decoder(x,x_downsample,encoder_cnn)
        return x
    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
        