import logging
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from segmentation_models_pytorch.encoders import get_encoder
import torch.utils.model_zoo as model_zoo

logger = logging.getLogger(__name__)

class CNN_Encoder(nn.Module):
    def __init__(self, CNN_fam='efficientnet-b4', pre_train='imagenet'):
        super().__init__()
        self.cnn_fam=CNN_fam
        self.ccn_pre=pre_train
        self.cnn_encoder= self.get_CNN_Encoder()
    
    def get_CNN_Encoder(self):
        """
        This method takes the CNN type and pre-training type,
        pre-training type (parameter):
            imagenet       ==> ImageNet
            micronet       ==>MicroNet
            image-micronet ==> image->MicroNet           
        create the encoder with specific type of ptr-training weights
        return the encoder and the channels (the output of each layer)
        """
        if self.ccn_pre == 'imagenet':
            encoder = get_encoder(name=self.cnn_fam, in_channels=3, depth=5, weights=self.ccn_pre,)
        else:
            encoder = get_encoder(name=self.cnn_fam, in_channels=3, depth=5, weights=None,)
            url = self.get_pretrained_microscopynet_url(self.cnn_fam, self.ccn_pre)
            encoder.load_state_dict(model_zoo.load_url(url))
        return encoder
    def get_channels(self):
      channels = self.cnn_encoder.out_channels[2:]
      inver=channels[::-1]
      return channels, inver

    def get_pretrained_microscopynet_url(self, encoder, encoder_weights, version=1.1, self_supervision=''):
        """  A PyTorch impl of:  Microstructure segmentation with deep learning encoders pre-trained on a large microscopy dataset
            https://www.nature.com/articles/s41524-022-00878-5
        Get the url to download the specified pretrained encoder.
        Args:
            encoder (str): pretrained encoder model name (e.g. resnet50)
                encoder_weights (str): pretraining dataset, either 'micronet' or
                'imagenet-micronet' with the latter indicating the encoder	
                was first pretrained on imagenet and then finetuned on microscopynet
            version (float): model version to use, defaults to latest.
                  Current options are 1.0 or 1.1.
            self_supervision (str): self-supervision method used. If self-supervision
                was not used set to '' (which is default).
        Returns:
            str: url to download the pretrained model
        """
        # there is an error with the name for resnext101_32x8d so catch and return
        # (currently there is only version 1.0 for this model so don't need to check version.)
        if encoder == 'resnext101_32x8d':
           return 'https://nasa-public-data.s3.amazonaws.com/microscopy_segmentation_models/resnext101_pretrained_microscopynet_v1.0.pth.tar'
        # only resnet50/micronet has version 1.1 so I'm not going to overcomplicate this right now.
        if encoder != 'resnet50' or encoder_weights != 'micronet':
           version = 1.0
        # setup self-supervision
        if self_supervision != '':
           version = 1.0
           self_supervision = '_' + self_supervision
        
        # correct for name change for URL
        if encoder_weights == 'micronet':
            encoder_weights = 'microscopynet'
        elif encoder_weights == 'image-micronet':
            encoder_weights = 'imagenet-microscopynet'
        else:
            raise ValueError("encoder_weights must be 'micronet' or 'image-micronet'")
        
        
        # get url
        url_base = 'https://nasa-public-data.s3.amazonaws.com/microscopy_segmentation_models/'
        url_end = '_v%s.pth.tar' %str(version)
        return url_base + f'{encoder}{self_supervision}_pretrained_{encoder_weights}' + url_end
    
    def forward(self, x):
        x = self.cnn_encoder(x)
        encoder_cnn_list = []
        for i in range(len(x)-2):
            m = x[i+2]
            m = m.flatten(2)
            m = m.transpose(-1, -2)
            encoder_cnn_list.append(m)
        return encoder_cnn_list
