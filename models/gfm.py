"""
Modified from SPADE (https://arxiv.org/abs/1903.07291)

Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn

# Function to easily switch between GroupNorm and BatchNorm
def create_normalization(norm_type, channels, num_groups=32):
    if norm_type == 'group':
        return nn.GroupNorm(num_groups, channels)
    elif norm_type == 'batch':
        return nn.BatchNorm2d(channels)
    else:
        raise ValueError(f'Unknown normalization type: {norm_type}')


class SAFT(nn.Module):
    '''
    |norm_nc|: the #channels of the normalized activations, hence the output dim of SAFT
    |label_nc|: the #channels of the input semantic map, hence the input dim of SAFT
    '''
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        # kernel size is set to 1 to encode segmentation map
        ks = 1
        pw = ks // 2

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.SiLU(),
            nn.Conv2d(nhidden, nhidden, kernel_size=ks, padding=pw),
            nn.SiLU(),
            nn.Conv2d(nhidden, nhidden, kernel_size=ks, padding=pw),
            nn.SiLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x_normalized, cond):
        # Produce scaling and bias conditioned on semantic map
        actv = self.mlp_shared(cond)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = x_normalized * (1 + gamma) + beta

        return out

class GFM(nn.Module):
    def __init__(self, input_nc, seg_nc, text_nc, norm_type='group'):
        super().__init__()
        # Choose the appropriate normalization (GroupNorm or BatchNorm)
        self.param_free_norm = create_normalization(norm_type, input_nc)

        # Image SFT (SAFT block with Segmentation Mask)
        self.image_sft = SAFT(input_nc, seg_nc)

        # Text SFT (SAFT block with Segmentation-CLIP map)
        self.text_sft = SAFT(input_nc, text_nc)

    def forward(self, input_feat, seg_mask, scm):

        # Normalize the input feature
        input_feat_normalized = self.param_free_norm(input_feat)
        
        # Apply the Image SAFT
        image_sft_out = self.image_sft(input_feat_normalized, seg_mask)

        # Apply the Text SAFT
        text_sft_out = self.text_sft(input_feat_normalized, scm)

        # Sum the outputs from both Image and Text SAFT blocks
        fused_output = image_sft_out + text_sft_out
        
        return fused_output


# create a 3 layer 1x1 convolutional network as encoder
class SCM_encoder(nn.Module):
    def __init__(self, input_nc, output_nc=128):
        super(SCM_encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_nc, input_nc//2, kernel_size=1)
        self.conv2 = nn.Conv2d(input_nc//2, input_nc//4, kernel_size=1)
        self.conv3 = nn.Conv2d(input_nc//4, output_nc, kernel_size=1)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)

        return x

# descripton-CLIP MAP
# 添加了选取前topk个最相似的text token
class DCM_encoder(nn.Module):
    def __init__(self, input_nc=78848, output_nc=1024,
                 d_model=256, d_image=512, d_text=1024,
                 topk=50):
        super().__init__()
        hidden_nc = 4096
        input_nc = topk * d_model
        self.encoder = nn.Sequential(
            nn.Linear(input_nc, hidden_nc),
            nn.ReLU(),
            nn.Linear(hidden_nc, output_nc)  # 最终维度 1024
        )
        self.proj_image = nn.Linear(d_image, d_model)
        self.proj_text = nn.Linear(d_text, d_model)

        self.topk = topk
        self.d_model = d_model
    
    def forward(self, seg_embedding, des_embedding):
        image_proj = self.proj_image(seg_embedding)
        text_proj = self.proj_text(des_embedding)

        res = text_proj @ image_proj.transpose(-1, -2)
        topk_logits = res.max(-1)[0]
        topk_indexes = torch.topk(topk_logits, self.topk, dim=0)[1]

        # 2025.4.25，现在已经跑起来保持原本顺序不变的代码。现在注释掉这里跑一下每保持原本顺序的测试
        # 测试结果出来了，在230000steps之前，没有之前效果好，现在试试保证原始顺序不变的效果
        topk_indexes, _ = torch.sort(topk_indexes, dim=0)  # 保证原始顺序不变

        topk_des_embedding = torch.gather(text_proj, 0, topk_indexes.unsqueeze(-1).repeat(1, self.d_model))

        final_des_embedding = self.encoder(topk_des_embedding.view(-1))
        return final_des_embedding