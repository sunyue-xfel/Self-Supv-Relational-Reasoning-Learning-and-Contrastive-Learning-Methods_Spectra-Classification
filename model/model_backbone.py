# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import SplitBatchNorm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)


class linear_classifier(torch.nn.Module):
    def __init__(self, method ,feature_size, nb_class):  #opt.linear_layer
        super(linear_classifier, self).__init__()
        self.feature_size = feature_size
        if method == 'single':
            self.linear_layer = torch.nn.Linear(self.feature_size, nb_class)
        else:
            self.linear_layer = torch.nn.Sequential(
                    torch.nn.Linear(self.feature_size,64),
                    torch.nn.BatchNorm1d(64),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(64, nb_class) )
        self._create_weights()  
        
#     def _create_weights(self, mean=0.0, std=0.05):
#         for module in self.modules():
#             if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
#                 nn.init.orthogonal(module.weight)
#             if isinstance(module, nn.BatchNorm1d):
#                 module.weight.data.normal_(mean, std) 
                
    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std) 
            if isinstance(module, nn.BatchNorm1d):
                module.weight.data.normal_(mean, std) 

    def forward(self, x):
        cout = self.linear_layer(x)
        return cout


class SimConv4(torch.nn.Module):
    def __init__(self, feature_size=64):
        super(SimConv4, self).__init__()
        self.feature_size = feature_size
        self.name = "conv4"

        self.layer1 = torch.nn.Sequential(
            nn.Conv1d(1, 8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(8),
          torch.nn.ReLU()
        )

        self.layer2 = torch.nn.Sequential(
            nn.Conv1d(8, 16, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(16),
          torch.nn.ReLU(),
        )

        self.layer3 = torch.nn.Sequential(
            nn.Conv1d(16, 32, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(32),
          torch.nn.ReLU(),
        )

        self.layer4 = torch.nn.Sequential(
            nn.Conv1d(32, 64, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(64),
          torch.nn.ReLU(),
          torch.nn.AdaptiveAvgPool1d(1)
        )

        self.flatten = torch.nn.Flatten()

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data)
            #        nn.init.xavier_normal_(m.bias.data)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        ############################################################
#         x_ = x.view(x.shape[0], 1, -1)

        x = torch.squeeze(x,-1)
#         print('before conv:',x.shape)
        x_ = torch.unsqueeze(x,1)
        ############################################################
#         print('conv input shape:',x_.shape)
        h = self.layer1(x_)  # (B, 1, D)->(B, 8, D/2)
        h = self.layer2(h)  # (B, 8, D/2)->(B, 16, D/4)
        h = self.layer3(h)  # (B, 16, D/4)->(B, 32, D/8)
#         print('conv3 shape:',h.shape)
        h = self.layer4(h)  # (B, 32, D/8)->(B, 64, 1)
#         print('conv4 shape:',h.shape)
        h = self.flatten(h)
#         print('backbone output shape:',h.shape)
        h = F.normalize(h, dim=1)
        return h

    
    
    

    
    
    
    
###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################

import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv1d, ReLU,AdaptiveMaxPool1d, AdaptiveAvgPool1d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool1d, MaxPool1d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
torch_ver = torch.__version__[:3]

__all__ = ['PAM_Module', 'CAM_Module']


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self):
        super(CAM_Module, self).__init__()
#         self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X N)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, N= x.size()
        proj_query = x
        proj_key = x.permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x

        out = torch.bmm(attention, proj_value)
#         out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out, attention


    
    
class Print(nn.Module):
    def forward(self, x):
        print(x.size())
        return x

    
import math
import torch
import torch.nn as nn
import torch.nn.functional as F





class CovSC_FdFwd(nn.Module):
    """backbone + projection head"""
    def __init__(self, opt, head='FdFwd'):
        super(CovSC_FdFwd, self).__init__()
#         model_fun, dim_in = model_dict[name]

        OUT_Dim =1  
        O_FeatCOV=1
        O_CHENNEL = 16
        O_EmbDim = opt.feature_size
    
        dim_in = opt.feature_size
        
        self.encoder =  ConvSC_NoFdfwd( 
            O_channel  = O_CHENNEL,
            output_dim = OUT_Dim,
            hidden_size = 150,
            O_feature = O_FeatCOV,
            embed_dim = O_EmbDim,
            num_heads = 2
        )
        
        if head == 'FdFwd':
            self.head = FdFwd(opt)
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
  
        #################################
        self._create_weights()  
    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std) 
            if isinstance(module, nn.BatchNorm1d):
                module.weight.data.normal_(mean, std) 
         #################################
        
    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat

    
    
    

class ResMLP(nn.Module):
    """ Channel attention module"""
    def __init__(self,opt):
        super(ResMLP, self).__init__()
        
        embed_dim = opt.feature_size
        dim_feedforward = opt.Dim_FdFwd
    
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)
        self.activation = nn.ReLU()
            
        self._create_weights()
        
    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std) 
            if isinstance(module, nn.BatchNorm1d):
                module.weight.data.normal_(mean, std) 
    
    def forward(self,x):   
#         src = self.norm1(x)
        src = x
        src2 = self.linear2(self.activation(self.linear1(src)))
        src = src + src2
#         src = self.norm2(src)
        In_Conv  = src.view(src.size(0), -1)     
        return In_Conv

    
    
class FdFwd(Module):
    """ Channel attention module"""
    def __init__(self,opt):
        super(FdFwd, self).__init__()
        
        embed_dim = opt.feature_size
        dim_feedforward = opt.Dim_FdFwd
        
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.activation = nn.ReLU()  
         
        self._create_weights()     

    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std) 
            if isinstance(module, nn.BatchNorm1d):
                module.weight.data.normal_(mean, std) 
    def forward(self,x):   
        src = self.norm1(x)
        src2 = self.linear2(self.activation(self.linear1(src)))
        src = src + src2
#         src = self.norm2(src)
        In_Conv  = src.view(src.size(0), -1)     
        return In_Conv







N = 4020
AdptPool = 236
class ConvSC(nn.Module):
    def __init__(self, n_feature=64,  O_channel=8, O_feature=1,output_dim = 1,dim_feedforward =1024,  hidden_size = 150, embed_dim =256, num_heads = 4):
        
        super(ConvSC,self).__init__()

        # weighted attention fusing
#         self.alpha1 = Parameter(torch.zeros(1))
#         self.alpha2 = Parameter(torch.zeros(1))
        
        self.conv1 = nn.Sequential(nn.Conv1d(1, n_feature, kernel_size=5,stride=2),
                                   nn.BatchNorm1d(n_feature),
                                   nn.ReLU(),
                                   nn.MaxPool1d(2)
                                   )
        
        
#         self.conv2 = nn.Sequential(nn.Conv1d(n_feature, n_feature, kernel_size=3,stride=2),
#                                    nn.ReLU(),
#                                    nn.MaxPool1d(2)
#                                    )

#         self.conv3 = nn.Sequential(nn.Conv1d(n_feature, O_channel, kernel_size=5),
#                                    nn.BatchNorm1d(O_channel),
#                                    nn.ReLU(),
#                                    nn.MaxPool1d(2)
#                                    )
    ################################new change##################################
        self.conv3 = nn.Sequential(nn.Conv1d(n_feature, O_channel, kernel_size=5),
                                   nn.BatchNorm1d(O_channel),
                                   nn.ReLU(),
                                   nn.MaxPool1d(2),
                                   nn.AdaptiveAvgPool1d(AdptPool)
                                   )
    
    ####################################################################
        self.dec_conv3 = nn.Sequential(nn.Conv1d(O_channel, 1, kernel_size=1),
                                   nn.BatchNorm1d(1),
                                   nn.ReLU(),
#                                    nn.MaxPool1d(2)
                                   )
        input_shape_emb = (1, 1, N) # just for test, in order to get the shape of FNN input neurons            
        self.output_dimension_emb = self._get_conv_output_emb(input_shape_emb)
        
        input_shape_fc = (1, O_channel, embed_dim) 
        self.output_dimension_fc = self._get_conv_output_fc(input_shape_fc)
        
        
        self.embeding = nn.Sequential( 
#             nn.Linear(self.output_dimension_emb, embed_dim),
            nn.Linear(AdptPool, embed_dim),
            nn.LayerNorm(embed_dim)
        )
                                      
        self.MultiHeadAttention = nn.MultiheadAttention(O_channel, num_heads)
        self.C_Attention_F = CAM_Module().to(device)
        
        self.LayerNorm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Sequential(
            nn.Linear(self.output_dimension_fc*1, O_feature),
            nn.Sigmoid()
        )
    
    
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.activation = nn.ReLU()
      
    
    
#         self.fc1 = nn.Sequential(
#             nn.Linear(self.output_dimension, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, O_feature),            
#             nn.Sigmoid()
# #             nn.Dropout(0.5)
#         )
            
        self._create_weights()
        


    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std) 
            if isinstance(module, nn.BatchNorm1d):
                module.weight.data.normal_(mean, std) 
#         initrange = 0.1
#         nn.init.uniform_(self.encoder.weight, -initrange, initrange)

                
#         for m in self.modules():
#             if isinstance(m, torch.nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, torch.nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             if isinstance(m, nn.Conv1d):
#                 nn.init.xavier_normal_(m.weight.data)
#             #        nn.init.xavier_normal_(m.bias.data)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
                
                
#     def _get_conv_output_emb(self, shape):
    def _get_conv_output_emb(self, x):
        x = torch.rand(x)
#         x = torch.rand(shape)
        x = self.conv1(x)
        x = self.conv3(x)
        output_dimension_emb = x.shape[-1]
        return output_dimension_emb 


    def _get_conv_output_fc(self, shape):
        x = torch.rand(shape)
        x = self.dec_conv3(x)
        output_dimension_fc = x.shape[-1]
        return output_dimension_fc 
    
    def forward(self, x):
        In_Conv = torch.squeeze(x,-1)
        In_Conv = torch.unsqueeze(In_Conv,1)
    
#         print('Conv SC model input shape:',x.shape)
#         In_Conv = x
#         In_Conv = torch.reshape(In_Conv,(-1,1,N))
        
        In_Conv  = self.conv1(In_Conv)         
#         In_Conv  = self.conv2(In_Conv)
        In_Conv  = self.conv3(In_Conv)
#         print('Conv output:',In_Conv.shape)
        
        In_Conv_Att  = self.embeding(In_Conv)
#         print('embeding output/In_Conv_Att:',In_Conv_Att.shape)  # BATCH, C, Spatial size
        In_Conv_Att_S = In_Conv_Att.permute(2, 0, 1)  #(L, N, E) (L,N,E) where L is the target sequence length, N is the batch size, E is the embedding dimension.
#         print('In_Conv_Att_S:',In_Conv_Att_S.shape)
        
        ### attn_output: (L, N, E) where L is the target sequence length, N is the batch size, E is the embedding dimension.
        ### attn_output_weights: (N, L, S) where N is the batch size, L is the target sequence length, S is the source sequence length.
        In_Att_S,attention_S = self.MultiHeadAttention(In_Conv_Att_S,In_Conv_Att_S,In_Conv_Att_S)
        In_Att_C,attention_C = self.C_Attention_F(In_Conv_Att)
#         print('In_Att_C,attention_C:',In_Att_C.shape,attention_C.shape)
#         print('In_Att_S,attention_S:',In_Att_S.shape,attention_S.shape)
        
        # residual connection and feature fusion:
        In_Conv  = (In_Conv_Att+In_Att_S.permute(1,2,0))  +In_Att_C   
        
#         print('Spatial attention output shape:',In_Conv.shape)
        
#         In_Conv  = (In_Conv_Att+In_Att_S) +In_Att_C         
#         In_Conv  = self.alpha1*(In_Conv_Att+In_Att_S) + self.alpha1*In_Att_C 
        
        In_Conv  = self.LayerNorm(In_Conv)
        In_Conv  = self.dec_conv3(In_Conv)
#         print('dec_conv3:',In_Conv.shape)
        
         
        src = self.norm1(In_Conv)
        src2 = self.linear2(self.activation(self.linear1(src)))
        src = src + src2
        src = self.norm2(src)
               

#         In_Conv  = src.view(src.size(0), -1)        
        
        In_Conv  = In_Conv.view(In_Conv.size(0), -1)


#         print('dec_conv3, output_dimension_fc:',In_Conv.shape, self.output_dimension_fc, self.output_dimension_emb)
        
#         In_Conv  = self.fc1(In_Conv)
#         return In_Conv, attention_S, attention_C, In_Conv_Att
        return In_Conv





    
    

# N = 4020
# AdptPool = 236
# class ConvSC_NoFdfwd(nn.Module):
#     def __init__(self, n_feature=64,  O_channel=8, O_feature=1,output_dim = 1,dim_feedforward =1024,  hidden_size = 150, embed_dim =256, num_heads = 4):
        
#         super(ConvSC_NoFdfwd,self).__init__()

#         # weighted attention fusing
# #         self.alpha1 = Parameter(torch.zeros(1))
# #         self.alpha2 = Parameter(torch.zeros(1))
        
#         self.conv1 = nn.Sequential(nn.Conv1d(1, n_feature, kernel_size=5,stride=2),
#                                    nn.BatchNorm1d(n_feature),
#                                    nn.ReLU(),
#                                    nn.MaxPool1d(2)
#                                    )
        
        
# #         self.conv2 = nn.Sequential(nn.Conv1d(n_feature, n_feature, kernel_size=3,stride=2),
# #                                    nn.ReLU(),
# #                                    nn.MaxPool1d(2)
# #                                    )

# #         self.conv3 = nn.Sequential(nn.Conv1d(n_feature, O_channel, kernel_size=5),
# #                                    nn.BatchNorm1d(O_channel),
# #                                    nn.ReLU(),
# #                                    nn.MaxPool1d(2)
# #                                    )
#     ################################new change##################################
#         self.conv3 = nn.Sequential(nn.Conv1d(n_feature, O_channel, kernel_size=5),
#                                    nn.BatchNorm1d(O_channel),
#                                    nn.ReLU(),
#                                    nn.MaxPool1d(2),
#                                    nn.AdaptiveAvgPool1d(AdptPool)
#                                    )
    
#     ####################################################################
#         self.dec_conv3 = nn.Sequential(nn.Conv1d(O_channel, 1, kernel_size=1),
#                                    nn.BatchNorm1d(1),
#                                    nn.ReLU(),
# #                                    nn.MaxPool1d(2)
#                                    )
#         input_shape_emb = (1, 1, N) # just for test, in order to get the shape of FNN input neurons            
#         self.output_dimension_emb = self._get_conv_output_emb(input_shape_emb)
        
#         input_shape_fc = (1, O_channel, embed_dim) 
#         self.output_dimension_fc = self._get_conv_output_fc(input_shape_fc)
        
        
#         self.embeding = nn.Sequential( 
# #             nn.Linear(self.output_dimension_emb, embed_dim),
#             nn.Linear(AdptPool, embed_dim),
#             nn.LayerNorm(embed_dim)
#         )
                                      
#         self.MultiHeadAttention = nn.MultiheadAttention(O_channel, num_heads)
#         self.C_Attention_F = CAM_Module().to(device)
        
#         self.LayerNorm = nn.LayerNorm(embed_dim)

            
#         self._create_weights()
        


#     def _create_weights(self, mean=0.0, std=0.05):
#         for module in self.modules():
#             if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
#                 module.weight.data.normal_(mean, std) 
#             if isinstance(module, nn.BatchNorm1d):
#                 module.weight.data.normal_(mean, std) 
# #         initrange = 0.1
# #         nn.init.uniform_(self.encoder.weight, -initrange, initrange)              
                
# #     def _get_conv_output_emb(self, shape):
#     def _get_conv_output_emb(self, x):
#         x = torch.rand(x)
# #         x = torch.rand(shape)
#         x = self.conv1(x)
#         x = self.conv3(x)
#         output_dimension_emb = x.shape[-1]
#         return output_dimension_emb 


#     def _get_conv_output_fc(self, shape):
#         x = torch.rand(shape)
#         x = self.dec_conv3(x)
#         output_dimension_fc = x.shape[-1]
#         return output_dimension_fc 
    
#     def forward(self, x):
#         In_Conv = torch.squeeze(x,-1)
#         In_Conv = torch.unsqueeze(In_Conv,1)
    
        
#         In_Conv  = self.conv1(In_Conv)         
# #         In_Conv  = self.conv2(In_Conv)
#         In_Conv  = self.conv3(In_Conv)
# #         print('Conv output:',In_Conv.shape)
        
#         In_Conv_Att  = self.embeding(In_Conv)
# #         print('embeding output/In_Conv_Att:',In_Conv_Att.shape)  # BATCH, C, Spatial size
#         In_Conv_Att_S = In_Conv_Att.permute(2, 0, 1)  #(L, N, E) (L,N,E) where L is the target sequence length, N is the batch size, E is the embedding dimension.
# #         print('In_Conv_Att_S:',In_Conv_Att_S.shape)
        
#         ### attn_output: (L, N, E) where L is the target sequence length, N is the batch size, E is the embedding dimension.
#         ### attn_output_weights: (N, L, S) where N is the batch size, L is the target sequence length, S is the source sequence length.
#         In_Att_S,attention_S = self.MultiHeadAttention(In_Conv_Att_S,In_Conv_Att_S,In_Conv_Att_S)
#         In_Att_C,attention_C = self.C_Attention_F(In_Conv_Att)
# #         print('In_Att_C,attention_C:',In_Att_C.shape,attention_C.shape)
# #         print('In_Att_S,attention_S:',In_Att_S.shape,attention_S.shape)
        
#         # residual connection and feature fusion:
#         In_Conv  = (In_Conv_Att+In_Att_S.permute(1,2,0))  +In_Att_C   
        
# #         print('Spatial attention output shape:',In_Conv.shape)
        
# #         In_Conv  = (In_Conv_Att+In_Att_S) +In_Att_C         
# #         In_Conv  = self.alpha1*(In_Conv_Att+In_Att_S) + self.alpha1*In_Att_C 
        
#         In_Conv  = self.LayerNorm(In_Conv)
#         In_Conv  = self.dec_conv3(In_Conv)
# #         print('dec_conv3:',In_Conv.shape)
              
        
#         In_Conv  = In_Conv.view(In_Conv.size(0), -1)


# #         print('dec_conv3, output_dimension_fc:',In_Conv.shape, self.output_dimension_fc, self.output_dimension_emb)
        
# #         In_Conv  = self.fc1(In_Conv)
# #         return In_Conv, attention_S, attention_C, In_Conv_Att
#         return In_Conv





    

from typing import Type, Any, Callable, Union, List, Optional
   

N = 4020
AdptPool = 236
class ConvSC_NoFdfwd(nn.Module): #ConvSC_NoFdfwd
    def __init__(self, n_feature=64,  O_channel=8, O_feature=1,output_dim = 1,dim_feedforward =1024,  hidden_size = 150, embed_dim =256, 
                 num_heads = 4,Sim_Batch = 256,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
      
        super(ConvSC_NoFdfwd,self).__init__()

        
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
            
        self.conv1 = nn.Sequential(nn.Conv1d(1, n_feature, kernel_size=5,stride=2),
#                                    nn.BatchNorm1d(n_feature),
                                   norm_layer(n_feature),
                                   nn.ReLU(),
                                   nn.MaxPool1d(2)
                                   )
 
        self.conv3 = nn.Sequential(nn.Conv1d(n_feature, O_channel, kernel_size=5),
#                                    nn.BatchNorm1d(O_channel),
                                   norm_layer(O_channel),
                                   nn.ReLU(),
                                   nn.MaxPool1d(2),
                                   nn.AdaptiveAvgPool1d(AdptPool)
                                   )

        self.dec_conv3 = nn.Sequential(nn.Conv1d(O_channel, 1, kernel_size=1),
#                                    nn.BatchNorm1d(1),
                                   norm_layer(1),
                                   nn.ReLU(),
#                                    nn.MaxPool1d(2)
                                   )
        input_shape_emb = (Sim_Batch, 1, N) # just for test, in order to get the shape of FNN input neurons            
        self.output_dimension_emb = self._get_conv_output_emb(input_shape_emb)
        
        input_shape_fc = (Sim_Batch, O_channel, embed_dim) 
        self.output_dimension_fc = self._get_conv_output_fc(input_shape_fc)
        
        
        self.embeding = nn.Sequential( 
#             nn.Linear(self.output_dimension_emb, embed_dim),
            nn.Linear(AdptPool, embed_dim),
            nn.LayerNorm(embed_dim)
        )
                                      
        self.MultiHeadAttention = nn.MultiheadAttention(O_channel, num_heads)
        self.C_Attention_F = CAM_Module().to(device)
        
        self.LayerNorm = nn.LayerNorm(embed_dim)
            
        self._create_weights()
        


    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std) 
            if isinstance(module, nn.BatchNorm1d) or isinstance(module,SplitBatchNorm):
                module.weight.data.normal_(mean, std) 
                
#     def _create_weights(self, mean=0.0, std=0.05):
#         for module in self.modules():
#             if isinstance(module, (nn.Conv1d, nn.Linear)):
#                 nn.init.orthogonal(module.weight)
#             if isinstance(module, nn.BatchNorm1d):
#                 module.weight.data.normal_(mean, std) 
        #################################        
 
                
                
#     def _get_conv_output_emb(self, shape):
    def _get_conv_output_emb(self, x):
        x = torch.rand(x)
#         x = torch.rand(shape)
        x = self.conv1(x)
        x = self.conv3(x)
        output_dimension_emb = x.shape[-1]
        return output_dimension_emb 


    def _get_conv_output_fc(self, shape):
        x = torch.rand(shape)
        x = self.dec_conv3(x)
        output_dimension_fc = x.shape[-1]
        return output_dimension_fc 
    
    def forward(self, x):
        In_Conv = torch.squeeze(x,-1)
        In_Conv = torch.unsqueeze(In_Conv,1)
        
#         print('In_Conv:',In_Conv.shape)
        
        In_Conv  = self.conv1(In_Conv)         
#         In_Conv  = self.conv2(In_Conv)
        In_Conv  = self.conv3(In_Conv)
#         print('Conv output:',In_Conv.shape)
        
        In_Conv_Att  = self.embeding(In_Conv)
#         print('embeding output/In_Conv_Att:',In_Conv_Att.shape)  # BATCH, C, Spatial size
        In_Conv_Att_S = In_Conv_Att.permute(2, 0, 1)  #(L, N, E) (L,N,E) where L is the target sequence length, N is the batch size, E is the embedding dimension.
#         print('In_Conv_Att_S:',In_Conv_Att_S.shape)
        
        ### attn_output: (L, N, E) where L is the target sequence length, N is the batch size, E is the embedding dimension.
        ### attn_output_weights: (N, L, S) where N is the batch size, L is the target sequence length, S is the source sequence length.
        In_Att_S,attention_S = self.MultiHeadAttention(In_Conv_Att_S,In_Conv_Att_S,In_Conv_Att_S)
        In_Att_C,attention_C = self.C_Attention_F(In_Conv_Att)
#         print('In_Att_C,attention_C:',In_Att_C.shape,attention_C.shape)
#         print('In_Att_S,attention_S:',In_Att_S.shape,attention_S.shape)
        
        # residual connection and feature fusion:
        In_Conv  = (In_Conv_Att+In_Att_S.permute(1,2,0))  +In_Att_C   
        
#         print('Spatial attention output shape:',In_Conv.shape)
      
        In_Conv  = self.LayerNorm(In_Conv)
        In_Conv  = self.dec_conv3(In_Conv)
#         print('dec_conv3:',In_Conv.shape)
        
      
        
        In_Conv  = In_Conv.view(In_Conv.size(0), -1)


#         print('dec_conv3, output_dimension_fc:',In_Conv.shape, self.output_dimension_fc, self.output_dimension_emb)
        
#         In_Conv  = self.fc1(In_Conv)
#         return In_Conv, attention_S, attention_C, In_Conv_Att
        return In_Conv


   


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)

    
######################################Relation Reasoning modules in RltRsn model###########################
class relation_head(nn.Module):
    """backbone + projection head"""
    def __init__(self, feature_size, hidfeatsize):
        super(relation_head, self).__init__()
        self.hidfeatsize = hidfeatsize
        self.head =  torch.nn.Sequential(
                             torch.nn.Linear(feature_size*2, self.hidfeatsize),
                             torch.nn.BatchNorm1d(self.hidfeatsize),
                             torch.nn.LeakyReLU(),
                             torch.nn.Linear(self.hidfeatsize, 1)  )

        self._create_weights()  
    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
#                 nn.init.orthogonal(module.weight)
                module.weight.data.normal_(mean, std) 
            if isinstance(module, nn.BatchNorm1d):
                module.weight.data.normal_(mean, std) 

    def forward(self, x):
        feat = self.head(x)
        return feat    


    
class cls_head(nn.Module):
    """backbone + projection head"""
    def __init__(self, feature_size,hidfeatsize, nb_class):
        super(cls_head, self).__init__()
        self.hidfeatsize = hidfeatsize
        self.head = torch.nn.Sequential(
            torch.nn.Linear(feature_size*2, self.hidfeatsize),
            torch.nn.BatchNorm1d(self.hidfeatsize),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.hidfeatsize, nb_class),
            torch.nn.Softmax(),
        )

        self._create_weights()  
    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
#                 nn.init.orthogonal(module.weight)
                module.weight.data.normal_(mean, std) 
            if isinstance(module, nn.BatchNorm1d):
                module.weight.data.normal_(mean, std) 

    def forward(self, x):
        feat = self.head(x)
        return feat    


            
            
class tmp_cls_head(nn.Module):
    """backbone + projection head"""
    def __init__(self, feature_size,hidfeatsize, tmp_C):
        super(tmp_cls_head, self).__init__()
        self.hidfeatsize = hidfeatsize
        self.head = torch.nn.Sequential(
#             torch.nn.Linear(feature_size, 128),
#             torch.nn.BatchNorm1d(128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(feature_size, tmp_C), #128
            torch.nn.Softmax(), 
        )

        self._create_weights()  
    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
#                 nn.init.orthogonal(module.weight)
                module.weight.data.normal_(mean, std) 
            if isinstance(module, nn.BatchNorm1d):
                module.weight.data.normal_(mean, std) 

    def forward(self, x):
        feat = self.head(x)
        return feat        
    
    
# for m in model.modules():
#     if isinstance(m, (nn.Conv1d, nn.Linear)):
#         nn.init.orthogonal(m.weight)
