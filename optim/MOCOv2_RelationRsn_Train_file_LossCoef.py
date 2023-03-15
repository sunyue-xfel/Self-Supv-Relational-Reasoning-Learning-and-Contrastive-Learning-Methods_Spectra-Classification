# -*- coding: utf-8 -*-
################# Adapted from:####################
###################### https://github.com/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
###################################################   

from datetime import datetime
from functools import partial
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse
import json
import math
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


from typing import Type, Any, Callable, Union, List, Optional

import torch
import utils.transforms as transforms
from dataloader.ucr2018 import UCR2018, MultiUCR2018_InterIntra, MultiUCR2018_InterIntra_Moco
import torch.utils.data as data
from optim.pytorchtools import EarlyStopping
# from model.model_backbone import SimConv4
import utils.transforms as transforms_ts
from model.model_backbone import SimConv4,ConvSC,linear_classifier, ConvSC_NoFdfwd
from model.model_backbone import relation_head,cls_head, tmp_cls_head
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import adjust_learning_rate, warmup_learning_rate
import math
import numpy as np
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from losses import SupConLoss

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

def RelationRsn_MOCOv2_train(x_train, y_train, x_val, y_val, x_test, y_test, nb_class, opt):
    # construct data loader
    # Those are the transformations used in the paper
    prob = 1 # 0.2  # Transform Probability
    cutout = transforms_ts.Cutout(sigma=0.1, p=prob)
#     jitter = transforms_ts.Jitter(sigma=0.2, p=prob)  # CIFAR10    
    jitter = transforms_ts.Jitter(sigma=0.1, p=prob)  # CIFAR10

    scaling = transforms_ts.Scaling(sigma=0.4, p=prob)
    magnitude_warp = transforms_ts.MagnitudeWrap(sigma=0.3, knot=4, p=prob)
    time_warp = transforms_ts.TimeWarp(sigma=0.2, knot=8, p=prob)
    window_slice = transforms_ts.WindowSlice(reduce_ratio=0.8, p=prob)
#     window_warp = transforms_ts.WindowWarp(window_ratio=0.3, scales=(0.5, 2), p=prob)    
    window_warp = transforms_ts.WindowWarp(window_ratio=0.3, scales=(0.7, 1.5), p=prob)


    transforms_list = {'jitter': [jitter],
                       'cutout': [cutout],
                       'scaling': [scaling],
                       'magnitude_warp': [magnitude_warp],
                       'time_warp': [time_warp],
                       'window_slice': [window_slice],
                       'window_warp': [window_warp],
                       'G0': [jitter, magnitude_warp, window_slice],
                       'G1': [jitter, time_warp, window_slice],
                       'G2': [jitter, time_warp, window_slice, window_warp, cutout],
                       'none': []}

    transforms_targets = list()
    for name in opt.aug_type:
        for item in transforms_list[name]:
            transforms_targets.append(item)

#     train_transform = transforms_ts.Compose(transforms_targets + [transforms_ts.ToTensor()])
#     transform_lineval = transforms.Compose([transforms.ToTensor()])

#     train_set_lineval = UCR2018(data=x_train, targets=y_train, transform =TwoCropTransform(train_transform))
#     val_set_lineval = UCR2018(data=x_val, targets=y_val, transform =TwoCropTransform(transform_lineval))
#     test_set_lineval = UCR2018(data=x_test, targets=y_test, transform =TwoCropTransform(transform_lineval))

    ##########################################=my=#######################################
    cut_piece = transforms.CutPiece(sigma = opt.piece_size,typeC = opt.CutPiece_type)
    nb_class = opt.CutPiece_type        
    tmp_C = opt.tmp_C
    #################################################################################

#     transforms_targets = [transforms_list[name] for name in opt.aug_type]
    
    train_transform = transforms.Compose(transforms_targets)
    tensor_transform = transforms.ToTensor()
    
    Moco_train_transform = transforms.Compose(transforms_targets + [transforms_ts.ToTensor()])
    print('Moco_train_transform:',Moco_train_transform)
    train_set_lineval = MultiUCR2018_InterIntra_Moco(data=x_train, targets=y_train, K= opt.K, 
                                        transform=train_transform, transform_cut=cut_piece,
                                        totensor_transform=tensor_transform, moco_transform =TwoCropTransform(Moco_train_transform) )
#     train_set_lineval = MultiUCR2018_InterIntra(data=x_train, targets=y_train, K= opt.K, 
#                                         transform=train_transform, transform_cut=cut_piece,
#                                         totensor_transform=tensor_transform)
    val_set_lineval = MultiUCR2018_InterIntra(data=x_val, targets=y_val, K= opt.K, 
                                        transform=train_transform, transform_cut=cut_piece,
                                        totensor_transform=tensor_transform)
    test_set_lineval = UCR2018(data=x_test, targets=y_test, transform =tensor_transform)
    
    

    train_loader_lineval = torch.utils.data.DataLoader(train_set_lineval, batch_size=opt.batch_size, shuffle=True)
    val_loader_lineval = torch.utils.data.DataLoader(val_set_lineval, batch_size=opt.batch_size, shuffle=False)
    test_loader_lineval = torch.utils.data.DataLoader(test_set_lineval, batch_size=opt.batch_size, shuffle=False)

    
    OUT_Dim =1  
    O_FeatCOV=1
    O_CHENNEL = 16
    O_EmbDim = opt.feature_size

    if opt.backbone == 'RltRsn_Moco_ConvSC':
        backbone_lineval = ModelMoCo(
            dim = opt.moco_dim,
            K = opt.moco_k,
            m = opt.moco_m,
            T = opt.moco_t,
            bn_splits= opt.bn_splits,
            symmetric= opt.symmetric,
            opt = opt,
            nb_class = opt.CutPiece_type, 
            tmp_C = opt.tmp_C,
            feature_size = opt.moco_dim #opt.feature_size
            ).to(device)
        
        model = backbone_lineval#.encoder_q
        print('model:',backbone_lineval)
    else:
        backbone_lineval = SimConv4().to(device)
    
    optimizer = set_optimizer(opt, backbone_lineval)
#     optimizer = torch.optim.Adam([{'params': backbone_lineval.parameters()}], lr=opt.learning_rate, weight_decay=opt.weight_decay)
#     optimizer = optim.SGD(backbone_lineval.parameters(),
#                           lr=opt.learning_rate,
#                           momentum=opt.momentum,
#                           weight_decay=opt.lambda_l2)
#     optimizer = torch.optim.SGD([
#         {'params': self.backbone_lineval.parameters()},
#         {'params': self.relation_head.parameters()},
#         {'params': self.cls_head.parameters()},  {'params': self.tmp_cls_head.parameters()}   ], lr=opt.learning_rate)
       
    BCE = torch.nn.BCEWithLogitsLoss()
    c_criterion = nn.CrossEntropyLoss()

#     backbone_lineval.train()        
#     self.relation_head.train()
#     self.cls_head.train()
#     self.tmp_cls_head.train()
        
    
#     criterion = SupConLoss(temperature = opt.temp)
#     torch.save(backbone_lineval.state_dict(), '{}/backbone_init.tar'.format(opt.ckpt_dir))
    
    ################################################################
    file2print_detail_test = open(opt.logfile, 'a+')
#     f =  open('Moco_RltRsn_training_details.txt', 'w')
#     Trn_filename = './Moco_RltRsn_training_details_{:.4f}_lr{:.4f}_{}.txt'.format(opt.loss_coef,opt.learning_rate, opt.aug_type)
#     f =  open(str(Trn_filename), 'w')
    Trn_filename = './Moco_RltRsn_training_details_{:.4f}_lr{:.4f}_{}.txt'.format(opt.loss_coef,opt.learning_rate, '_'.join(opt.aug_type))
    f =  open(str(Trn_filename), 'w')
    ################################################################

    print('RelationRsn_Moco Train:')
    loss_epoch = []
    inter_loss_epoch = []
    intra_loss_epoch = []
    tmp_loss_epoch = []
    loss_Moco_epoch = []
    
    early_stop_n = 0
    print('opt.loss_coef:',opt.loss_coef)
        
    for epoch in range(opt.epochs):
        backbone_lineval.train()
        
        adjust_learning_rate(opt, optimizer, epoch)
    
        total_loss, total_num, avg_loss = 0.0, 0, 0.0
        

#         for i, (data, data_augmented0, data_augmented1, data_label, tmp_target) in enumerate(train_loader_lineval):
        for i, (data_moco, data, data_augmented0, data_augmented1, data_label, tmp_target) in enumerate(train_loader_lineval):
        
            im_1, im_2 = data_moco[0].to(device), data_moco[1].to(device) 
#             im_1, im_2 = data[0].to(device), data[1].to(device)   
            if epoch==0:
                print('im_1, im_2 shape,len(data):',im_1.shape, im_2.shape, len(data))
            
            bsz = data[0].shape[0]
            print('batch size:',bsz)

            # warm-up learning rate
            warmup_learning_rate(opt, epoch, i, len(train_loader_lineval), optimizer)
            
#             loss = backbone_lineval(im_1, im_2, data, data_augmented0, data_augmented1, data_label, tmp_target, BCE, c_criterion)
            loss, loss_Moco = backbone_lineval(im_1, im_2, data, data_augmented0, data_augmented1, data_label, tmp_target, BCE, c_criterion, f, i, LossCoef= opt.loss_coef)
              
                
            # SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_num += bsz
            total_loss += loss.item() * bsz
        avg_loss = total_loss / total_num
        loss_epoch.append(avg_loss)
        
#         inter_loss_epoch.append(inter_loss)
#         intra_loss_epoch.append(intra_loss)
#         tmp_loss_epoch.append(tmp_loss)
#         loss_Moco_epoch.append(loss_Moco)

        
        print('[Train-{}][{}] loss: {:.5f}; learning rate: {:.3f}\t %' \
              .format(epoch + 1, opt.model_name, avg_loss, optimizer.param_groups[0]['lr']))
        
#         if (epoch>300) & (epoch % opt.save_freq == 0):
#             torch.save({'epoch': epoch, 'state_dict': backbone_lineval.state_dict(), 'optimizer' : optimizer.state_dict(),}, 
#                        '{}/backbone_{epoch}.tar'.format(opt.ckpt_dir,epoch=epoch))
            
        ######################################################################
        print('[Train-{}][{}] loss: {:.5f}; \t %' \
            .format(epoch + 1, opt.model_name, avg_loss), file=file2print_detail_test)         
        file2print_detail_test.flush()
        ######################################################################
            
           
 
        if (epoch>99) & (epoch % opt.save_freq == 0):
            torch.save({'epoch': epoch, 'state_dict': backbone_lineval.state_dict(), 'optimizer' : optimizer.state_dict(),},
                       '{}/backbone_{epoch}.tar'.format(opt.ckpt_dir,epoch=epoch))           
 
        if(loss_Moco<7.7):
            early_stop_n = early_stop_n+1
        if early_stop_n == 10:
            torch.save({'epoch': epoch, 'state_dict': backbone_lineval.state_dict(), 'optimizer' : optimizer.state_dict(),},
                       '{}/backbone_med.tar'.format(opt.ckpt_dir))
        # save model
        torch.save({'epoch': epoch, 'state_dict': backbone_lineval.state_dict(), 'optimizer' : optimizer.state_dict(),},
                   '{}/backbone_last.tar'.format(opt.ckpt_dir))
        
    f.close()
    return avg_loss,loss_epoch 
#     return avg_loss,loss_epoch, inter_loss_epoch,  intra_loss_epoch, tmp_loss_epoch,  loss_Moco_epoch



# SplitBatchNorm: simulate multi-gpu behavior of BatchNorm in one gpu by splitting alone the batch dimension
# implementation adapted from https://github.com/davidcpage/cifar10-fast/blob/master/torch_backend.py
class SplitBatchNorm(nn.BatchNorm1d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        
    def forward(self, input):
        N, C, L = input.shape
#         print('SplitBatchNorm input shape:',input.shape)
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, L), running_mean_split, running_var_split, 
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, L)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(
                input, self.running_mean, self.running_var, 
                self.weight, self.bias, False, self.momentum, self.eps)
        
        
bn_splits = 8
# norm_layer = partial(SplitBatchNorm, num_splits=opt.bn_splits) if opt.bn_splits > 1 else nn.BatchNorm1d
# print('norm_layer:',norm_layer)



class MocoConvSC(nn.Module):
    """backbone + projection head"""
    def __init__(self, opt, head='mlp', feat_dim=128):
        super(MocoConvSC, self).__init__()
#         model_fun, dim_in = model_dict[name]

        OUT_Dim =1  
        O_FeatCOV=1
        O_CHENNEL = 16
        O_EmbDim = opt.feature_size

        norm_layer = partial(SplitBatchNorm, num_splits=opt.bn_splits) if opt.bn_splits > 1 else nn.BatchNorm1d
    
        dim_in = opt.feature_size
        
        self.encoder =  ConvSC_NoFdfwd( 
            O_channel  = O_CHENNEL,
            output_dim = OUT_Dim,
            hidden_size = 150,
            O_feature = O_FeatCOV,
            embed_dim = O_EmbDim,
            num_heads = 2,
            norm_layer = norm_layer
        )
        
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
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
        
        #################################
#         self._create_weights()  
#     def _create_weights(self, mean=0.0, std=0.05):
#         for module in self.modules():
#             if isinstance(module, (nn.Conv1d, nn.Linear)):
#                 nn.init.orthogonal(module.weight)
#             if isinstance(module, nn.BatchNorm1d):
#                 module.weight.data.normal_(mean, std) 
        #################################   
    
    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat

    
    


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

    
def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer




######################################Define MoCo wrapper################################
#########################################################################################
class ModelMoCo(nn.Module):
    def __init__(self, opt, dim=128, K=512, m=0.99, T=0.1, bn_splits=8, symmetric=True , feature_size=64, nb_class=3, tmp_C = 5 ):
        super(ModelMoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.symmetric = symmetric

        # create the encoders
        self.encoder_q = MocoConvSC(opt=opt,feat_dim=opt.moco_dim)
        self.encoder_k = MocoConvSC(opt=opt,feat_dim=opt.moco_dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        
        #####################################
        # relation-reasoning related modules:
        #####################################
        self.hidfeatsize = 64 #128 #64
        
#         self.relation_head = torch.nn.Sequential(
#                                  torch.nn.Linear(feature_size*2, self.hidfeatsize),
#                                  torch.nn.BatchNorm1d(self.hidfeatsize),
#                                  torch.nn.LeakyReLU(),
#                                  torch.nn.Linear(self.hidfeatsize, 1)
#         )
#         self.cls_head = torch.nn.Sequential(
#             torch.nn.Linear(feature_size*2, self.hidfeatsize),
#             torch.nn.BatchNorm1d(self.hidfeatsize),
#             torch.nn.LeakyReLU(),
#             torch.nn.Linear(self.hidfeatsize, nb_class),
#             torch.nn.Softmax(),
#         )
#         self.tmp_cls_head = torch.nn.Sequential(
# #             torch.nn.Linear(feature_size, 128),
# #             torch.nn.BatchNorm1d(128),
#             torch.nn.LeakyReLU(),
#             torch.nn.Linear(feature_size, tmp_C), #128
#             torch.nn.Softmax(),
#         )
            
        self.relation_head = relation_head(feature_size,self.hidfeatsize)
        self.cls_head = cls_head(feature_size, self.hidfeatsize, nb_class)
        self.tmp_cls_head = tmp_cls_head(feature_size, self.hidfeatsize, tmp_C)

         
        
#     def _create_weights(self, mean=0.0, std=0.05):
#         for module in self.modules():
#             if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
#                 module.weight.data.normal_(mean, std) 
#             if isinstance(module, nn.BatchNorm1d):
#                 module.weight.data.normal_(mean, std) 
                
                
    def aggregate(self, features, K):
        relation_pairs_list = list()
        targets_list = list()
        size = int(features.shape[0] / K)
#         print('features.shape,size:',features.shape,size)
        shifts_counter=1
        for index_1 in range(0, size*K, size):
            for index_2 in range(index_1+size, size*K, size):

            # Using the 'cat' aggregation function by default
                pos1 = features[index_1:index_1 + size]
                pos2 = features[index_2:index_2 + size]
#                 print('pos1',pos1.shape)
                pos_pair = torch.cat([pos1,
                                      pos2], 1)  # (batch_size, fz*2)

                # Shuffle without collisions by rolling the mini-batch (negatives)
                neg1 = torch.roll(features[index_2:index_2 + size],
                                  shifts=shifts_counter, dims=0)
                neg_pair1 = torch.cat([pos1, neg1], 1) # (batch_size, fz*2)

                relation_pairs_list.append(pos_pair)
                relation_pairs_list.append(neg_pair1)

                targets_list.append(torch.ones(size, dtype=torch.float32).to(device))
                targets_list.append(torch.zeros(size, dtype=torch.float32).to(device))

                shifts_counter+=1
                if(shifts_counter>=size):
                    shifts_counter=1 # avoid identity pairs
        relation_pairs = torch.cat(relation_pairs_list, 0).to(device)  # K(K-1) * (batch_size, fz*2)
        targets = torch.cat(targets_list, 0).to(device)
        return relation_pairs, targets        
        
    def run_test(self, predict, labels):
        correct = 0
        pred = predict.data.max(1)[1]
        correct = pred.eq(labels.data).cpu().sum()
        return correct, len(labels.data)
    
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
#         assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).to(device)#.cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def contrastive_loss(self, im_q, im_k):
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
#         q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)

            k = self.encoder_k(im_k_)  # keys: NxC
#             k = nn.functional.normalize(k, dim=1)  # already normalized

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

#         print('key,queue shape:',q.shape,k.shape,l_pos.shape,self.queue.clone().detach().shape)
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)#.cuda()
        
        loss = nn.CrossEntropyLoss().to(device)(logits, labels)   
#         loss = nn.CrossEntropyLoss().cuda()(logits, labels)


        return loss, q, k

    def forward(self, im1, im2,data, data_augmented0, data_augmented1, data_label,tmp_target, BCE, c_criterion,f, i, LossCoef ):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """

        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()
            

        # compute loss
        if self.symmetric:  # asymmetric loss
            loss_12, q1, k2 = self.contrastive_loss(im1, im2)
            loss_21, q2, k1 = self.contrastive_loss(im2, im1)
            loss_Moco = loss_12 + loss_21
            k = torch.cat([k1, k2], dim=0)
        else:  # asymmetric loss
            loss_Moco, q, k = self.contrastive_loss(im1, im2)

        self._dequeue_and_enqueue(k)
        
        

        #######################################################
        # Relation reasoning part
        #######################################################

#         self.encoder_q.train()
        self.relation_head.train()
        self.cls_head.train()
        self.tmp_cls_head.train()              

  
        K = len(data) # tot augmentations
#                 print('i,K:',i,K)
        x = torch.cat(data, 0).to(device)

        x_cut0 = torch.cat(data_augmented0, 0).to(device)
        x_cut1 = torch.cat(data_augmented1, 0).to(device)
        c_label = torch.cat(data_label, 0).to(device)


#         features = self.encoder_q.encoder(x)
#         features_cut0 = self.encoder_q.encoder(x_cut0)
#         features_cut1 = self.encoder_q.encoder(x_cut1)
        
        features = self.encoder_q(x)
        features_cut0 = self.encoder_q(x_cut0)
        features_cut1 = self.encoder_q(x_cut1)

#                 print('x_cut0,features_cut0',x_cut0.shape, features_cut0.shape)

        features_cls = torch.cat([features_cut0, features_cut1], 1)

#                 print('features_cut0,features_cls',features_cut0.shape, features_cls.shape)

        # aggregation function
        relation_pairs, targets = self.aggregate(features, K)
#                 print('relation_pairs:',relation_pairs.shape)
        # relation_pairs_intra, targets_intra = self.aggregate_intra(features_cut0, features_cut1, K)

        # forward pass (relation head)
#                 print('relation_pairs:',relation_pairs.shape)
#                 print('self.relation_head:',self.relation_head)

        score = self.relation_head(relation_pairs).squeeze()
        c_output = self.cls_head(features_cls)
        correct_cls, length_cls = self.run_test(c_output, c_label)

        #######################################################
        tmp_c_output = self.tmp_cls_head(features)
        tmp_label = torch.cat(tmp_target, 0).to(device)
        tmp_correct_cls, tmp_length_cls = self.run_test(tmp_c_output, tmp_label)
        #######################################################

        # cross-entropy loss and backward
        loss = BCE(score, targets)
        loss_c = c_criterion(c_output, c_label)
        #######################################################
        tmp_loss_c = c_criterion(tmp_c_output, tmp_label)
        
        print('inter-rlt-loss {:.5f}, intra-rlt-loss {:.5f}, tmp-loss {:.5f}, moco-loss {:.5f} \t %'.format(loss.item(),loss_c.item() ,tmp_loss_c.item() , loss_Moco.item()))
        
#         tol_loss = loss +  loss_c + tmp_loss_c + loss_Moco/10

        if LossCoef == np.inf:
            tol_loss = loss_Moco
        else:
            tol_loss = loss +  loss_c + tmp_loss_c + loss_Moco*LossCoef
        
#         tol_loss = loss + loss_c + tmp_loss_c + loss_Moco/10
        
#         tol_loss /=4
        #######################################################        


        # estimate the accuracy
        predicted = torch.round(torch.sigmoid(score))
        correct = predicted.eq(targets.view_as(predicted)).sum()
        accuracy = (100.0 * correct / float(len(targets)))
        accuracy_cls = 100. * correct_cls / length_cls
        tmp_accuracy_cls = 100. * tmp_correct_cls / tmp_length_cls

                
        if i == 0:
#             f.write('{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(loss.item(),loss_c.item() ,tmp_loss_c.item(), loss_Moco.item()   ))
            f.write('{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(loss.item(),loss_c.item() ,tmp_loss_c.item(), loss_Moco.item(), accuracy.item(), accuracy_cls.item(), tmp_accuracy_cls.item()   ))
            f.write('\n')       

        return tol_loss , loss_Moco
#         return tol_loss ,  loss, loss_c, tmp_loss_c, loss_Moco
#         return loss_Moco


    
    
    
######################################### Define train/test######################################################## 
