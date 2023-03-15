# -*- coding: utf-8 -*-

import torch
import numpy
import utils.transforms as transforms
from dataloader.ucr2018 import *
import torch.utils.data as data
from model.model_RelationalReasoning import *
from model.model_backbone import SimConv4,ConvSC,ConvSC_NoFdfwd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def pretrain_IntraSampleRel(x_train, y_train, opt):
    K = opt.K
    batch_size = opt.batch_size  # 128 has been used in the paper
    tot_epochs = opt.epochs  # 400 has been used in the paper
    feature_size = opt.feature_size
    ckpt_dir = opt.ckpt_dir
    selfsup_model = opt.backbone

#     prob = 0.2  # Transform Probability
    prob = 0.2  # Transform Probability
    raw = transforms.Raw()
    cutout = transforms.Cutout(sigma=0.1, p=prob)
    jitter = transforms.Jitter(sigma=0.2, p=prob)
    scaling = transforms.Scaling(sigma=0.4, p=prob)
    magnitude_warp = transforms.MagnitudeWrap(sigma=0.3, knot=4, p=prob)
    time_warp = transforms.TimeWarp(sigma=0.2, knot=8, p=prob)
    window_slice = transforms.WindowSlice(reduce_ratio=0.8, p=prob)
    window_warp = transforms.WindowWarp(window_ratio=0.3, scales=(0.5, 2), p=prob)

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
                       'none': [raw]}

    transforms_targets = list()
    for name in opt.aug_type:
        for item in transforms_list[name]:
            transforms_targets.append(item)

    train_transform = transforms.Compose(transforms_targets)

#     if '2C' in opt.class_type:
#         cut_piece = transforms.CutPiece2C(sigma=opt.piece_size)
#         nb_class=2
#     elif '3C' in opt.class_type:
#         cut_piece = transforms.CutPiece3C(sigma=opt.piece_size)
#         nb_class=3
#     elif '4C' in opt.class_type:
#         cut_piece = transforms.CutPiece4C(sigma=opt.piece_size)
#         nb_class=4
#     elif '5C' in opt.class_type:
#         cut_piece = transforms.CutPiece5C(sigma=opt.piece_size)
#         nb_class = 5
#     elif '6C' in opt.class_type:
#         cut_piece = transforms.CutPiece6C(sigma=opt.piece_size)
#         nb_class = 6
#     elif '7C' in opt.class_type:
#         cut_piece = transforms.CutPiece7C(sigma=opt.piece_size)
#         nb_class = 7
#     elif '8C' in opt.class_type:
#         cut_piece = transforms.CutPiece8C(sigma=opt.piece_size)
#         nb_class = 8


##########################################=my=#######################################
    cut_piece = transforms.CutPiece(sigma = opt.piece_size,typeC = opt.CutPiece_type)
    nb_class = opt.CutPiece_type        
#################################################################################
    
    tensor_transform = transforms.ToTensor()
    
    OUT_Dim =1  
    O_FeatCOV=1
    O_CHENNEL = 16
    if selfsup_model=='ConvSC':
        backbone = ConvSC( 
            O_channel  = O_CHENNEL,
            output_dim = OUT_Dim,
            hidden_size = 150,
            O_feature = O_FeatCOV,
            embed_dim = 256,
            num_heads = 2
        ).to(device)
    else:
        backbone = SimConv4().to(device)
    
#     backbone = SimConv4().to(device)
    
    print('backbone:',backbone)
    print('x_train',x_train.shape)
    
    model = RelationalReasoning_Intra(backbone, feature_size, nb_class,tmp_C).to(device)
#     model = RelationalReasoning_Intra(backbone, feature_size, nb_class).to(device)

    train_set = MultiUCR2018_Intra(data=x_train, targets=y_train, K=K,
                               transform=train_transform, transform_cut=cut_piece,
                               totensor_transform=tensor_transform)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    torch.save(model.backbone.state_dict(), '{}/backbone_init.tar'.format(ckpt_dir))
    acc_max, epoch_max = model.train(tot_epochs=tot_epochs, train_loader=train_loader, opt=opt)

    torch.save(model.backbone.state_dict(), '{}/backbone_last.tar'.format(ckpt_dir))

    return acc_max, epoch_max


def pretrain_InterSampleRel(x_train, y_train, opt):
    K = opt.K
    batch_size = opt.batch_size  # 128 has been used in the paper
    tot_epochs = opt.epochs  # 400 has been used in the paper
    feature_size = opt.feature_size
    ckpt_dir = opt.ckpt_dir
    selfsup_model = opt.backbone

    prob = 0.2  # Transform Probability
    raw = transforms.Raw()
    cutout = transforms.Cutout(sigma=0.1, p=prob)
    jitter = transforms.Jitter(sigma=0.2, p=prob)
    scaling = transforms.Scaling(sigma=0.4, p=prob)
    magnitude_warp = transforms.MagnitudeWrap(sigma=0.3, knot=4, p=prob)
    time_warp = transforms.TimeWarp(sigma=0.2, knot=8, p=prob)
    window_slice = transforms.WindowSlice(reduce_ratio=0.8, p=prob)
    window_warp = transforms.WindowWarp(window_ratio=0.3, scales=(0.5, 2), p=prob)

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
                       'none': [raw]}

    transforms_targets = list()
    for name in opt.aug_type:
        for item in transforms_list[name]:
            transforms_targets.append(item)
    train_transform = transforms.Compose(transforms_targets + [transforms.ToTensor()])
    
##########################################=my=#######################################      
    tmp_C = opt.tmp_C
#################################################################################

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print('more than 1 gpu!')
        else:
            print('single gpu')
    else:
        print('no gpu available')
    print(torch.cuda.is_available())
    

#     backbone = SimConv4().to(device)   

    OUT_Dim =1  
    O_FeatCOV=1
    O_CHENNEL = 16
    if selfsup_model=='ConvSC':
        backbone = ConvSC( 
            O_channel  = O_CHENNEL,
            output_dim = OUT_Dim,
            hidden_size = 150,
            O_feature = O_FeatCOV,
            embed_dim = 256,
            num_heads = 2
        ).to(device)
    else:
        backbone = SimConv4().to(device)
    print('backbone:',backbone)
    print('x_train',x_train.shape)
    
    
    
    
#     model = RelationalReasoning(backbone, feature_size).to(device)
    ########################################################################
    model = RelationalReasoning(backbone, feature_size, tmp_C).to(device)
    ########################################################################

    train_set = MultiUCR2018(data=x_train, targets=y_train, K=K, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    torch.save(model.backbone.state_dict(), '{}/backbone_init.tar'.format(ckpt_dir))
    acc_max, epoch_max = model.train(tot_epochs=tot_epochs, train_loader=train_loader, opt=opt)

    torch.save(model.backbone.state_dict(), '{}/backbone_last.tar'.format(ckpt_dir))

    return acc_max, epoch_max


def pretrain_SelfTime(x_train, y_train, opt):
    K = opt.K
    batch_size = opt.batch_size  # 128 has been used in the paper
    tot_epochs = opt.epochs  # 400 has been used in the paper
    feature_size = opt.feature_size
    ckpt_dir = opt.ckpt_dir
    selfsup_model = opt.backbone

    prob = 0.3 #0.2  # Transform Probability
#     prob = 0.2  # Transform Probability
    cutout = transforms.Cutout(sigma=0.1, p=prob)
    jitter = transforms.Jitter(sigma=0.2, p=prob)
    scaling = transforms.Scaling(sigma=0.4, p=prob)
    magnitude_warp = transforms.MagnitudeWrap(sigma=0.3, knot=4, p=prob)
    time_warp = transforms.TimeWarp(sigma=0.2, knot=8, p=prob)
#     window_slice = transforms.WindowSlice(reduce_ratio=0.8, p=prob)
    window_slice = transforms.WindowSlice(reduce_ratio= numpy.random.uniform(0.82,0.985), p=0.3)
    window_warp = transforms.WindowWarp(window_ratio=0.3, scales=(0.5, 2), p=prob)

    transforms_list = {'jitter': jitter,
                       'cutout': cutout,
                       'scaling': scaling,
                       'magnitude_warp': magnitude_warp,
                       'time_warp': time_warp,
                       'window_slice': window_slice,
                       'window_warp': window_warp,
                       'G0': [jitter, magnitude_warp, window_slice],
                       'G1': [jitter, time_warp, window_slice],
                       'G2': [jitter, time_warp, window_slice, window_warp, cutout],
                       'none': []}

    transforms_targets = [transforms_list[name] for name in opt.aug_type]
    train_transform = transforms.Compose(transforms_targets)
    tensor_transform = transforms.ToTensor()

#     if '2C' in opt.class_type:
#         cut_piece = transforms.CutPiece2C(sigma=opt.piece_size)
#         nb_class=2
#     elif '3C' in opt.class_type:
#         cut_piece = transforms.CutPiece3C(sigma=opt.piece_size)
#         nb_class=3
#     elif '4C' in opt.class_type:
#         cut_piece = transforms.CutPiece4C(sigma=opt.piece_size)
#         nb_class=4
#     elif '5C' in opt.class_type:
#         cut_piece = transforms.CutPiece5C(sigma=opt.piece_size)
#         nb_class = 5
#     elif '6C' in opt.class_type:
#         cut_piece = transforms.CutPiece6C(sigma=opt.piece_size)
#         nb_class = 6
#     elif '7C' in opt.class_type:
#         cut_piece = transforms.CutPiece7C(sigma=opt.piece_size)
#         nb_class = 7
#     elif '8C' in opt.class_type:
#         cut_piece = transforms.CutPiece8C(sigma=opt.piece_size)
#         nb_class = 8

##########################################=my=#######################################
    cut_piece = transforms.CutPiece(sigma = opt.piece_size,typeC = opt.CutPiece_type)
    nb_class = opt.CutPiece_type        
    tmp_C = opt.tmp_C
#################################################################################

#     backbone = SimConv4().to(device)
    
    OUT_Dim =1  
    O_FeatCOV=1
    O_CHENNEL = 16

    if selfsup_model=='ConvSC':
        backbone = ConvSC( 
            O_channel  = O_CHENNEL,
            output_dim = OUT_Dim,
            hidden_size = 150,
            O_feature = O_FeatCOV,
            ################################
            dim_feedforward = feature_size*2,
            ################################
            embed_dim = feature_size,
            num_heads = 2
        ).to(device)
    elif selfsup_model=='ConvSC_NoFdfwd':
        backbone = ConvSC_NoFdfwd( 
            O_channel  = O_CHENNEL,
            output_dim = OUT_Dim,
            hidden_size = 150,
            O_feature = O_FeatCOV,
            ################################
#             dim_feedforward = feature_size*2,
            ################################
            embed_dim = feature_size,
            num_heads = 2
        ).to(device)
    else:
        backbone = SimConv4().to(device)
    
    print('backbone:',backbone)
    print('x_train',x_train.shape)
#     model = RelationalReasoning_InterIntra(backbone, feature_size, nb_class,tmp_C).to(device)

    ######################################################################
    model = RelationalReasoning_InterIntra_Tmp(backbone, feature_size, nb_class,tmp_C).to(device)
    ######################################################################

    train_set = MultiUCR2018_InterIntra(data=x_train, targets=y_train, K=K,
                                        transform=train_transform, transform_cut=cut_piece,
                                        totensor_transform=tensor_transform)
    ######################################################################
#     print('train_set:',len(train_set),train_set[0][0][0].shape,len(train_set[0][0]),len(train_set[0]))
    ######################################################################
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    print('train_loader:',len(train_loader))
    torch.save(model.backbone.state_dict(), '{}/backbone_init.tar'.format(ckpt_dir))
    acc_max, epoch_max = model.train(tot_epochs=tot_epochs, train_loader=train_loader, opt=opt)

    torch.save(model.backbone.state_dict(), '{}/backbone_last.tar'.format(ckpt_dir))

    return acc_max, epoch_max




######################20230115#######################
def pretrain_SelfTime_AugProb1(x_train, y_train, opt):
    K = opt.K
    batch_size = opt.batch_size  # 128 has been used in the paper
    tot_epochs = opt.epochs  # 400 has been used in the paper
    feature_size = opt.feature_size
    ckpt_dir = opt.ckpt_dir
    selfsup_model = opt.backbone

    prob = 1 #0.2  # Transform Probability
#     prob = 0.2  # Transform Probability
    cutout = transforms.Cutout(sigma=0.1, p=prob)
    jitter = transforms.Jitter(sigma=0.2, p=prob)
    scaling = transforms.Scaling(sigma=0.4, p=prob)
    magnitude_warp = transforms.MagnitudeWrap(sigma=0.3, knot=4, p=prob)
    time_warp = transforms.TimeWarp(sigma=0.2, knot=8, p=prob)
#     window_slice = transforms.WindowSlice(reduce_ratio=0.8, p=prob)
    window_slice = transforms.WindowSlice(reduce_ratio= numpy.random.uniform(0.82,0.985), p=prob)
    window_warp = transforms.WindowWarp(window_ratio=0.3, scales=(0.5, 2), p=prob)

    transforms_list = {'jitter': jitter,
                       'cutout': cutout,
                       'scaling': scaling,
                       'magnitude_warp': magnitude_warp,
                       'time_warp': time_warp,
                       'window_slice': window_slice,
                       'window_warp': window_warp,
                       'G0': [jitter, magnitude_warp, window_slice],
                       'G1': [jitter, time_warp, window_slice],
                       'G2': [jitter, time_warp, window_slice, window_warp, cutout],
                       'none': []}

    transforms_targets = [transforms_list[name] for name in opt.aug_type]
    train_transform = transforms.Compose(transforms_targets)
    tensor_transform = transforms.ToTensor()


##########################################=my=#######################################
    cut_piece = transforms.CutPiece(sigma = opt.piece_size,typeC = opt.CutPiece_type)
    nb_class = opt.CutPiece_type        
    tmp_C = opt.tmp_C
#################################################################################

#     backbone = SimConv4().to(device)
    
    OUT_Dim =1  
    O_FeatCOV=1
    O_CHENNEL = 16

    if selfsup_model=='ConvSC':
        backbone = ConvSC( 
            O_channel  = O_CHENNEL,
            output_dim = OUT_Dim,
            hidden_size = 150,
            O_feature = O_FeatCOV,
            ################################
            dim_feedforward = feature_size*2,
            ################################
            embed_dim = feature_size,
            num_heads = 2
        ).to(device)
    elif selfsup_model=='ConvSC_NoFdfwd':
        backbone = ConvSC_NoFdfwd( 
            O_channel  = O_CHENNEL,
            output_dim = OUT_Dim,
            hidden_size = 150,
            O_feature = O_FeatCOV,
            ################################
#             dim_feedforward = feature_size*2,
            ################################
            embed_dim = feature_size,
            num_heads = 2
        ).to(device)
    else:
        backbone = SimConv4().to(device)
    
    print('backbone:',backbone)
    print('x_train',x_train.shape)
#     model = RelationalReasoning_InterIntra(backbone, feature_size, nb_class,tmp_C).to(device)

    ######################################################################
    model = RelationalReasoning_InterIntra_Tmp_20230115(backbone, feature_size, nb_class,tmp_C).to(device)
    ######################################################################

    train_set = MultiUCR2018_InterIntra(data=x_train, targets=y_train, K=K,
                                        transform=train_transform, transform_cut=cut_piece,
                                        totensor_transform=tensor_transform)
    ######################################################################
#     print('train_set:',len(train_set),train_set[0][0][0].shape,len(train_set[0][0]),len(train_set[0]))
    ######################################################################
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    print('train_loader:',len(train_loader))
    torch.save(model.backbone.state_dict(), '{}/backbone_init.tar'.format(ckpt_dir))
    acc_max, epoch_max = model.train(tot_epochs=tot_epochs, train_loader=train_loader, opt=opt)

    torch.save(model.backbone.state_dict(), '{}/backbone_last.tar'.format(ckpt_dir))

    return acc_max, epoch_max




def pretrain_SelfTime_NoTmp(x_train, y_train, opt):
    K = opt.K
    batch_size = opt.batch_size  # 128 has been used in the paper
    tot_epochs = opt.epochs  # 400 has been used in the paper
    feature_size = opt.feature_size
    ckpt_dir = opt.ckpt_dir
    selfsup_model = opt.backbone

    prob = 0.2  # Transform Probability
#     prob = 0.2  # Transform Probability
    cutout = transforms.Cutout(sigma=0.1, p=prob)
    jitter = transforms.Jitter(sigma=0.2, p=prob)
    scaling = transforms.Scaling(sigma=0.4, p=prob)
    magnitude_warp = transforms.MagnitudeWrap(sigma=0.3, knot=4, p=prob)
    time_warp = transforms.TimeWarp(sigma=0.2, knot=8, p=prob)
#     window_slice = transforms.WindowSlice(reduce_ratio=0.8, p=prob)
    window_slice = transforms.WindowSlice(reduce_ratio= numpy.random.uniform(0.82,0.98), p=0.3)
    window_warp = transforms.WindowWarp(window_ratio=0.3, scales=(0.5, 2), p=prob)

    transforms_list = {'jitter': jitter,
                       'cutout': cutout,
                       'scaling': scaling,
                       'magnitude_warp': magnitude_warp,
                       'time_warp': time_warp,
                       'window_slice': window_slice,
                       'window_warp': window_warp,
                       'G0': [jitter, magnitude_warp, window_slice],
                       'G1': [jitter, time_warp, window_slice],
                       'G2': [jitter, time_warp, window_slice, window_warp, cutout],
                       'none': []}

    transforms_targets = [transforms_list[name] for name in opt.aug_type]
    train_transform = transforms.Compose(transforms_targets)
    tensor_transform = transforms.ToTensor()

#     if '2C' in opt.class_type:
#         cut_piece = transforms.CutPiece2C(sigma=opt.piece_size)
#         nb_class=2
#     elif '3C' in opt.class_type:
#         cut_piece = transforms.CutPiece3C(sigma=opt.piece_size)
#         nb_class=3
#     elif '4C' in opt.class_type:
#         cut_piece = transforms.CutPiece4C(sigma=opt.piece_size)
#         nb_class=4
#     elif '5C' in opt.class_type:
#         cut_piece = transforms.CutPiece5C(sigma=opt.piece_size)
#         nb_class = 5
#     elif '6C' in opt.class_type:
#         cut_piece = transforms.CutPiece6C(sigma=opt.piece_size)
#         nb_class = 6
#     elif '7C' in opt.class_type:
#         cut_piece = transforms.CutPiece7C(sigma=opt.piece_size)
#         nb_class = 7
#     elif '8C' in opt.class_type:
#         cut_piece = transforms.CutPiece8C(sigma=opt.piece_size)
#         nb_class = 8

##########################################=my=#######################################
    cut_piece = transforms.CutPiece(sigma = opt.piece_size,typeC = opt.CutPiece_type)
    nb_class = opt.CutPiece_type        
    tmp_C = opt.tmp_C
#################################################################################

#     backbone = SimConv4().to(device)
    
    OUT_Dim =1  
    O_FeatCOV=1
    O_CHENNEL = 16

    if selfsup_model=='ConvSC':
        backbone = ConvSC( 
            O_channel  = O_CHENNEL,
            output_dim = OUT_Dim,
            hidden_size = 150,
            O_feature = O_FeatCOV,
            ################################
            dim_feedforward = feature_size*2,
            ################################
            embed_dim = feature_size,
            num_heads = 2
        ).to(device)
    elif selfsup_model=='ConvSC_NoFdfwd':
        backbone = ConvSC_NoFdfwd( 
            O_channel  = O_CHENNEL,
            output_dim = OUT_Dim,
            hidden_size = 150,
            O_feature = O_FeatCOV,
            ################################
#             dim_feedforward = feature_size*2,
            ################################
            embed_dim = feature_size,
            num_heads = 2
        ).to(device)
    else:
        backbone = SimConv4().to(device)
    
    print('backbone:',backbone)
    print('x_train',x_train.shape)
    model = RelationalReasoning_InterIntra_Org(backbone, feature_size, nb_class,tmp_C).to(device)

    ######################################################################
#     model = RelationalReasoning_InterIntra_Tmp(backbone, feature_size, nb_class,tmp_C).to(device)
    ######################################################################

    train_set = MultiUCR2018_InterIntra(data=x_train, targets=y_train, K=K,
                                        transform=train_transform, transform_cut=cut_piece,
                                        totensor_transform=tensor_transform)
    ######################################################################
#     print('train_set:',len(train_set),train_set[0][0][0].shape,len(train_set[0][0]),len(train_set[0]))
    ######################################################################
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    print('train_loader:',len(train_loader))
    torch.save(model.backbone.state_dict(), '{}/backbone_init.tar'.format(ckpt_dir))
    acc_max, epoch_max = model.train(tot_epochs=tot_epochs, train_loader=train_loader, opt=opt)

    torch.save(model.backbone.state_dict(), '{}/backbone_last.tar'.format(ckpt_dir))

    return acc_max, epoch_max









def pretrain_SelfTime_C1(x_train, y_train, opt):
    K = opt.K
    batch_size = opt.batch_size  # 128 has been used in the paper
    tot_epochs = opt.epochs  # 400 has been used in the paper
    feature_size = opt.feature_size
    ckpt_dir = opt.ckpt_dir
    selfsup_model = opt.backbone

    prob = 0.3 #0.2  # Transform Probability
#     prob = 0.2  # Transform Probability
    cutout = transforms.Cutout(sigma=0.1, p=prob)
    jitter = transforms.Jitter(sigma=0.2, p=prob)
    scaling = transforms.Scaling(sigma=0.4, p=prob)
    magnitude_warp = transforms.MagnitudeWrap(sigma=0.3, knot=4, p=prob)
    time_warp = transforms.TimeWarp(sigma=0.2, knot=8, p=prob)
#     window_slice = transforms.WindowSlice(reduce_ratio=0.8, p=prob)
    window_slice = transforms.WindowSlice(reduce_ratio= numpy.random.uniform(0.82,0.985), p=0.3)
    window_warp = transforms.WindowWarp(window_ratio=0.3, scales=(0.5, 2), p=prob)

    transforms_list = {'jitter': jitter,
                       'cutout': cutout,
                       'scaling': scaling,
                       'magnitude_warp': magnitude_warp,
                       'time_warp': time_warp,
                       'window_slice': window_slice,
                       'window_warp': window_warp,
                       'G0': [jitter, magnitude_warp, window_slice],
                       'G1': [jitter, time_warp, window_slice],
                       'G2': [jitter, time_warp, window_slice, window_warp, cutout],
                       'none': []}

    transforms_targets = [transforms_list[name] for name in opt.aug_type]
    train_transform = transforms.Compose(transforms_targets)
    tensor_transform = transforms.ToTensor()

#     if '2C' in opt.class_type:
#         cut_piece = transforms.CutPiece2C(sigma=opt.piece_size)
#         nb_class=2
#     elif '3C' in opt.class_type:
#         cut_piece = transforms.CutPiece3C(sigma=opt.piece_size)
#         nb_class=3
#     elif '4C' in opt.class_type:
#         cut_piece = transforms.CutPiece4C(sigma=opt.piece_size)
#         nb_class=4
#     elif '5C' in opt.class_type:
#         cut_piece = transforms.CutPiece5C(sigma=opt.piece_size)
#         nb_class = 5
#     elif '6C' in opt.class_type:
#         cut_piece = transforms.CutPiece6C(sigma=opt.piece_size)
#         nb_class = 6
#     elif '7C' in opt.class_type:
#         cut_piece = transforms.CutPiece7C(sigma=opt.piece_size)
#         nb_class = 7
#     elif '8C' in opt.class_type:
#         cut_piece = transforms.CutPiece8C(sigma=opt.piece_size)
#         nb_class = 8

##########################################=my=#######################################
    cut_piece = transforms.CutPiece(sigma = opt.piece_size,typeC = opt.CutPiece_type)
    nb_class = opt.CutPiece_type        
    tmp_C = opt.tmp_C
#################################################################################

#     backbone = SimConv4().to(device)
    
    OUT_Dim =1  
    O_FeatCOV=1
    O_CHENNEL = 16

    if selfsup_model=='ConvSC':
        backbone = ConvSC( 
            O_channel  = O_CHENNEL,
            output_dim = OUT_Dim,
            hidden_size = 150,
            O_feature = O_FeatCOV,
            ################################
            dim_feedforward = feature_size*2,
            ################################
            embed_dim = feature_size,
            num_heads = 2
        ).to(device)
    elif selfsup_model=='ConvSC_NoFdfwd':
        backbone = ConvSC_NoFdfwd( 
            O_channel  = O_CHENNEL,
            output_dim = OUT_Dim,
            hidden_size = 150,
            O_feature = O_FeatCOV,
            ################################
#             dim_feedforward = feature_size*2,
            ################################
            embed_dim = feature_size,
            num_heads = 2
        ).to(device)
    else:
        backbone = SimConv4().to(device)
    
    print('backbone:',backbone)
    print('x_train',x_train.shape)
#     model = RelationalReasoning_InterIntra(backbone, feature_size, nb_class,tmp_C).to(device)

    ######################################################################
    model = RelationalReasoning_InterIntra_Tmp_C1(backbone, feature_size, nb_class,tmp_C).to(device)
    ######################################################################

    train_set = MultiUCR2018_InterIntra(data=x_train, targets=y_train, K=K,
                                        transform=train_transform, transform_cut=cut_piece,
                                        totensor_transform=tensor_transform)
    ######################################################################
#     print('train_set:',len(train_set),train_set[0][0][0].shape,len(train_set[0][0]),len(train_set[0]))
    ######################################################################
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    print('train_loader:',len(train_loader))
    torch.save(model.backbone.state_dict(), '{}/backbone_init.tar'.format(ckpt_dir))
    acc_max, epoch_max = model.train(tot_epochs=tot_epochs, train_loader=train_loader, opt=opt)

#     torch.save(model.backbone.state_dict(), '{}/backbone_last.tar'.format(ckpt_dir))

    return acc_max, epoch_max

