# -*- coding: utf-8 -*-

import torch
from optim.pytorchtools import EarlyStopping
import torch.nn as nn
import pandas as pd
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from utils.utils import adjust_learning_rate, warmup_learning_rate

class RelationalReasoning(torch.nn.Module):

#     def __init__(self, backbone, feature_size=64):
    def __init__(self, backbone, feature_size=64 ,tmp_C = 10):
        
        super(RelationalReasoning, self).__init__()
        self.backbone = backbone
        self.relation_head = torch.nn.Sequential(
                                 torch.nn.Linear(feature_size*2, 256),
                                 torch.nn.BatchNorm1d(256),
                                 torch.nn.LeakyReLU(),
                                 torch.nn.Linear(256, 1))
        ###############################my change##############################
        self.tmp_cls_head = torch.nn.Sequential(
#             torch.nn.Linear(feature_size, 128),
#             torch.nn.BatchNorm1d(128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(feature_size, tmp_C), #128
            torch.nn.Softmax(),
        )
        print('tmp_C:',tmp_C)
        ######################################################################

    def run_test(self, predict, labels):
        correct = 0
        pred = predict.data.max(1)[1]
        correct = pred.eq(labels.data).cpu().sum()
        return correct, len(labels.data)
    
    def aggregate(self, features, K):
        relation_pairs_list = list()
        targets_list = list()
        size = int(features.shape[0] / K)
        shifts_counter=1
        for index_1 in range(0, size*K, size):
            for index_2 in range(index_1+size, size*K, size):
            # Using the 'cat' aggregation function by default
                pos1 = features[index_1:index_1 + size]
                pos2 = features[index_2:index_2+size]
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

    def train(self, tot_epochs, train_loader, opt):
        patience = opt.patience
        early_stopping = EarlyStopping(patience, verbose=True,
                                       checkpoint_pth='{}/backbone_best.tar'.format(opt.ckpt_dir))

        optimizer = torch.optim.Adam([
                      {'params': self.backbone.parameters()},
                      {'params': self.relation_head.parameters()}], lr=opt.learning_rate)  
        BCE = torch.nn.BCEWithLogitsLoss()
        c_criterion = nn.CrossEntropyLoss()
        self.backbone.train()
        self.relation_head.train()
        ###############################################
        self.tmp_cls_head.train()
        ###############################################
        epoch_max = 0
        acc_max=0
        
        ################################################################
        file2print_detail_test = open(opt.logfile, 'a+')
        ################################################################
        
        for epoch in range(tot_epochs):

            acc_epoch=0
            loss_epoch=0
            ###############################################
            tmp_acc_epoch_cls = 0
            ###############################################
          # the real target is discarded (unsupervised)
            for i, (data_augmented, tmp_target) in enumerate(train_loader):
                K = len(data_augmented) # tot augmentations
                x = torch.cat(data_augmented, 0).to(device)

                
                
                optimizer.zero_grad()
                # forward pass (backbone)
                features = self.backbone(x)
                
                #######################################################
                tmp_c_output = self.tmp_cls_head(features)
                tmp_label = torch.cat(tmp_target, 0).to(device)
                tmp_correct_cls, tmp_length_cls = self.run_test(tmp_c_output, tmp_label)
                #######################################################
                
                # aggregation function
                relation_pairs, targets = self.aggregate(features, K)

                # forward pass (relation head)
                score = self.relation_head(relation_pairs).squeeze()
                # cross-entropy loss and backward
                                
                loss = BCE(score, targets)
                #######################################################
                tmp_loss_c = c_criterion(tmp_c_output, tmp_label)
                loss+=  tmp_loss_c
                #######################################################
                
                loss.backward()
                optimizer.step()
                # estimate the accuracy
                predicted = torch.round(torch.sigmoid(score))
                correct = predicted.eq(targets.view_as(predicted)).sum()
                accuracy = (100.0 * correct / float(len(targets)))
                acc_epoch += accuracy.item()
                loss_epoch += loss.item()
                
                #######################################################
                tmp_accuracy_cls = 100. * tmp_correct_cls / tmp_length_cls
                tmp_acc_epoch_cls += tmp_accuracy_cls.item()
                #######################################################

            acc_epoch /= len(train_loader)
            loss_epoch /= len(train_loader)

            #######################################################
            tmp_acc_epoch_cls /= len(train_loader)
            #######################################################                

#             if acc_epoch>acc_max:
#                 acc_max = acc_epoch
#                 epoch_max = epoch
                
            #######################################################
            if (acc_epoch + tmp_acc_epoch_cls)>acc_max:
                acc_max = (acc_epoch + tmp_acc_epoch_cls)
                epoch_max = epoch
                
            #######################################################
                
            early_stopping(acc_epoch + tmp_acc_epoch_cls, self.backbone)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if (epoch+1)%opt.save_freq==0:
                print("[INFO] save backbone at epoch {}!".format(epoch))
                torch.save(self.backbone.state_dict(), '{}/backbone_{}.tar'.format(opt.ckpt_dir, epoch))

            print('Epoch [{}][{}][{}] loss= {:.5f}; Epoch ACC.= {:.2f}%, Tmp_CLS.= {:.2f}%, Max ACC.= {:.1f}%, Max Epoch={}' \
                .format(epoch + 1, opt.model_name, opt.dataset_name,
                        loss_epoch, acc_epoch, tmp_acc_epoch_cls, acc_max, epoch_max))
                       

            ######################################################################
            print('Epoch [{}][{}][{}] loss= {:.5f}; Epoch ACC.= {:.2f}%, Tmp_CLS.= {:.2f}%,'
                'Max ACC.= {:.1f}%, Max Epoch={}' \
                .format(epoch + 1, opt.model_name, opt.dataset_name,
                        loss_epoch, acc_epoch, tmp_acc_epoch_cls, acc_max, epoch_max), file=file2print_detail_test)         
            file2print_detail_test.flush()
            ######################################################################
            
        return acc_max, epoch_max


class RelationalReasoning_Intra(torch.nn.Module):

    def __init__(self, backbone, feature_size=64, nb_class=3):
        super(RelationalReasoning_Intra, self).__init__()
        self.backbone = backbone

        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(feature_size*2, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, nb_class),
            torch.nn.Softmax(),
        )

    def run_test(self, predict, labels):
        correct = 0
        pred = predict.data.max(1)[1]
        correct = pred.eq(labels.data).cpu().sum()
        return correct, len(labels.data)

    def train(self, tot_epochs, train_loader, opt):
        patience = opt.patience
        early_stopping = EarlyStopping(patience, verbose=True,
                                       checkpoint_pth='{}/backbone_best.tar'.format(opt.ckpt_dir))

        optimizer = torch.optim.Adam([
                      {'params': self.backbone.parameters()},
            {'params': self.cls_head.parameters()},
        ], lr=opt.learning_rate)
        c_criterion = nn.CrossEntropyLoss()

        self.backbone.train()
        self.cls_head.train()
        epoch_max = 0
        acc_max=0
        for epoch in range(tot_epochs):

            acc_epoch=0
            acc_epoch_cls=0
            loss_epoch=0
          # the real target is discarded (unsupervised)
            for i, (data_augmented0, data_augmented1, data_label, _) in enumerate(train_loader):
                K = len(data_augmented0) # tot augmentations
                x_cut0 = torch.cat(data_augmented0, 0).to(device)
                x_cut1 = torch.cat(data_augmented1, 0).to(device)
                c_label = torch.cat(data_label, 0).to(device)

                optimizer.zero_grad()
                # forward pass (backbone)
                features_cut0 = self.backbone(x_cut0)
                features_cut1 = self.backbone(x_cut1)
                features_cls = torch.cat([features_cut0, features_cut1], 1)

                # score_intra = self.relation_head(relation_pairs_intra).squeeze()
                c_output = self.cls_head(features_cls)
                correct_cls, length_cls = self.run_test(c_output, c_label)

                loss_c = c_criterion(c_output, c_label)
                loss=loss_c

                loss.backward()
                optimizer.step()
                # estimate the accuracy
                loss_epoch += loss.item()

                accuracy_cls = 100. * correct_cls / length_cls
                acc_epoch_cls += accuracy_cls.item()

            acc_epoch_cls /= len(train_loader)
            loss_epoch /= len(train_loader)

            if acc_epoch_cls>acc_max:
                acc_max = acc_epoch_cls
                epoch_max = epoch

            early_stopping(acc_epoch_cls, self.backbone)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if (epoch+1)%opt.save_freq==0:
                print("[INFO] save backbone at epoch {}!".format(epoch))
                torch.save(self.backbone.state_dict(), '{}/backbone_{}.tar'.format(opt.ckpt_dir, epoch))

            print('Epoch [{}][{}][{}] loss= {:.5f}; Epoch ACC.= {:.2f}%, CLS.= {:.2f}%, '
                'Max ACC.= {:.1f}%, Max Epoch={}' \
                .format(epoch + 1, opt.model_name, opt.dataset_name,
                        loss_epoch, acc_epoch,acc_epoch_cls, acc_max, epoch_max))
        return acc_max, epoch_max


    
    

class RelationalReasoning_InterIntra_Org(torch.nn.Module):
    def __init__(self, backbone, feature_size=64, nb_class=3, tmp_C = 10):
        super(RelationalReasoning_InterIntra_Org, self).__init__()
        self.backbone = backbone

        self.relation_head = torch.nn.Sequential(
                                 torch.nn.Linear(feature_size*2, 256),
                                 torch.nn.BatchNorm1d(256),
                                 torch.nn.LeakyReLU(),
                                 torch.nn.Linear(256, 1))
        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(feature_size*2, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, nb_class),
            torch.nn.Softmax(),
        )
        ###############################my change##############################
#         self.tmp_cls_head = torch.nn.Sequential(
#             torch.nn.Linear(feature_size, 128),
#             torch.nn.BatchNorm1d(128),
#             torch.nn.LeakyReLU(),
#             torch.nn.Linear(128, tmp_C),
#             torch.nn.Softmax(),
#         )
        ######################################################################
        # self.softmax = nn.Softmax()

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
                pos2 = features[index_2:index_2+size]
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

    def train(self, tot_epochs, train_loader, opt):
        patience = opt.patience
        early_stopping = EarlyStopping(patience, verbose=True,
                                   checkpoint_pth='{}/backbone_best.tar'.format(opt.ckpt_dir))

        optimizer = torch.optim.Adam([
                      {'params': self.backbone.parameters()},
                      {'params': self.relation_head.parameters()},
            {'params': self.cls_head.parameters()},        ], lr=opt.learning_rate)
        BCE = torch.nn.BCEWithLogitsLoss()
        c_criterion = nn.CrossEntropyLoss()

        self.backbone.train()
        self.relation_head.train()
        self.cls_head.train()
        epoch_max = 0
        acc_max=0
        for epoch in range(tot_epochs):

            acc_epoch=0
            acc_epoch_cls=0
            loss_epoch=0
          # the real target is discarded (unsupervised)
            for i, (data, data_augmented0, data_augmented1, data_label, tmp_target) in enumerate(train_loader):
                K = len(data) # tot augmentations
#                 print('i,K:',i,K)
                x = torch.cat(data, 0).to(device)
        
                x_cut0 = torch.cat(data_augmented0, 0).to(device)
                x_cut1 = torch.cat(data_augmented1, 0).to(device)
                c_label = torch.cat(data_label, 0).to(device)

                #######################################################
#                 print('data,x:',i,x.shape,(data[0].shape))
#                 print('x:',x.shape)
                #######################################################
    
                optimizer.zero_grad()
                # forward pass (backbone)
#                 N = data[0][1]  ### the lenght of the input data
                features = self.backbone(x)
#                 N = data_augmented0[0][1]  ###the lenght of the cutted input data
                features_cut0 = self.backbone(x_cut0)
                features_cut1 = self.backbone(x_cut1)

#                 print('x_cut0,features_cut0',x_cut0.shape, features_cut0.shape)
                
                features_cls = torch.cat([features_cut0, features_cut1], 1)
                
#                 print('features_cut0,features_cls',features_cut0.shape, features_cls.shape)
                
                # aggregation function
                relation_pairs, targets = self.aggregate(features, K)
#                 print('relation_pairs:',relation_pairs.shape)
                # relation_pairs_intra, targets_intra = self.aggregate_intra(features_cut0, features_cut1, K)

                # forward pass (relation head)
                score = self.relation_head(relation_pairs).squeeze()
                c_output = self.cls_head(features_cls)
                correct_cls, length_cls = self.run_test(c_output, c_label)
                
                #######################################################
#                 tmp_label = tmp_target.to(device)
#                 tmp_features = self.backbone(data)
#                 tmp_c_output = self.tmp_cls_head(tmp_features)
#                 tmp_correct_cls, tmp_length_cls = self.run_test(tmp_c_output, tmp_label)
                #######################################################

                # cross-entropy loss and backward
                loss = BCE(score, targets)
                loss_c = c_criterion(c_output, c_label)
                loss+=loss_c

                loss.backward()
                optimizer.step()
                # estimate the accuracy
                predicted = torch.round(torch.sigmoid(score))
                correct = predicted.eq(targets.view_as(predicted)).sum()
                accuracy = (100.0 * correct / float(len(targets)))
                acc_epoch += accuracy.item()
                loss_epoch += loss.item()

                accuracy_cls = 100. * correct_cls / length_cls
                acc_epoch_cls += accuracy_cls.item()

            acc_epoch /= len(train_loader)
            acc_epoch_cls /= len(train_loader)
            loss_epoch /= len(train_loader)

            if (acc_epoch+acc_epoch_cls)>acc_max:
                acc_max = (acc_epoch+acc_epoch_cls)
                epoch_max = epoch

            early_stopping((acc_epoch+acc_epoch_cls), self.backbone)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if (epoch+1)%opt.save_freq==0:
                print("[INFO] save backbone at epoch {}!".format(epoch))
                torch.save(self.backbone.state_dict(), '{}/backbone_{}.tar'.format(opt.ckpt_dir, epoch))

            print('Epoch [{}][{}][{}] loss= {:.5f}; Epoch ACC.= {:.2f}%, CLS.= {:.2f}%, '
                'Max ACC.= {:.1f}%, Max Epoch={}' \
                .format(epoch + 1, opt.model_name, opt.dataset_name,
                        loss_epoch, acc_epoch,acc_epoch_cls, acc_max, epoch_max))
        return acc_max, epoch_max

    
    
    
    
class RelationalReasoning_InterIntra(torch.nn.Module):
    ##################################################################
    def __init__(self, backbone, feature_size=64, nb_class=3, tmp_C = 10):
        #################################################################
        super(RelationalReasoning_InterIntra, self).__init__()
        self.backbone = backbone
        

        self.hidfeatsize = 64
        self.relation_head = torch.nn.Sequential(
                                 torch.nn.Linear(feature_size*2, self.hidfeatsize),
                                 torch.nn.BatchNorm1d(self.hidfeatsize),
                                 torch.nn.LeakyReLU(),
                                 torch.nn.Linear(self.hidfeatsize, 1))
        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(feature_size*2, self.hidfeatsize),
            torch.nn.BatchNorm1d(self.hidfeatsize),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.hidfeatsize, nb_class),
            torch.nn.Softmax(),
        )
        ###############################my change##############################
        self.tmp_cls_head = torch.nn.Sequential(
#             torch.nn.Linear(feature_size, 128),
#             torch.nn.BatchNorm1d(128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(feature_size, tmp_C), #128
            torch.nn.Softmax(),
        )
        ######################################################################
        # self.softmax = nn.Softmax()
        
        

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
                pos2 = features[index_2:index_2+size]
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

    def train(self, tot_epochs, train_loader, opt):
        patience = opt.patience
        early_stopping = EarlyStopping(patience, verbose=True,
                                   checkpoint_pth='{}/backbone_best.tar'.format(opt.ckpt_dir))

  
        
        optimizer = torch.optim.Adam([
                      {'params': self.backbone.parameters()},
                      {'params': self.relation_head.parameters()},
            {'params': self.cls_head.parameters()},  {'params': self.tmp_cls_head.parameters()}   ], lr=opt.learning_rate)
        BCE = torch.nn.BCEWithLogitsLoss()
        c_criterion = nn.CrossEntropyLoss()

        self.backbone.train()
        self.relation_head.train()
        self.cls_head.train()
        ###############################################
        self.tmp_cls_head.train()
        ###############################################
        epoch_max = 0
        acc_max=0
        
        ################################################################
        file2print_detail_test = open(opt.logfile, 'a+')
        ################################################################
        
        for epoch in range(tot_epochs):   
            
            
            #########################=WILLL BE DELETED LATER, GPU LIMITED!########################################
#             if epoch==0:            
#                 Tmp_checkpoint_pth='{}/backbone_best_p0.tar'.format(opt.ckpt_dir)
#                 print('Tmp_checkpoint_pth:',Tmp_checkpoint_pth)
#                 checkpoint = torch.load(Tmp_checkpoint_pth, map_location='cpu')
#                 self.backbone.load_state_dict(checkpoint)
            #########################=WILLL BE DELETED LATER, GPU LIMITED!########################################
      

            acc_epoch=0
            acc_epoch_cls=0
            ###############################################
            tmp_acc_epoch_cls = 0
            ###############################################
            loss_epoch=0
          # the real target is discarded (unsupervised)
            for i, (data, data_augmented0, data_augmented1, data_label, tmp_target) in enumerate(train_loader):
                K = len(data) # tot augmentations
#                 print('i,K:',i,K)
                x = torch.cat(data, 0).to(device)
        
                x_cut0 = torch.cat(data_augmented0, 0).to(device)
                x_cut1 = torch.cat(data_augmented1, 0).to(device)
                c_label = torch.cat(data_label, 0).to(device)

                #######################################################
#                 print('data,x:',i,x.shape,(data[0].shape))
#                 print('x:',x.shape)
                #######################################################
    
                optimizer.zero_grad()
                # forward pass (backbone)
#                 N = data[0][1]  ### the lenght of the input data
                features = self.backbone(x)
#                 N = data_augmented0[0][1]  ###the lenght of the cutted input data
                features_cut0 = self.backbone(x_cut0)
                features_cut1 = self.backbone(x_cut1)

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
#                 aug_data = torch.cat(data, 0).to(device)
#                 tmp_features = self.backbone(aug_data)
#                 tmp_c_output = self.tmp_cls_head(tmp_features)
                tmp_c_output = self.tmp_cls_head(features)
                tmp_label = torch.cat(tmp_target, 0).to(device)
                tmp_correct_cls, tmp_length_cls = self.run_test(tmp_c_output, tmp_label)
                #######################################################

                # cross-entropy loss and backward
                loss = BCE(score, targets)
                loss_c = c_criterion(c_output, c_label)
#                 loss+=loss_c
                #######################################################
                tmp_loss_c = c_criterion(tmp_c_output, tmp_label)
                loss+= loss_c + tmp_loss_c
                #######################################################

                loss.backward()
                optimizer.step()
                # estimate the accuracy
                predicted = torch.round(torch.sigmoid(score))
                correct = predicted.eq(targets.view_as(predicted)).sum()
                accuracy = (100.0 * correct / float(len(targets)))
                acc_epoch += accuracy.item()
                loss_epoch += loss.item()

                accuracy_cls = 100. * correct_cls / length_cls
                acc_epoch_cls += accuracy_cls.item()
                
                #######################################################
                tmp_accuracy_cls = 100. * tmp_correct_cls / tmp_length_cls
                tmp_acc_epoch_cls += tmp_accuracy_cls.item()
                #######################################################
                
                

            acc_epoch /= len(train_loader)
            acc_epoch_cls /= len(train_loader)
            loss_epoch /= len(train_loader)
            
            #######################################################
            tmp_acc_epoch_cls /= len(train_loader)
            #######################################################
            

            if (acc_epoch+acc_epoch_cls + tmp_acc_epoch_cls)>acc_max:
                acc_max = (acc_epoch+acc_epoch_cls+ tmp_acc_epoch_cls)
                epoch_max = epoch

            ######################################################################
            early_stopping((acc_epoch+acc_epoch_cls +tmp_acc_epoch_cls), self.backbone)
            ######################################################################
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if (epoch+1)%opt.save_freq==0:
                print("[INFO] save backbone at epoch {}!".format(epoch))
                torch.save(self.backbone.state_dict(), '{}/backbone_{}.tar'.format(opt.ckpt_dir, epoch))

            print('Epoch [{}][{}][{}] loss= {:.5f}; Epoch ACC.= {:.2f}%, CLS.= {:.2f}%, Tmp_CLS.= {:.2f}%,'
                'Max ACC.= {:.1f}%, Max Epoch={}' \
                .format(epoch + 1, opt.model_name, opt.dataset_name,
                        loss_epoch, acc_epoch,acc_epoch_cls, tmp_acc_epoch_cls, acc_max, epoch_max))
            

            ######################################################################
            print('Epoch [{}][{}][{}] loss= {:.5f}; Epoch ACC.= {:.2f}%, CLS.= {:.2f}%, Tmp_CLS.= {:.2f}%,'
                'Max ACC.= {:.1f}%, Max Epoch={}' \
                .format(epoch + 1, opt.model_name, opt.dataset_name,
                        loss_epoch, acc_epoch,acc_epoch_cls, tmp_acc_epoch_cls, acc_max, epoch_max), file=file2print_detail_test)         
            file2print_detail_test.flush()
            ######################################################################
    
        return acc_max, epoch_max




    
    
class RelationalReasoning_InterIntra_Tmp(torch.nn.Module):
    ##################################################################
    def __init__(self, backbone, feature_size=64, nb_class=3, tmp_C = 10):
        #################################################################
        super(RelationalReasoning_InterIntra_Tmp, self).__init__()
        self.backbone = backbone
        

        self.hidfeatsize = 64
        self.relation_head = torch.nn.Sequential(
                                 torch.nn.Linear(feature_size*2, self.hidfeatsize),
                                 torch.nn.BatchNorm1d(self.hidfeatsize),
                                 torch.nn.LeakyReLU(),
                                 torch.nn.Linear(self.hidfeatsize, 1)
        )
        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(feature_size*2, self.hidfeatsize),
            torch.nn.BatchNorm1d(self.hidfeatsize),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.hidfeatsize, nb_class),
            torch.nn.Softmax(),
        )
#         self.relation_head = torch.nn.Sequential(
# #             torch.nn.BatchNorm1d(feature_size*2),
#             torch.nn.ReLU(),
#             torch.nn.Linear(feature_size*2, 1)
#         )
#         self.cls_head = torch.nn.Sequential(
# #             torch.nn.BatchNorm1d(feature_size*2),
#             torch.nn.ReLU(),
#             torch.nn.Linear(feature_size*2, nb_class),
#             torch.nn.Softmax(),
#         )
        ###############################my change##############################
        self.tmp_cls_head = torch.nn.Sequential(
#             torch.nn.Linear(feature_size, 128),
#             torch.nn.BatchNorm1d(128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(feature_size, tmp_C), #128
            torch.nn.Softmax(),
        )
        ######################################################################
        # self.softmax = nn.Softmax()
        
        

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
                pos2 = features[index_2:index_2+size]
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

    def train(self, tot_epochs, train_loader, opt):
        patience = opt.patience
        early_stopping = EarlyStopping(patience, verbose=True,
                                   checkpoint_pth='{}/backbone_best.tar'.format(opt.ckpt_dir))

  
        
        optimizer = torch.optim.Adam([
                      {'params': self.backbone.parameters()},
                      {'params': self.relation_head.parameters()},
            {'params': self.cls_head.parameters()},  {'params': self.tmp_cls_head.parameters()}   ], lr=opt.learning_rate)
        BCE = torch.nn.BCEWithLogitsLoss()
        c_criterion = nn.CrossEntropyLoss()

        self.backbone.train()
        self.relation_head.train()
        self.cls_head.train()
        ###############################################
        self.tmp_cls_head.train()
        ###############################################
        epoch_max = 0
        acc_max=0
        
        ################################################################
        file2print_detail_test = open(opt.logfile, 'a+')
        ################################################################
        
        list_acc_epoch=  list() 
        list_acc_epoch_cls =  list() 
        list_tmp_acc_epoch_cls =  list() 
        
        list_loss_epoch=  list() 
        list_loss_c =  list() 
        list_loss_c_tmp =  list() 
        
        for epoch in range(tot_epochs):   
            
            
            #########################=WILLL BE DELETED LATER, GPU LIMITED!########################################
#             if epoch==0:            
#                 Tmp_checkpoint_pth='{}/backbone_best_p0.tar'.format(opt.ckpt_dir)
#                 print('Tmp_checkpoint_pth:',Tmp_checkpoint_pth)
#                 checkpoint = torch.load(Tmp_checkpoint_pth, map_location='cpu')
#                 self.backbone.load_state_dict(checkpoint)
            #########################=WILLL BE DELETED LATER, GPU LIMITED!########################################
      

            acc_epoch=0
            acc_epoch_cls=0
            ###############################################
            tmp_acc_epoch_cls = 0
            ###############################################
            loss_epoch=0
            
            loss_inter_epoch=0
            loss_c_epoch=0
            loss_c_tmp_epoch=0
            
          # the real target is discarded (unsupervised)
            for i, (data, data_augmented0, data_augmented1, data_label, tmp_target) in enumerate(train_loader):
                K = len(data) # tot augmentations
#                 print('i,K:',i,K)
                x = torch.cat(data, 0).to(device)
        
                x_cut0 = torch.cat(data_augmented0, 0).to(device)
                x_cut1 = torch.cat(data_augmented1, 0).to(device)
                c_label = torch.cat(data_label, 0).to(device)

                #######################################################
#                 print('data,x:',i,x.shape,(data[0].shape))
#                 print('x:',x.shape)
                #######################################################
    
                optimizer.zero_grad()
                # forward pass (backbone)
#                 N = data[0][1]  ### the lenght of the input data
                features = self.backbone(x)
#                 N = data_augmented0[0][1]  ###the lenght of the cutted input data
                features_cut0 = self.backbone(x_cut0)
                features_cut1 = self.backbone(x_cut1)

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
#                 aug_data = torch.cat(data, 0).to(device)
#                 tmp_features = self.backbone(aug_data)
#                 tmp_c_output = self.tmp_cls_head(tmp_features)
                tmp_c_output = self.tmp_cls_head(features)
                tmp_label = torch.cat(tmp_target, 0).to(device)
                tmp_correct_cls, tmp_length_cls = self.run_test(tmp_c_output, tmp_label)
                #######################################################

                # cross-entropy loss and backward
                loss_inter = BCE(score, targets)
                loss_c = c_criterion(c_output, c_label)
#                 loss+=loss_c
                #######################################################
                tmp_loss_c = c_criterion(tmp_c_output, tmp_label)
                loss = loss_inter + loss_c + tmp_loss_c
                #######################################################

                loss.backward()
                optimizer.step()
                # estimate the accuracy
                predicted = torch.round(torch.sigmoid(score))
                correct = predicted.eq(targets.view_as(predicted)).sum()
                accuracy = (100.0 * correct / float(len(targets)))
                acc_epoch += accuracy.item()
                loss_epoch += loss.item()

                accuracy_cls = 100. * correct_cls / length_cls
                acc_epoch_cls += accuracy_cls.item()
                
                #######################################################
                tmp_accuracy_cls = 100. * tmp_correct_cls / tmp_length_cls
                tmp_acc_epoch_cls += tmp_accuracy_cls.item()
                #######################################################
                
                loss_inter_epoch += loss_inter.item()
                loss_c_epoch += loss_c.item()
                loss_c_tmp_epoch += tmp_loss_c.item()

            acc_epoch /= len(train_loader)
            acc_epoch_cls /= len(train_loader)
            loss_epoch /= len(train_loader)
            
            loss_inter_epoch /= len(train_loader)
            loss_c_epoch /= len(train_loader)
            loss_c_tmp_epoch /= len(train_loader)

            #######################################################
            tmp_acc_epoch_cls /= len(train_loader)
            #######################################################
            
            ################===========##trainig progress###########
            list_acc_epoch.append(acc_epoch)
            list_acc_epoch_cls.append(acc_epoch_cls)
            list_tmp_acc_epoch_cls.append(tmp_acc_epoch_cls)
            list_loss_epoch.append(loss_inter_epoch)
            list_loss_c.append(loss_c_epoch)
            list_loss_c_tmp.append(loss_c_tmp_epoch)
            
            Loss_Acc_df = pd.DataFrame({"list_acc_epoch": list_acc_epoch, "list_acc_epoch_cls": list_acc_epoch_cls, "list_tmp_acc_epoch_cls": list_tmp_acc_epoch_cls, "list_loss_epoch": list_loss_epoch, "list_loss_c": list_loss_c, "list_loss_c_tmp": list_loss_c_tmp})  
            Loss_Acc_df.to_pickle("/gpfs/exfel/data/user/sunyue/Spectra_classification_Third_Paper/SelfTime/log/Relation_reason_loss_acc.pkl")  
            ################======================##################

            if (acc_epoch+acc_epoch_cls + tmp_acc_epoch_cls)>acc_max:
                acc_max = (acc_epoch+acc_epoch_cls+ tmp_acc_epoch_cls)
                epoch_max = epoch

            ######################################################################
            early_stopping((acc_epoch+acc_epoch_cls +tmp_acc_epoch_cls), self.backbone)
            ######################################################################
            if early_stopping.early_stop:
                print("Early stopping")
                break

#             if (epoch+1)%opt.save_freq==0:
            if epoch>188 and (epoch+1)%opt.save_freq==0:
                print("[INFO] save backbone at epoch {}!".format(epoch))
                torch.save(self.backbone.state_dict(), '{}/backbone_{}.tar'.format(opt.ckpt_dir, epoch))

            print('Epoch [{}][{}][{}] loss= {:.5f}; Epoch ACC.= {:.2f}%, CLS.= {:.2f}%, Tmp_CLS.= {:.2f}%,'
                'Max ACC.= {:.1f}%, Max Epoch={}' \
                .format(epoch + 1, opt.model_name, opt.dataset_name,
                        loss_epoch, acc_epoch,acc_epoch_cls, tmp_acc_epoch_cls, acc_max, epoch_max))
            

            ######################################################################
            print('Epoch [{}][{}][{}] loss= {:.5f}; Epoch ACC.= {:.2f}%, CLS.= {:.2f}%, Tmp_CLS.= {:.2f}%,'
                'Max ACC.= {:.1f}%, Max Epoch={}' \
                .format(epoch + 1, opt.model_name, opt.dataset_name,
                        loss_epoch, acc_epoch,acc_epoch_cls, tmp_acc_epoch_cls, acc_max, epoch_max), file=file2print_detail_test)         
            file2print_detail_test.flush()
            ######################################################################
    
        return acc_max, epoch_max




    
    
    
    
    
    
    
    
    
    
    
      
class RelationalReasoning_InterIntra_Tmp_C1(torch.nn.Module):
    ##################################################################
    def __init__(self, backbone, feature_size=64, nb_class=3, tmp_C = 10):
        #################################################################
        super(RelationalReasoning_InterIntra_Tmp_C1, self).__init__()
        self.backbone = backbone
        

        self.hidfeatsize = 64
        self.relation_head = torch.nn.Sequential(
                                 torch.nn.Linear(feature_size*2, self.hidfeatsize),
                                 torch.nn.BatchNorm1d(self.hidfeatsize),
                                 torch.nn.LeakyReLU(),
                                 torch.nn.Linear(self.hidfeatsize, 1)
        )
        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(feature_size*2, self.hidfeatsize),
            torch.nn.BatchNorm1d(self.hidfeatsize),
            torch.nn.LeakyReLU(),
#             torch.nn.ReLU(),
            torch.nn.Linear(self.hidfeatsize, nb_class),
            torch.nn.Softmax(),
        )
#         self.relation_head = torch.nn.Sequential(
# #             torch.nn.BatchNorm1d(feature_size*2),
#             torch.nn.ReLU(),
#             torch.nn.Linear(feature_size*2, 1)
#         )
#         self.cls_head = torch.nn.Sequential(
# #             torch.nn.BatchNorm1d(feature_size*2),
#             torch.nn.ReLU(),
#             torch.nn.Linear(feature_size*2, nb_class),
#             torch.nn.Softmax(),
#         )
        ############################### my change ##############################
        self.tmp_cls_head = torch.nn.Sequential(
            torch.nn.Linear(feature_size*2, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.LeakyReLU(),
#             torch.nn.ReLU(),
            torch.nn.Linear(128, tmp_C), #128
            torch.nn.Softmax(),
        )
        ######################################################################
        # self.softmax = nn.Softmax()
        

    def aggregate_Tmp(self, features, K, tmp_target):
        relation_pairs_list = list()
        targets_list = list()
        size = int(features.shape[0] / K)
        tmp_target_1 = tmp_target[0]
        #########################
#         print('tmp_target_1:',tmp_target_1)

        shifts_counter = 0
        for index_1 in range(0, size*K, size):
            for index_2 in range(index_1+size, size*3, size):
                # Using the 'cat' aggregation function by default
                pos1 = features[index_1:index_1 + size]
                # Shuffle without collisions by rolling the mini-batch (negatives)
                neg1 = torch.roll(features[index_2:index_2 + size],
                                  shifts = shifts_counter, dims=0)
                neg_pair1 = torch.cat([pos1, neg1], 1) # (batch_size, fz*2)

                relation_pairs_list.append(neg_pair1)
                
                tmp_target_2 = torch.roll(tmp_target_1, shifts = shifts_counter, dims=0)
                ####################################
#                 print('tmp_target_2:',tmp_target_2)
                
                tmp_label = torch.abs(tmp_target_1 - tmp_target_2)
                ####################################
#                 print('tmp_label:',tmp_label)
                
                targets_list.append(tmp_label.to(device))
                ####################################
#                 print('targets_list:',targets_list)
                
                shifts_counter+=1
                if(shifts_counter >= size):
                    shifts_counter = 1 # avoid identity pairs
                    
                ####################################
#                 print('shifts_counter:',shifts_counter)
        relation_pairs = torch.cat(relation_pairs_list, 0).to(device)  # K(K-1) * (batch_size, fz*2)
        targets = torch.cat(targets_list, 0).to(device)
        return relation_pairs, targets
    

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

    def train(self, tot_epochs, train_loader, opt):
        patience = opt.patience
        early_stopping = EarlyStopping(patience, verbose=True,
                                   checkpoint_pth='{}/backbone_best.tar'.format(opt.ckpt_dir))

  
        
        optimizer = torch.optim.Adam([
                      {'params': self.backbone.parameters()},
                      {'params': self.relation_head.parameters()},
            {'params': self.cls_head.parameters()},  {'params': self.tmp_cls_head.parameters()}   ], lr=opt.learning_rate)
        BCE = torch.nn.BCEWithLogitsLoss()
        c_criterion = nn.CrossEntropyLoss()

        self.backbone.train()
        self.relation_head.train()
        self.cls_head.train()
        ###############################################
        self.tmp_cls_head.train()
        ###############################################
        epoch_max = 0
        acc_max=0
        
        ################################################################
        file2print_detail_test = open(opt.logfile, 'a+')
        ################################################################
        
        list_acc_epoch=  list()
        list_acc_epoch_cls =  list()
        list_tmp_acc_epoch_cls =  list()
        
        list_loss_epoch=  list() 
        list_loss_c =  list() 
        list_loss_c_tmp =  list() 
        
        for epoch in range(tot_epochs):   
            
            
            #########################=WILLL BE DELETED LATER, GPU LIMITED!########################################
#             if epoch==0:            
#                 Tmp_checkpoint_pth='{}/backbone_best_p0.tar'.format(opt.ckpt_dir)
#                 print('Tmp_checkpoint_pth:',Tmp_checkpoint_pth)
#                 checkpoint = torch.load(Tmp_checkpoint_pth, map_location='cpu')
#                 self.backbone.load_state_dict(checkpoint)
            #########################=WILLL BE DELETED LATER, GPU LIMITED!########################################
      

            acc_epoch=0
            acc_epoch_cls=0
            ###############################################
            tmp_acc_epoch_cls = 0
            ###############################################
            loss_epoch=0
            
            loss_inter_epoch=0
            loss_c_epoch=0
            loss_c_tmp_epoch=0
            
          # the real target is discarded (unsupervised)
            for i, (data, data_augmented0, data_augmented1, data_label, tmp_target) in enumerate(train_loader):
                K = len(data) # tot augmentations
#                 print('i,K:',i,K)
                x = torch.cat(data, 0).to(device)
        
                x_cut0 = torch.cat(data_augmented0, 0).to(device)
                x_cut1 = torch.cat(data_augmented1, 0).to(device)
                c_label = torch.cat(data_label, 0).to(device)

                #######################################################
#                 print('data,x:',i,x.shape,(data[0].shape))
#                 print('x:',x.shape)
                #######################################################
    
                optimizer.zero_grad()
                # forward pass (backbone)
#                 N = data[0][1]  ### the lenght of the input data
                features = self.backbone(x)
#                 N = data_augmented0[0][1]  ###the lenght of the cutted input data
                features_cut0 = self.backbone(x_cut0)
                features_cut1 = self.backbone(x_cut1)
#                 print('x_cut0,features_cut0',x_cut0.shape, features_cut0.shape)
                
                features_cls = torch.cat([features_cut0, features_cut1], 1)
#                 print('features_cut0,features_cls',features_cut0.shape, features_cls.shape)
                
    
                # aggregation function
                relation_pairs, targets = self.aggregate(features, K)
#                 print('relation_pairs:',relation_pairs.shape)
        
                score = self.relation_head(relation_pairs).squeeze()
                c_output = self.cls_head(features_cls)
                correct_cls, length_cls = self.run_test(c_output, c_label)
                
                
                
                #######################################################
                # aggregation function
                relation_pairs_Tmp, targets_Tmp = self.aggregate_Tmp(features, K, tmp_target)
#                 print('relation_pairs_Tmp:',relation_pairs_Tmp.shape)
                tmp_c_output = self.tmp_cls_head(relation_pairs_Tmp)
#                 tmp_label = torch.cat(targets_Tmp, 0).to(device)
                tmp_label = targets_Tmp.to(device)
                tmp_correct_cls, tmp_length_cls = self.run_test(tmp_c_output, tmp_label)
                #######################################################
                

                # cross-entropy loss and backward
                loss_inter = BCE(score, targets)
                loss_c = c_criterion(c_output, c_label)
#                 loss+=loss_c
                #######################################################
                tmp_loss_c = c_criterion(tmp_c_output, tmp_label)
                loss = loss_inter + loss_c + tmp_loss_c
                #######################################################

                loss.backward()
                optimizer.step()
                # estimate the accuracy
                predicted = torch.round(torch.sigmoid(score))
                correct = predicted.eq(targets.view_as(predicted)).sum()
                accuracy = (100.0 * correct / float(len(targets)))
                acc_epoch += accuracy.item()
                loss_epoch += loss.item()

                accuracy_cls = 100. * correct_cls / length_cls
                acc_epoch_cls += accuracy_cls.item()
                
                #######################################################
                tmp_accuracy_cls = 100. * tmp_correct_cls / tmp_length_cls
                tmp_acc_epoch_cls += tmp_accuracy_cls.item()
                #######################################################
                
                loss_inter_epoch += loss_inter.item()
                loss_c_epoch += loss_c.item()
                loss_c_tmp_epoch += tmp_loss_c.item()

            acc_epoch /= len(train_loader)
            acc_epoch_cls /= len(train_loader)
            loss_epoch /= len(train_loader)
            
            loss_inter_epoch /= len(train_loader)
            loss_c_epoch /= len(train_loader)
            loss_c_tmp_epoch /= len(train_loader)

            #######################################################
            tmp_acc_epoch_cls /= len(train_loader)
            #######################################################
            
            ################===========##trainig progress###########
            list_acc_epoch.append(acc_epoch)
            list_acc_epoch_cls.append(acc_epoch_cls)
            list_tmp_acc_epoch_cls.append(tmp_acc_epoch_cls)
            list_loss_epoch.append(loss_inter_epoch)
            list_loss_c.append(loss_c_epoch)
            list_loss_c_tmp.append(loss_c_tmp_epoch)
            
            Loss_Acc_df = pd.DataFrame({"list_acc_epoch": list_acc_epoch, "list_acc_epoch_cls": list_acc_epoch_cls, "list_tmp_acc_epoch_cls": list_tmp_acc_epoch_cls, "list_loss_epoch": list_loss_epoch, "list_loss_c": list_loss_c, "list_loss_c_tmp": list_loss_c_tmp})  
            Loss_Acc_df.to_pickle("/gpfs/exfel/data/user/sunyue/Spectra_classification_Third_Paper/SelfTime/log/Relation_reason_loss_acc_Tmp.pkl")  
            ################======================##################

            if (acc_epoch+acc_epoch_cls + tmp_acc_epoch_cls)>acc_max:
                acc_max = (acc_epoch+acc_epoch_cls+ tmp_acc_epoch_cls)
                epoch_max = epoch

            ######################################################################
            early_stopping((acc_epoch+acc_epoch_cls +tmp_acc_epoch_cls), self.backbone)
            ######################################################################
            if early_stopping.early_stop:
                print("Early stopping")
                break

#             if (epoch+1)%opt.save_freq==0:
            if epoch>188 and (epoch+1)%opt.save_freq==0:
                print("[INFO] save backbone at epoch {}!".format(epoch))
                torch.save(self.backbone.state_dict(), '{}/backbone_{}.tar'.format(opt.ckpt_dir, epoch))

            torch.save(self.backbone.state_dict(), '{}/backbone_last.tar'.format(opt.ckpt_dir))
                
            print('Epoch [{}][{}][{}] loss= {:.5f}; Epoch ACC.= {:.2f}%, CLS.= {:.2f}%, Tmp_CLS.= {:.2f}%,'
                'Max ACC.= {:.1f}%, Max Epoch={}' \
                .format(epoch + 1, opt.model_name, opt.dataset_name,
                        loss_epoch, acc_epoch,acc_epoch_cls, tmp_acc_epoch_cls, acc_max, epoch_max))
            

            ######################################################################
            print('Epoch [{}][{}][{}] loss= {:.5f}; Epoch ACC.= {:.2f}%, CLS.= {:.2f}%, Tmp_CLS.= {:.2f}%,'
                'Max ACC.= {:.1f}%, Max Epoch={}' \
                .format(epoch + 1, opt.model_name, opt.dataset_name,
                        loss_epoch, acc_epoch,acc_epoch_cls, tmp_acc_epoch_cls, acc_max, epoch_max), file=file2print_detail_test)         
            file2print_detail_test.flush()
            ######################################################################
    
        return acc_max, epoch_max



    
    
        
    
class RelationalReasoning_InterIntra_Tmp_20230115(torch.nn.Module):
    ##################################################################
    def __init__(self, backbone, feature_size=64, nb_class=3, tmp_C = 10):
        #################################################################
        super(RelationalReasoning_InterIntra_Tmp_20230115, self).__init__()
        self.backbone = backbone
        

        self.hidfeatsize = 64
        self.relation_head = torch.nn.Sequential(
                                 torch.nn.Linear(feature_size*2, self.hidfeatsize),
                                 torch.nn.BatchNorm1d(self.hidfeatsize),
                                 torch.nn.LeakyReLU(),
                                 torch.nn.Linear(self.hidfeatsize, 1)
        )
        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(feature_size*2, self.hidfeatsize),
            torch.nn.BatchNorm1d(self.hidfeatsize),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.hidfeatsize, nb_class),
            torch.nn.Softmax(),
        )
#         self.relation_head = torch.nn.Sequential(
# #             torch.nn.BatchNorm1d(feature_size*2),
#             torch.nn.ReLU(),
#             torch.nn.Linear(feature_size*2, 1)
#         )
#         self.cls_head = torch.nn.Sequential(
# #             torch.nn.BatchNorm1d(feature_size*2),
#             torch.nn.ReLU(),
#             torch.nn.Linear(feature_size*2, nb_class),
#             torch.nn.Softmax(),
#         )
        ###############################my change##############################
        self.tmp_cls_head = torch.nn.Sequential(
#             torch.nn.Linear(feature_size, 128),
#             torch.nn.BatchNorm1d(128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(feature_size, tmp_C), #128
            torch.nn.Softmax(),
        )
        ######################################################################
        # self.softmax = nn.Softmax()
        
        

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
                pos2 = features[index_2:index_2+size]
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

    def train(self, tot_epochs, train_loader, opt):
        patience = opt.patience
        early_stopping = EarlyStopping(patience, verbose=True,
                                   checkpoint_pth='{}/backbone_best.tar'.format(opt.ckpt_dir))

  
        
        optimizer = torch.optim.Adam([
                      {'params': self.backbone.parameters()},
                      {'params': self.relation_head.parameters()},
            {'params': self.cls_head.parameters()},  {'params': self.tmp_cls_head.parameters()}   ], lr=opt.learning_rate)
        BCE = torch.nn.BCEWithLogitsLoss()
        c_criterion = nn.CrossEntropyLoss()

        self.backbone.train()
        self.relation_head.train()
        self.cls_head.train()
        ###############################################
        self.tmp_cls_head.train()
        ###############################################
        epoch_max = 0
        acc_max=0
        
        ################################################################
        file2print_detail_test = open(opt.logfile, 'a+')
        ################################################################
        
        list_acc_epoch=  list() 
        list_acc_epoch_cls =  list() 
        list_tmp_acc_epoch_cls =  list() 
        
        list_loss_epoch=  list() 
        list_loss_c =  list() 
        list_loss_c_tmp =  list() 
        
        for epoch in range(tot_epochs):   
            

            adjust_learning_rate(opt, optimizer, epoch)            
            #########################=WILLL BE DELETED LATER, GPU LIMITED!########################################
#             if epoch==0:            
#                 Tmp_checkpoint_pth='{}/backbone_best_p0.tar'.format(opt.ckpt_dir)
#                 print('Tmp_checkpoint_pth:',Tmp_checkpoint_pth)
#                 checkpoint = torch.load(Tmp_checkpoint_pth, map_location='cpu')
#                 self.backbone.load_state_dict(checkpoint)
            #########################=WILLL BE DELETED LATER, GPU LIMITED!########################################
      

            acc_epoch=0
            acc_epoch_cls=0
            ###############################################
            tmp_acc_epoch_cls = 0
            ###############################################
            loss_epoch=0
            
            loss_inter_epoch=0
            loss_c_epoch=0
            loss_c_tmp_epoch=0
            
          # the real target is discarded (unsupervised)
            for i, (data, data_augmented0, data_augmented1, data_label, tmp_target) in enumerate(train_loader):
                K = len(data) # tot augmentations
#                 print('i,K:',i,K)
                x = torch.cat(data, 0).to(device)
        
                x_cut0 = torch.cat(data_augmented0, 0).to(device)
                x_cut1 = torch.cat(data_augmented1, 0).to(device)
                c_label = torch.cat(data_label, 0).to(device)

            
                #######################################################
                warmup_learning_rate(opt, epoch, i, len(train_loader), optimizer)
#                 print('data,x:',i,x.shape,(data[0].shape))
#                 print('x:',x.shape)
                #######################################################
    
                optimizer.zero_grad()
                # forward pass (backbone)
#                 N = data[0][1]  ### the lenght of the input data
                features = self.backbone(x)
#                 N = data_augmented0[0][1]  ###the lenght of the cutted input data
                features_cut0 = self.backbone(x_cut0)
                features_cut1 = self.backbone(x_cut1)

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
#                 aug_data = torch.cat(data, 0).to(device)
#                 tmp_features = self.backbone(aug_data)
#                 tmp_c_output = self.tmp_cls_head(tmp_features)
                tmp_c_output = self.tmp_cls_head(features)
                tmp_label = torch.cat(tmp_target, 0).to(device)
                tmp_correct_cls, tmp_length_cls = self.run_test(tmp_c_output, tmp_label)
                #######################################################

                # cross-entropy loss and backward
                loss_inter = BCE(score, targets)
                loss_c = c_criterion(c_output, c_label)
#                 loss+=loss_c
                #######################################################
                tmp_loss_c = c_criterion(tmp_c_output, tmp_label)
                loss = loss_inter + loss_c + tmp_loss_c
                #######################################################

                loss.backward()
                optimizer.step()
                # estimate the accuracy
                predicted = torch.round(torch.sigmoid(score))
                correct = predicted.eq(targets.view_as(predicted)).sum()
                accuracy = (100.0 * correct / float(len(targets)))
                acc_epoch += accuracy.item()
                loss_epoch += loss.item()

                accuracy_cls = 100. * correct_cls / length_cls
                acc_epoch_cls += accuracy_cls.item()
                
                #######################################################
                tmp_accuracy_cls = 100. * tmp_correct_cls / tmp_length_cls
                tmp_acc_epoch_cls += tmp_accuracy_cls.item()
                #######################################################
                
                loss_inter_epoch += loss_inter.item()
                loss_c_epoch += loss_c.item()
                loss_c_tmp_epoch += tmp_loss_c.item()

            acc_epoch /= len(train_loader)
            acc_epoch_cls /= len(train_loader)
            loss_epoch /= len(train_loader)
            
            loss_inter_epoch /= len(train_loader)
            loss_c_epoch /= len(train_loader)
            loss_c_tmp_epoch /= len(train_loader)

            #######################################################
            tmp_acc_epoch_cls /= len(train_loader)
            #######################################################
            
            ################===========##trainig progress###########
            list_acc_epoch.append(acc_epoch)
            list_acc_epoch_cls.append(acc_epoch_cls)
            list_tmp_acc_epoch_cls.append(tmp_acc_epoch_cls)
            list_loss_epoch.append(loss_inter_epoch)
            list_loss_c.append(loss_c_epoch)
            list_loss_c_tmp.append(loss_c_tmp_epoch)
            
            Loss_Acc_df = pd.DataFrame({"list_acc_epoch": list_acc_epoch, "list_acc_epoch_cls": list_acc_epoch_cls, "list_tmp_acc_epoch_cls": list_tmp_acc_epoch_cls, "list_loss_epoch": list_loss_epoch, "list_loss_c": list_loss_c, "list_loss_c_tmp": list_loss_c_tmp})  
            Loss_Acc_df.to_pickle("/gpfs/exfel/data/user/sunyue/Spectra_classification_Third_Paper/SelfTime/log/Relation_reason_loss_acc.pkl")  
            ################======================##################

            if (acc_epoch+acc_epoch_cls + tmp_acc_epoch_cls)>acc_max:
                acc_max = (acc_epoch+acc_epoch_cls+ tmp_acc_epoch_cls)
                epoch_max = epoch

            ######################################################################
            early_stopping((acc_epoch+acc_epoch_cls +tmp_acc_epoch_cls), self.backbone)
            ######################################################################
            if early_stopping.early_stop:
                print("Early stopping")
                break

#             if (epoch+1)%opt.save_freq==0:
            if epoch>188 and (epoch+1)%opt.save_freq==0:
                print("[INFO] save backbone at epoch {}!".format(epoch))
                torch.save(self.backbone.state_dict(), '{}/backbone_{}.tar'.format(opt.ckpt_dir, epoch))

            print('Epoch [{}][{}][{}] loss= {:.5f}; Epoch ACC.= {:.2f}%, CLS.= {:.2f}%, Tmp_CLS.= {:.2f}%,'
                'Max ACC.= {:.1f}%, Max Epoch={}' \
                .format(epoch + 1, opt.model_name, opt.dataset_name,
                        loss_epoch, acc_epoch,acc_epoch_cls, tmp_acc_epoch_cls, acc_max, epoch_max))
            

            ######################################################################
            print('Epoch [{}][{}][{}] loss= {:.5f}; Epoch ACC.= {:.2f}%, CLS.= {:.2f}%, Tmp_CLS.= {:.2f}%,'
                'Max ACC.= {:.1f}%, Max Epoch={}' \
                .format(epoch + 1, opt.model_name, opt.dataset_name,
                        loss_epoch, acc_epoch,acc_epoch_cls, tmp_acc_epoch_cls, acc_max, epoch_max), file=file2print_detail_test)         
            file2print_detail_test.flush()
            ######################################################################
    
        return acc_max, epoch_max


