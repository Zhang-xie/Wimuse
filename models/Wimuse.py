import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch

import pytorch_lightning as pl
import time
from .Basic_Blocks import *
from .STS import STLPL
from .similarity import Pair_metric
from torchinfo import summary

from pytorch_lightning.metrics import ConfusionMatrix


class NaiveCrossEntropy(nn.Module):
    """
    This the implementation of naive crossentropy for two distibutions
    
    NOTE: Compute the crossentropy for a minibatch of probability pairs (p and q).
    """

    def __init__(self, reduction = "mean", dim = 1):
        super().__init__()
        self.reduction =reduction
        self.dim = dim

    def forward(self, p, q):
        crossentropy = torch.sum(- p*torch.log(q),dim= self.dim)
        if self.reduction == "sum":
            return torch.sum(crossentropy)
        elif self.reduction == "mean":
            return torch.mean(crossentropy)
        return torch.mean(crossentropy)



# class ResidualAdaptor(nn.Module):
#     def __init__(self, inplanes, planes, stride=1):
#         super(ResidualAdaptor, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm1d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm1d(planes)
#         self.stride = stride

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#         return out


class ResidualAdaptor(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(ResidualAdaptor, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class KDMTS_RA(nn.Module):
    def __init__(self,inchannel_SE, groups_SE, layers_DE, strides_DE, block_DE, inchannel_DE, inchannel_Cl,
                 num_categories):
        super(KDMTS_RA, self).__init__()
        self.SE = ShallowExtractor(inchannel=inchannel_SE,groups=groups_SE)
        self.DE = DeepExtractor(layers=layers_DE,strides=strides_DE,block=block_DE,inchannel=inchannel_DE)
        self.TSE = torch.nn.ModuleList()
        # if inchannel_SE == 52:
        #     self.combine_layer = nn.AdaptiveAvgPool1d(12)
        # else:
        #     self.combine_layer = nn.AdaptiveAvgPool1d(200)
        self.classifiers = torch.nn.ModuleList()
        for i,element in enumerate(num_categories):
            self.classifiers.append(Classifier(inchannel=inchannel_Cl,num_categories=element))
            self.TSE.append(ResidualAdaptor(inplanes=inchannel_DE, planes=inchannel_Cl, stride=2))

    def forward(self, x):
        shallow_feature = self.SE(x)
        deep_feature = self.DE(shallow_feature)

        # deep_feature_combine = self.combine_layer(deep_feature)
        task_specific_features = []
        for i,element in enumerate(self.TSE):
            task_specific_feature = element(shallow_feature)
            # specific_feature = self.combine_layer(task_specific_feature)
            task_specific_features.append(torch.cat((deep_feature,task_specific_feature),dim=2))

        predictions = []
        for i,element in enumerate(self.classifiers):
            predictions.append(element(task_specific_features[i]))
        return predictions,shallow_feature,deep_feature


class KDMTS_RA_PL(pl.LightningModule):
    def __init__(self,inchannel,group,num_categories,task_index,checkpoint_path,lamda):
        # task_inex: 1-act/gesture ; 2-location; 3-user; 4-room; 5-orientation;
        # num_categories and task_index are two list in this model.
        super().__init__()
        self.task_index = task_index
        self.model = KDMTS_RA(inchannel_SE=inchannel, groups_SE=group, layers_DE=[1,2], strides_DE=[1,2],block_DE=BasicBlock,
                         inchannel_DE=128*group, inchannel_Cl=128*group,num_categories=num_categories)
        self.num_categories = num_categories
        self.ComfusionMatrix_alltasks = []  # this list storage the comfusion matrices of all tasks at every epoch. 
                                            # the shape of this list for two task scenario is :[task1_epoch0, task2_epoch0, task1_epoch1, task2_epoch2 ...]
        self.lamda = lamda

        self.teacher_models = torch.nn.ModuleList()
        self.linear_transfers = torch.nn.ModuleList()
        for i,element in enumerate(num_categories):
            state_dict = torch.load(checkpoint_path[i])
            teacher_model = STLPL(inchannel=inchannel,group=group,num_categories=element,task_index=task_index[i])
            teacher_model.load_state_dict(state_dict)
            teacher_model.eval()
            self.teacher_models.append(teacher_model)
            self.linear_transfers.append(conv1x1(in_planes=128*group, out_planes=128*group))

        self.euclidean = Pair_metric('Euclidean', average=False, norm_flag=True)
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = nn.ModuleList()
        self.validation_acc = nn.ModuleList()
        self.test_acc = nn.ModuleList()
        for i in range(len(task_index)):
            temp = []
            self.ComfusionMatrix_alltasks.append(temp)
            self.train_acc.append(pl.metrics.Accuracy())
            self.validation_acc.append(pl.metrics.Accuracy())
            self.test_acc.append(pl.metrics.Accuracy())
        
    def forward(self, x):
        predictions,shallow_feature,deep_feature = self.model(x)
        deep_feature_transformed = []
        deep_feature_teacher = []
        for i, element in enumerate(self.teacher_models):
            predictions_te, shallow_feature_te, deep_feature_te = element(x)
            deep_feature_teacher.append(deep_feature_te)
            deep_feature_transformed.append(self.linear_transfers[i](deep_feature))
        return predictions,deep_feature_transformed,deep_feature_teacher

    def training_step(self, batch, batch_idx):
        data = batch[0]
        labels = [batch[i] for i in self.task_index]
        predictions,deep_feature_transformed,deep_feature_teacher = self(data)
        losses = []
        for i in range(len(predictions)):
            temp = self.criterion(predictions[i],labels[i].long().squeeze())
            losses.append(temp)
            task_name = 'Train_loss' + str(self.task_index[i])
            self.log(task_name, temp)
            self.train_acc[i](predictions[i],labels[i].long().squeeze())
        for i in range(len(deep_feature_transformed)):
            losses.append(self.lamda *self.euclidean(deep_feature_transformed[i],deep_feature_teacher[i]))
        loss = 0
        for i in range(len(losses)):
            loss += losses[i]
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch[0]
        labels = [batch[i] for i in self.task_index]
        predictions, deep_feature_transformed, deep_feature_teacher = self(data)
        losses = []
        for i in range(len(predictions)):
            temp = self.criterion(predictions[i], labels[i].long().squeeze())
            losses.append(temp)
            task_name = 'Validation_loss' + str(self.task_index[i])
            self.log(task_name, temp)
            self.validation_acc[i](predictions[i], labels[i].long().squeeze())
        for i in range(len(deep_feature_transformed)):
            losses.append(self.lamda *self.euclidean(deep_feature_transformed[i],deep_feature_teacher[i]))
        loss = 0
        for i in range(len(losses)):
            loss += losses[i]
        return loss

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        data = batch[0]
        labels = [batch[i] for i in self.task_index]
        predictions, deep_feature_transformed, deep_feature_teacher = self(data)
        losses = []
        loss = 0
        for i in range(len(predictions)):
            temp = self.criterion(predictions[i], labels[i].long().squeeze())
            losses.append(temp)
            self.test_acc[i](predictions[i], labels[i].long().squeeze())
        for i in range(len(losses)):
            loss += losses[i]
        return loss/len(losses)

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x for x in outputs]).mean()
        acc = []
        for i in range(len(self.task_index)):
            acc.append(self.test_acc[i].compute())
        return {'avg_test_loss': avg_loss, 'avg_test_acc': acc}

    def on_validation_epoch_start(self):
        self.confmat_metrix = []
        for i, element in enumerate(self.num_categories):
            self.confmat_metrix.append(ConfusionMatrix(num_classes=element))

    def validation_epoch_end(self,val_step_outputs):
        average_acc = 0.0
        for i in range(len(self.task_index)):
            task_name = 'Validation_Acc' + str(self.task_index[i])
            self.log(task_name, self.validation_acc[i].compute())
            self.ComfusionMatrix_alltasks[i].append(self.confmat_metrix[i].compute())
            average_acc += self.validation_acc[i].compute()
        average_acc /= len(self.task_index)
        self.log('Validation_Acc', average_acc)

    def training_epoch_end(self, training_step_outputs):
        average_acc = 0.0
        for i in range(len(self.task_index)):
            task_name = 'Train_Acc' + str(self.task_index[i])
            self.log(task_name, self.train_acc[i].compute())
            average_acc += self.validation_acc[i].compute()
        average_acc /= len(self.task_index)
        self.log('Train_Acc', average_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[100,200,300,350],
                                                         gamma=0.5)
        return [optimizer,], [scheduler,]


# expanding the knowledge distillation to logits distillation
class Wimuse_PL(pl.LightningModule):
    def __init__(self,inchannel,group,num_categories,task_index,checkpoint_path,lamda,temperature):
        # task_inex: 1-act/gesture ; 2-location; 3-user; 4-room; 5-orientation;
        # num_categories and task_index are two list in this model.
        super().__init__()
        self.task_index = task_index
        self.model = KDMTS_RA(inchannel_SE=inchannel, groups_SE=group, layers_DE=[1,2], strides_DE=[1,2],block_DE=BasicBlock,
                         inchannel_DE=128*group, inchannel_Cl=128*group,num_categories=num_categories)
        self.num_categories = num_categories
        self.ComfusionMatrix_alltasks = []  # this list storage the comfusion matrices of all tasks at every epoch. 
                                            # the shape of this list for two task scenario is :[task1:[epoch_0,epoch_1, ...], task2:[epoch_0,epoch_1, ...],]
        self.lamda = lamda
        self.temperature = temperature

        self.teacher_models = torch.nn.ModuleList()
        self.linear_transfers = torch.nn.ModuleList()
        for i,element in enumerate(num_categories):
            state_dict = torch.load(checkpoint_path[i])
            teacher_model = STLPL(inchannel=inchannel,group=group,num_categories=element,task_index=task_index[i])
            teacher_model.load_state_dict(state_dict)
            teacher_model.eval()
            self.teacher_models.append(teacher_model)
            self.linear_transfers.append(conv1x1(in_planes=128*group, out_planes=128*group))

        self.euclidean = Pair_metric('Euclidean', average=False, norm_flag=True)
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.crossentropy = NaiveCrossEntropy(reduction="mean",dim=1)
        
        self.train_acc = nn.ModuleList()
        self.validation_acc = nn.ModuleList()
        self.test_acc = nn.ModuleList()
        for i in range(len(task_index)):
            temp = []
            self.ComfusionMatrix_alltasks.append(temp)
            self.train_acc.append(pl.metrics.Accuracy())
            self.validation_acc.append(pl.metrics.Accuracy())
            self.test_acc.append(pl.metrics.Accuracy())
        
    def forward(self, x):
        predictions,shallow_feature,deep_feature = self.model(x)
        deep_feature_transformed = []
        deep_feature_teacher = []
        predictions_teacher = []
        for i, element in enumerate(self.teacher_models):
            predictions_te, shallow_feature_te, deep_feature_te = element(x)
            predictions_teacher.append(predictions_te)
            deep_feature_teacher.append(deep_feature_te)
            deep_feature_transformed.append(self.linear_transfers[i](deep_feature))
        return predictions,deep_feature_transformed,deep_feature_teacher,predictions_teacher

    def training_step(self, batch, batch_idx):
        data = batch[0]
        labels = [batch[i] for i in self.task_index]
        predictions,deep_feature_transformed,deep_feature_teacher,predictions_teacher = self(data)
        losses = []
        for i in range(len(predictions)):
            temp = self.criterion(predictions[i],labels[i].long().squeeze())
            losses.append(temp)
            
            soft_predictions = self.softmax(predictions[i] / self.temperature)
            soft_predictions_teacher = self.softmax(predictions_teacher[i] / self.temperature)
            temp2 = self.crossentropy(soft_predictions,soft_predictions_teacher)  # calculate the crossentropy of above two distribution as the distillation loss
            losses.append(temp2)

            task_name = 'Train_loss' + str(self.task_index[i])
            self.log(task_name, temp)
            self.train_acc[i](predictions[i],labels[i].long().squeeze())
        for i in range(len(deep_feature_transformed)):
            losses.append(self.lamda *self.euclidean(deep_feature_transformed[i],deep_feature_teacher[i]))
        loss = 0
        for i in range(len(losses)):
            loss += losses[i]
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch[0]
        labels = [batch[i] for i in self.task_index]
        predictions, deep_feature_transformed, deep_feature_teacher, predictions_teacher = self(data)
        losses = []
        for i in range(len(predictions)):
            temp = self.criterion(predictions[i], labels[i].long().squeeze())
            losses.append(temp)

            soft_predictions = self.softmax(predictions[i] / self.temperature)
            soft_predictions_teacher = self.softmax(predictions_teacher[i] / self.temperature)
            temp2 = self.crossentropy(soft_predictions,soft_predictions_teacher)  # calculate the crossentropy of above two distribution as the distillation loss
            losses.append(temp2)

            task_name = 'Validation_loss' + str(self.task_index[i])
            self.log(task_name, temp)
            self.validation_acc[i](predictions[i], labels[i].long().squeeze())
        for i in range(len(deep_feature_transformed)):
            losses.append(self.lamda *self.euclidean(deep_feature_transformed[i],deep_feature_teacher[i]))
        loss = 0
        for i in range(len(losses)):
            loss += losses[i]
        return loss

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        data = batch[0]
        labels = [batch[i] for i in self.task_index]
        predictions, deep_feature_transformed, deep_feature_teacher, predictions_teacher = self(data)
        losses = []
        loss = 0
        for i in range(len(predictions)):
            temp = self.criterion(predictions[i], labels[i].long().squeeze())
            losses.append(temp)
            self.test_acc[i](predictions[i], labels[i].long().squeeze())
        for i in range(len(losses)):
            loss += losses[i]
        return loss/len(losses)

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x for x in outputs]).mean()
        acc = []
        for i in range(len(self.task_index)):
            acc.append(self.test_acc[i].compute())
        return {'avg_test_loss': avg_loss, 'avg_test_acc': acc}

    def on_validation_epoch_start(self):
        self.confmat_metrix = []
        for i, element in enumerate(self.num_categories):
            self.confmat_metrix.append(ConfusionMatrix(num_classes=element))

    def validation_epoch_end(self,val_step_outputs):
        average_acc = 0.0
        for i in range(len(self.task_index)):
            task_name = 'Validation_Acc' + str(self.task_index[i])
            self.log(task_name, self.validation_acc[i].compute())
            self.ComfusionMatrix_alltasks[i].append(self.confmat_metrix[i].compute())
            average_acc += self.validation_acc[i].compute()
        average_acc /= len(self.task_index)
        self.log('Validation_Acc', average_acc)

    def training_epoch_end(self, training_step_outputs):
        average_acc = 0.0
        for i in range(len(self.task_index)):
            task_name = 'Train_Acc' + str(self.task_index[i])
            self.log(task_name, self.train_acc[i].compute())
            average_acc += self.validation_acc[i].compute()
        average_acc /= len(self.task_index)
        self.log('Train_Acc', average_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[100,200,300,350],
                                                         gamma=0.5)
        return [optimizer,], [scheduler,]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    # SE = ShallowExtractor(inchannel=52,groups=1).to(device)
    # summary(SE, (52, 192))
    # DE = DeepExtractor(inchannel=128,layers=[1,2,2],strides=[1,2,2],block=BasicBlock).to(device)
    # summary(DE, (128, 48))
    # Cl = Classifier(inchannel=128,num_categories=6).to(device)
    # summary(Cl, (128, 12))

    # aril = torch.rand([2,52,192]).to(device)
    # widar = torch.rand([2, 90, 1800]).to(device)
    # csida = torch.rand([2, 342, 1800]).to(device)

    basic = KDMTS_RA(inchannel_SE=52, groups_SE=1, layers_DE=[1,2], strides_DE=[1,2],block_DE=BasicBlock,
                         inchannel_DE=128, inchannel_Cl=128,num_categories=[6,16],).to(device)
    start_time = time.time()
    summary(basic, (16,52, 192))
    end_time = time.time()
    print('time:', end_time - start_time)
    # result = MTL_basic(aril)
    # print(len(result))
    # print(len(result[0]))

