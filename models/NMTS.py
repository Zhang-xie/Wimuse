import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch

import pytorch_lightning as pl
import time

from .Basic_Blocks import *
from torchinfo import summary


class NMTS(nn.Module):
    def __init__(self,inchannel_SE, groups_SE, layers_DE, strides_DE, block_DE, inchannel_DE, inchannel_Cl,
                 num_categories):
        super(NMTS, self).__init__()
        self.SE = ShallowExtractor(inchannel=inchannel_SE,groups=groups_SE)
        self.DE = DeepExtractor(layers=layers_DE,strides=strides_DE,block=block_DE,inchannel=inchannel_DE)
        self.classifiers = torch.nn.ModuleList()
        for i,element in enumerate(num_categories):
            self.classifiers.append(Classifier(inchannel=inchannel_Cl,num_categories=element))

    def forward(self, x):
        shallow_feature = self.SE(x)
        deep_feature = self.DE(shallow_feature)
        predictions = []
        for i,element in enumerate(self.classifiers):
            predictions.append(element(deep_feature))
        return predictions


class NMTS_PL(pl.LightningModule):
    def __init__(self,inchannel,group,num_categories,task_index):
        # task_inex: 1-act/gesture ; 2-location; 3-user; 4-room; 5-orientation;
        # num_categories and task_index are two list in this model.
        super().__init__()
        self.task_index = task_index
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
        self.model = NMTS(inchannel_SE=inchannel, groups_SE=group, layers_DE=[1,2], strides_DE=[1,2],block_DE=BasicBlock,
                         inchannel_DE=128*group, inchannel_Cl=128*group,num_categories=num_categories)

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = nn.ModuleList()
        self.validation_acc = nn.ModuleList()
        self.test_acc = nn.ModuleList()
        for i in range(len(task_index)):
            self.train_acc.append(pl.metrics.Accuracy())
            self.validation_acc.append(pl.metrics.Accuracy())
            self.test_acc.append(pl.metrics.Accuracy())

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data = batch[0]
        labels = [batch[i] for i in self.task_index]
        predictions = self(data)
        losses = []
        for i in range(len(predictions)):
            temp = self.criterion(predictions[i],labels[i].long().squeeze())
            losses.append(temp)
            task_name = 'Train_loss' + str(self.task_index[i])
            self.log(task_name, temp)
            self.train_acc[i](predictions[i],labels[i].long().squeeze())
        loss = 0
        for i in range(len(losses)):
            loss += losses[i]
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch[0]
        labels = [batch[i] for i in self.task_index]
        predictions = self(data)
        losses = []
        for i in range(len(predictions)):
            temp = self.criterion(predictions[i], labels[i].long().squeeze())
            losses.append(temp)
            task_name = 'Validation_loss' + str(self.task_index[i])
            self.log(task_name, temp)
            self.validation_acc[i](predictions[i], labels[i].long().squeeze())
        loss = 0
        for i in range(len(losses)):
            loss += losses[i]
        return loss

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        data = batch[0]
        labels = [batch[i] for i in self.task_index]
        predictions = self(data)
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

    def validation_epoch_end(self,val_step_outputs):
        average_acc = 0.0
        for i in range(len(self.task_index)):
            task_name = 'Validation_Acc' + str(self.task_index[i])
            self.log(task_name, self.validation_acc[i].compute())
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
    basic = NMTS(inchannel_SE=52, groups_SE=1, layers_DE=[1,2], strides_DE=[1,2],block_DE=BasicBlock,
                         inchannel_DE =128, inchannel_Cl=128,num_categories=[6,16]).to(device)
    start_time = time.time()
    summary(basic, (16, 52, 192))
    end_time = time.time()
    print('time:', end_time - start_time)
    # result = MTL_basic(aril)
    # print(len(result))
    # print(len(result[0]))

