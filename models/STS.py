import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch

import pytorch_lightning as pl
import time
from .Basic_Blocks import *
from torchinfo import summary


class STS(nn.Module):
    def __init__(self,inchannel_SE, groups_SE, layers_DE, strides_DE,block_DE, inchannel_DE, inchannel_Cl,
                 num_categories,device):
        super(STS, self).__init__()
        self.SE = ShallowExtractor(inchannel=inchannel_SE,groups=groups_SE).to(device)
        self.DE = DeepExtractor(layers=layers_DE,strides=strides_DE,block=block_DE,inchannel=inchannel_DE).to(device)
        self.CL = Classifier(inchannel=inchannel_Cl,num_categories=num_categories).to(device)

    def forward(self, x):
        shallow_feature = self.SE(x)
        deep_feature = self.DE(shallow_feature)
        result = self.CL(deep_feature)
        return result,shallow_feature,deep_feature


class STS_PL(pl.LightningModule):
    def __init__(self,inchannel,group,num_categories,task_index):
        # task_inex: 1-act/gesture ; 2-location; 3-user; 4-room; 5-orientation;
        super().__init__()
        self.task_index = task_index
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
        self.model = STS(inchannel_SE=inchannel, groups_SE=group, layers_DE=[1,2], strides_DE=[1,2],block_DE=BasicBlock,
                         inchannel_DE=128*group, inchannel_Cl=128*group,num_categories=num_categories,device=device)

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = pl.metrics.Accuracy()
        self.validation_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

        self.acc_record = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data = batch[0]
        label = batch[self.task_index]
        prediction,shallow_feature,deep_feature = self(data)
        loss = self.criterion(prediction, label.long().squeeze())
        self.log("Train_loss", loss)
        self.train_acc(prediction, label.long().squeeze())
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch[0]
        label = batch[self.task_index]
        prediction,shallow_feature,deep_feature = self(data)
        loss = self.criterion(prediction, label.long().squeeze())
        self.log("Validation_loss", loss)
        self.validation_acc(prediction, label.long().squeeze())

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        data = batch[0]
        label = batch[self.task_index]
        prediction,shallow_feature,deep_feature = self(data)
        loss = self.criterion(prediction, label.long().squeeze())
        self.test_acc(prediction, label.long().squeeze())
        return loss

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x for x in outputs]).mean()
        return {'avg_test_loss': avg_loss, 'avg_test_acc': self.test_acc.compute()}

    def validation_epoch_end(self,val_step_outputs):
        self.log('Validation_Acc', self.validation_acc.compute())
        self.acc_record.append(self.validation_acc.compute())

    def training_epoch_end(self, training_step_outputs):
        self.log('Train_Acc', self.train_acc.compute())

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

    # basic = STS(inchannel_SE=52, groups_SE=1, layers_DE=[1,2,2], strides_DE=[1,2,2],block_DE=BasicBlock,
    #                      inchannel_DE=128, inchannel_Cl=128,num_categories=6,device=device).to(device)
    # start_time = time.time()
    # summary(basic, (16, 52, 192))
    # end_time = time.time()
    # print('time:', end_time - start_time)

    basic = STS(inchannel_SE=52, groups_SE=1, layers_DE=[1, 2, 2], strides_DE=[1, 2, 2], block_DE=BasicBlock,
                inchannel_DE=128, inchannel_Cl=128, num_categories=16, device=device).to(device)
    start_time = time.time()
    summary(basic, (16, 52, 192))
    end_time = time.time()
    print('time:', end_time - start_time)

