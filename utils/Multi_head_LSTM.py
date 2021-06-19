import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence
import random
import pytorch_lightning as pl
from torchsummary import summary


class RAW_Feature_Encoder(nn.Module):
    def __init__(self, in_c, out_feature_dim):
        super().__init__()
        self.l = nn.Sequential(
            nn.Conv2d(in_c, 32, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, out_feature_dim, kernel_size=1, bias=False),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        x = self.l(x)
        return x.squeeze()


class Seq_Classifier_RNN(nn.Module):
    def __init__(self, in_feature_dim, lstm_layer, gesture_class_nums,location_class_num,room_class_num,orientation_class_num,user_class_num):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=in_feature_dim, hidden_size=128, num_layers=lstm_layer
        )

        self.gse_decoder = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, gesture_class_nums)
        )
        self.loc_decoder = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, location_class_num)
        )
        self.room_decoder = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, room_class_num)
        )
        self.ori_decoder = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, orientation_class_num)
        )
        self.user_decoder = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, user_class_num)
        )

    def forward(self, input):
        x = pack_sequence(input, enforce_sorted=False)
        output, _ = self.rnn(x)   # (num_layers * num_directions, batch, hidden_size)
        seq_unpacked, lens_unpacked = pad_packed_sequence(output)
        lens_unpacked -= 1
        seq_out = torch.stack(
            [seq_unpacked[lens_unpacked[i], i, :] for i in range(len(lens_unpacked))]
        )
        pre_ges = self.gse_decoder(seq_out)
        pre_loc = self.loc_decoder(seq_out)
        pre_room = self.room_decoder(seq_out)
        pre_ori = self.ori_decoder(seq_out)
        pre_user = self.user_decoder(seq_out)

        return pre_ges,pre_loc,pre_room,pre_ori,pre_user


class Multi_CNN_LSTM_CSI(pl.LightningModule):
    def __init__(self,in_channel=3, feature_dim=64,lstm_layer_num=3,gesture_class_nums=6,location_class_num=5,room_class_num=3,orientation_class_num=5,user_class_num=17):
        super().__init__()
        self.in_channel = in_channel
        self.feature_dim = feature_dim
        self.lstm_layer_num = lstm_layer_num

        self.gesture_class_nums = gesture_class_nums
        self.location_class_num = location_class_num
        self.room_class_num = room_class_num
        self.orientation_class_num = orientation_class_num
        self.user_class_num = user_class_num

        self.CNN_encoder = RAW_Feature_Encoder(in_c=self.in_channel, out_feature_dim=self.feature_dim)
        self.LSTM_classifier = Seq_Classifier_RNN(in_feature_dim=self.feature_dim , lstm_layer=self.lstm_layer_num,
                                                  gesture_class_nums=self.gesture_class_nums,
                                                  location_class_num=self.location_class_num,
                                                  room_class_num=self.room_class_num,
                                                  orientation_class_num=self.orientation_class_num,
                                                  user_class_num=self.user_class_num)

        self.criterion = nn.CrossEntropyLoss(size_average=False)

        self.train_acc_ges= pl.metrics.Accuracy()
        self.train_acc_loc= pl.metrics.Accuracy()
        self.train_acc_room= pl.metrics.Accuracy()
        self.train_acc_ori= pl.metrics.Accuracy()
        self.train_acc_user= pl.metrics.Accuracy()

        self.val_acc_ges = pl.metrics.Accuracy()
        self.val_acc_loc = pl.metrics.Accuracy()
        self.val_acc_room = pl.metrics.Accuracy()
        self.val_acc_ori = pl.metrics.Accuracy()
        self.val_acc_user = pl.metrics.Accuracy()

    def training_step(self, batch, batch_idx):
        sample,ges_label,loc_label,room_label,ori_label,user_label = batch   # the data is a 5 dimensional tensor: [batch,time,in_channel/receiver,length,width]
        activity_label = torch.stack(ges_label)
        loc_label = torch.stack(loc_label)
        room_label = torch.stack(room_label)
        ori_label = torch.stack(ori_label)
        user_label = torch.stack(user_label)

        input_data = torch.cat(sample)  # [batch*time,in_channel,length,width]
        feature_out = self.CNN_encoder(input_data)
        seq_in = torch.split(feature_out, [len(x) for x in sample])  # [batch,time,feature_dim]
        pre_ges,pre_loc,pre_room,pre_ori,pre_user = self.LSTM_classifier(seq_in)

        loss_ges = self.criterion(pre_ges, activity_label.long().squeeze())
        loss_loc = self.criterion(pre_loc, loc_label.long().squeeze())
        loss_room = self.criterion(pre_room, room_label.long().squeeze())
        loss_ori = self.criterion(pre_ori, ori_label.long().squeeze())
        loss_user = self.criterion(pre_user, user_label.long().squeeze())
        loss = loss_ges + loss_loc + loss_room + loss_ori + loss_user

        loss_dic = {'loss_ges':loss_ges,'loss_loc':loss_loc,'loss_room':loss_room,'loss_ori':loss_ori,'loss_user':loss_user,}
        self.log_scalers(name="Tr_loss", valuedict= loss_dic)

        self.train_acc_ges(pre_ges, activity_label.long().squeeze())
        self.train_acc_loc(pre_loc, loc_label.long().squeeze())
        self.train_acc_room(pre_room, room_label.long().squeeze())
        self.train_acc_ori(pre_ori, ori_label.long().squeeze())
        self.train_acc_user(pre_user, user_label.long().squeeze())

        return loss

    def validation_step(self,  batch, batch_idx):
        sample, ges_label, loc_label, room_label, ori_label, user_label = batch  # the data is a 5 dimensional tensor: [batch,time,in_channel/receiver,length,width]
        activity_label = torch.stack(ges_label)
        loc_label = torch.stack(loc_label)
        room_label = torch.stack(room_label)
        ori_label = torch.stack(ori_label)
        user_label = torch.stack(user_label)

        input_data = torch.cat(sample)  # [batch*time,in_channel,length,width]
        feature_out = self.CNN_encoder(input_data)
        seq_in = torch.split(feature_out, [len(x) for x in sample])  # [batch,time,feature_dim]
        pre_ges, pre_loc, pre_room, pre_ori, pre_user = self.LSTM_classifier(seq_in)

        loss_ges = self.criterion(pre_ges, activity_label.long().squeeze())
        loss_loc = self.criterion(pre_loc, loc_label.long().squeeze())
        loss_room = self.criterion(pre_room, room_label.long().squeeze())
        loss_ori = self.criterion(pre_ori, ori_label.long().squeeze())
        loss_user = self.criterion(pre_user, user_label.long().squeeze())
        loss = loss_ges + loss_loc + loss_room + loss_ori + loss_user

        loss_dic = {'loss_ges': loss_ges, 'loss_loc': loss_loc, 'loss_room': loss_room, 'loss_ori': loss_ori,
                    'loss_user': loss_user, }
        self.log_scalers(name="Val_loss", valuedict=loss_dic)

        self.val_acc_ges(pre_ges, activity_label.long().squeeze())
        self.val_acc_loc(pre_loc, loc_label.long().squeeze())
        self.val_acc_room(pre_room, room_label.long().squeeze())
        self.val_acc_ori(pre_ori, ori_label.long().squeeze())
        self.val_acc_user(pre_user, user_label.long().squeeze())

        return loss

    def validation_epoch_end(self, val_step_outputs):
        acc_dic = {'val_acc_ges': self.val_acc_ges.compute(),'val_acc_loc': self.val_acc_loc.compute(),'val_acc_room':
            self.val_acc_room.compute(),'val_acc_ori': self.val_acc_ori.compute(),'val_acc_user': self.val_acc_user.compute(),}
        self.log_scalers(name="Val_acc", valuedict=acc_dic)
        # self.log('GesVa_Acc', self.val_acc.compute())

    def training_epoch_end(self, training_step_outputs):
        acc_dic = {'train_acc_ges': self.train_acc_ges.compute(), 'train_acc_loc': self.train_acc_loc.compute(), 'train_acc_room':
            self.train_acc_room.compute(), 'train_acc_ori': self.train_acc_ori.compute(),
                   'train_acc_user': self.train_acc_user.compute(), }
        self.log_scalers(name="Tr_acc", valuedict=acc_dic)
        # self.log('GesTr_Acc', self.train_acc.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[40,  80, 120, 160, 200, 240, 280,320,360,400],
                                                         gamma=0.5)
        return [optimizer, ], [scheduler, ]

    def log_scalers(self, name, valuedict):
        self.logger.experiment.add_scalars(
            name, valuedict, global_step=self.global_step
        )


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    model = RAW_Feature_Encoder(in_c=3, out_feature_dim=64).to(device)
    summary(model, (3,50,30))