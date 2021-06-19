import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence
import random
import pytorch_lightning as pl
from torchsummary import summary

class RAW_Feature_Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.l = nn.Sequential(
            nn.Conv2d(kwargs['in_c'],  kwargs['out_feature_dim'] //4, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(kwargs['out_feature_dim'] //4),
            nn.ReLU(),
            nn.Conv2d(kwargs['out_feature_dim'] //4, kwargs['out_feature_dim'] //3, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(kwargs['out_feature_dim'] //3),
            nn.ReLU(),
            nn.Conv2d(kwargs['out_feature_dim'] //3, kwargs['out_feature_dim'] //2, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(kwargs['out_feature_dim'] //2),
            nn.ReLU(),
            nn.Conv2d(kwargs['out_feature_dim'] //2, kwargs['out_feature_dim'], kernel_size=1, bias=False),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        x = self.l(x)
        return x.squeeze()

class Seq_Classifier_RNN(nn.Module):
    # kargs = {'in_feature_dim':None,'lstm_layer':None, 'gesture_class_nums':None,'location_class_num':None,'room_class_num':None,
    #           'orientation_class_num':None,'user_class_num':None}
    def __init__(self, kargs):
        super().__init__()
        self.room_class_true = False
        self.user_class_true = False
        self.location_class_true = False
        self.orientation_class_true = False
        self.gesture_class_true = False

        self.rnn = nn.LSTM(
            # input_size=in_feature_dim, hidden_size=128, num_layers=lstm_layer
            input_size= kargs['feature_dim'], hidden_size=kargs['hidden_size'], num_layers=kargs['lstm_layer_num']

        )
        if kargs["gesture_class_nums"] != 0:
            self.gesture_class_true = True
            self.gse_decoder = nn.Sequential(
                nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, kargs['gesture_class_nums'])
            )
        if kargs["location_class_num"] != 0 :
            self.location_class_true = True
            self.loc_decoder = nn.Sequential(
                nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, kargs['location_class_num'])
            )
        if  kargs["room_class_num"] != 0:
            self.room_class_true = True
            self.room_decoder = nn.Sequential(
                nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, kargs['room_class_num'])
            )
        if kargs["orientation_class_num"] != 0:
            self.orientation_class_true = True
            self.ori_decoder = nn.Sequential(
                nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, kargs['orientation_class_num'])
            )
        if  kargs["user_class_num"] != 0:
            self.user_class_true = True
            self.user_decoder = nn.Sequential(
                nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, kargs['user_class_num'])
            )

    def forward(self, input):############！！！返回值是字典
        pre = {}
        x = pack_sequence(input, enforce_sorted=False)
        output, _ = self.rnn(x)   # (num_layers * num_directions, batch, hidden_size)
        seq_unpacked, lens_unpacked = pad_packed_sequence(output)
        lens_unpacked -= 1
        seq_out = torch.stack(
            [seq_unpacked[lens_unpacked[i], i, :] for i in range(len(lens_unpacked))]
        )
        if self.gesture_class_true:
            pre_ges = self.gse_decoder(seq_out)
            pre['pre_ges']=pre_ges
        if self.location_class_true:
            pre_loc = self.loc_decoder(seq_out)
            pre['pre_loc'] = pre_loc
        if  self.room_class_true:
            pre_room = self.room_decoder(seq_out)
            pre['pre_room'] = pre_room
        if self.orientation_class_true :
            pre_ori = self.ori_decoder(seq_out)
            pre['pre_ori'] = pre_ori
        if  self.user_class_true:
            pre_user = self.user_decoder(seq_out)
            pre['pre_user'] = pre_user

        return pre


class Multi_CNN_LSTM_CSI(pl.LightningModule):
    # kargs={'in_channel':3, 'feature_dim':64,'lstm_layer_num':3,'gesture_class_nums':6,'location_class_num':5,'room_class_num':3,'orientation_class_num':5,
    #        'user_class_num':17}
    def __init__(self,**kargs):
        super().__init__()

        # 标识是否对该类数据进行分类操作
        self.gesture_class_true = False
        self.location_class_true = False
        self.room_class_true = False
        self.orientation_class_true = False
        self.user_class_true = False

        if kargs["gesture_class_nums"] != 0 :#如果要 将标识进行修改为true
            self.gesture_class_true = True
        if kargs["location_class_num"] != 0:
            self.location_class_true = True
        if kargs["room_class_num"] != 0:
            self.room_class_true = True
        if kargs["orientation_class_num"] != 0:
            self.orientation_class_true = True

        if kargs["user_class_num"] != 0:
            self.user_class_true = True

        self.CNN_encoder = RAW_Feature_Encoder(in_c=kargs['in_channel'], out_feature_dim=kargs['feature_dim'])
        self.LSTM_classifier = Seq_Classifier_RNN(kargs)

        self.criterion = nn.CrossEntropyLoss(size_average=False)

        if self.gesture_class_true:
            self.train_acc_ges= pl.metrics.Accuracy()
            self.val_acc_ges = pl.metrics.Accuracy()

        if self.location_class_true:
            self.train_acc_loc= pl.metrics.Accuracy()
            self.val_acc_loc = pl.metrics.Accuracy()

        if self.room_class_true:
            self.train_acc_room= pl.metrics.Accuracy()
            self.val_acc_room = pl.metrics.Accuracy()

        if self.orientation_class_true:
            self.train_acc_ori= pl.metrics.Accuracy()
            self.val_acc_ori = pl.metrics.Accuracy()

        if self.user_class_true:
            self.train_acc_user= pl.metrics.Accuracy()
            self.val_acc_user = pl.metrics.Accuracy()

    def training_step(self, batch, batch_idx):
        pre = {}
        loss = 0
        loss_dic = {}
        real_label = {}
        label = {'ges_label': [], 'loc_label': [], 'room_label': [], 'ori_label': [], 'user_label': []}
        # sample, ges_label, loc_label, room_label, ori_label, user_label = batch  # the data is a 5 dimensional tensor: [batch,time,in_channel/receiver,length,width]
        sample, real_label = batch
        for i in range(len(real_label)):
            label_dict = real_label[i]
            for key, value in label_dict.items():
                label[key].append(value)

        input_data = torch.cat(sample)  # [batch*time,in_channel,length,width]
        feature_out = self.CNN_encoder(input_data)
        seq_in = torch.split(feature_out, [len(x) for x in sample])  # [batch,time,feature_dim]
        # pre_ges, pre_loc, pre_room, pre_ori, pre_user = self.LSTM_classifier(seq_in)
        pre = self.LSTM_classifier(seq_in)

        if self.gesture_class_true:

            ges_label = tuple(label['ges_label'])
            activity_label = torch.stack(ges_label)
            pre_ges = pre["pre_ges"]
            loss_ges = self.criterion(pre_ges, activity_label.long().squeeze())
            loss_dic["loss_ges"] = loss_ges
            loss += loss_ges
            self.train_acc_ges(pre_ges, activity_label.long().squeeze())
        if self.location_class_true:
            print("location class task!")
            loc_label = tuple(label['loc_label'])
            loc_label = torch.stack(loc_label)
            pre_loc = pre["pre_loc"]
            loss_loc = self.criterion(pre_loc, loc_label.long().squeeze())
            loss_dic["loss_loc"] = loss_loc
            loss += loss_loc
            self.train_acc_loc(pre_loc, loc_label.long().squeeze())

        if self.room_class_true:
            print("room class task!")
            room_label = tuple(label['room_label'])
            room_label = torch.stack(room_label)
            pre_room = pre["pre_room"]
            loss_room = self.criterion(pre_room, room_label.long().squeeze())
            loss_dic["loss_room"] = loss_room
            loss += loss_room
            self.train_acc_room(pre_room, room_label.long().squeeze())

        if self.orientation_class_true:
            print("orientation class task!")
            ori_label = tuple(label['ori_label'])
            ori_label = torch.stack(ori_label)
            pre_ori = pre["pre_ori"]
            loss_ori = self.criterion(pre_ori, ori_label.long().squeeze())
            loss_dic["loss_ori"] = loss_ori
            loss += loss_ori
            self.train_acc_ori(pre_ori, ori_label.long().squeeze())

        if self.user_class_true:
            print("user class task!")
            user_label = tuple(label['user_label'])
            user_label = torch.stack(user_label)
            pre_user = pre["pre_user"]
            loss_user = self.criterion(pre_user, user_label.long().squeeze())
            loss_dic["'loss_user'"] = loss_user
            loss += loss_user
            self.train_acc_user(pre_user, user_label.long().squeeze())

        self.log_scalers(name="Tr_loss", valuedict=loss_dic)

        return loss

    def validation_step(self,  batch, batch_idx):
        pre = {}
        loss = 0
        loss_dic = {}
        real_label={}
        label = {'ges_label':[],'loc_label':[],'room_label':[],'ori_label':[],'user_label':[]}
        # sample, ges_label, loc_label, room_label, ori_label, user_label = batch  # the data is a 5 dimensional tensor: [batch,time,in_channel/receiver,length,width]
        sample,real_label = batch
        for i in range(len(real_label)):
            label_dict = real_label[i]
            for key, value in label_dict.items():
                label[key].append(value)

        input_data = torch.cat(sample)  # [batch*time,in_channel,length,width]
        feature_out = self.CNN_encoder(input_data)
        seq_in = torch.split(feature_out, [len(x) for x in sample])  # [batch,time,feature_dim]
        # pre_ges, pre_loc, pre_room, pre_ori, pre_user = self.LSTM_classifier(seq_in)
        pre = self.LSTM_classifier(seq_in)

        if self.gesture_class_true:
            ges_label =tuple(label['ges_label'])
            activity_label = torch.stack(ges_label)
            pre_ges = pre["pre_ges"]
            loss_ges = self.criterion(pre_ges, activity_label.long().squeeze())
            loss_dic["loss_ges"] = loss_ges
            loss += loss_ges
            self.val_acc_ges(pre_ges, activity_label.long().squeeze())

        if self.location_class_true:
            loc_label = tuple(label['loc_label'])
            loc_label = torch.stack(loc_label)
            pre_loc = pre["pre_loc"]
            loss_loc = self.criterion(pre_loc, loc_label.long().squeeze())
            loss_dic["loss_loc"] = loss_loc
            loss += loss_loc
            self.val_acc_loc(pre_loc, loc_label.long().squeeze())

        if self.room_class_true:
            room_label = tuple(label['room_label'])
            room_label = torch.stack(room_label)
            pre_room = pre["pre_room"]
            loss_room = self.criterion(pre_room, room_label.long().squeeze())
            loss_dic["loss_room"] = loss_room
            loss += loss_room
            self.val_acc_room(pre_room, room_label.long().squeeze())

        if self.orientation_class_true:
            ori_label = tuple(label['ori_label'])
            ori_label = torch.stack(ori_label)
            pre_ori = pre["pre_ori"]
            loss_ori = self.criterion(pre_ori, ori_label.long().squeeze())
            loss_dic["loss_ori"] = loss_ori
            loss += loss_ori
            self.val_acc_ori(pre_ori, ori_label.long().squeeze())

        if self.user_class_true:
            user_label = tuple(label['user_label'])
            user_label = torch.stack(user_label)
            pre_user = pre["pre_user"]
            loss_user = self.criterion(pre_user, user_label.long().squeeze())
            loss_dic["'loss_user'"] = loss_user
            loss += loss_user
            self.val_acc_user(pre_user, user_label.long().squeeze())

        self.log_scalers(name="Val_loss", valuedict=loss_dic)

        return loss

    def validation_epoch_end(self, val_step_outputs):
        acc_dic = { }
        if self.gesture_class_true:
            acc_dic['val_acc_ges']= self.val_acc_ges.compute()
        if self.location_class_true:
            acc_dic['val_acc_loc'] = self.val_acc_loc.compute()
        if self.room_class_true:
            acc_dic['val_acc_room'] = self.val_acc_room.compute()
        if self.orientation_class_true:
            acc_dic['val_acc_ori'] = self.val_acc_ori.compute()
        if self.user_class_true :
            acc_dic['val_acc_user'] = self.val_acc_user.compute()

        self.log_scalers(name="Val_acc", valuedict=acc_dic)
        # self.log('GesVa_Acc', self.val_acc.compute())

    def training_epoch_end(self, training_step_outputs):
        acc_dic = {}
        if self.gesture_class_true:
            acc_dic['train_acc_ges'] = self.train_acc_ges.compute()
        if self.location_class_true:
            acc_dic['train_acc_loc'] = self.train_acc_loc.compute()
        if self.room_class_true:
            acc_dic['train_acc_room'] = self.train_acc_room.compute()
        if self.orientation_class_true:
            acc_dic['train_acc_ori'] = self.train_acc_ori.compute()
        if self.user_class_true :
            acc_dic['train_acc_user'] = self.train_acc_user.compute()

        self.log_scalers(name="Tr_acc", valuedict=acc_dic)
        # self.log('GesTr_Acc', self.train_acc.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3,weight_decay = 0.1)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[40,  80, 120, 160, 200, 240, 280,320,360,400],
                                                         gamma=0.5)
        return [optimizer, ], [scheduler, ]

    def log_scalers(self, name, valuedict)  :
        self.logger.experiment.add_scalars(
            name, valuedict, global_step=self.global_step
        )
        # self.logger.experiment.close()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    model = RAW_Feature_Encoder(in_c=3, out_feature_dim=64).to(device)
    summary(model, (3,50,30))

