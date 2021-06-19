import zarr
import torch.utils.data
import zipfile
import tqdm
from pathlib import Path
import numpy as np
import io
import torch.utils.data as data
import torch
import os
import itertools, functools
from torch.utils.data import TensorDataset, DataLoader


class Mydata(data.Dataset):
    def __init__(self, root, roomid=None, userid=None, location=None, mode=None,shape=None,flag_preload=False,*args, **kwargs):
        super(Mydata, self).__init__()
        self.root = root
        self.batch_idx = 0
        self.num_class = 6
        self.mode = mode
        self.shape = shape

        self.group = zarr.open_group(root.as_posix(), mode="r")
        self.gesture = self.group.csi_label_act[:]  # gesture: 0~5
        self.room_label = self.group.csi_label_env[:]  # room: 0,1
        self.location = self.group.csi_label_loc[:]  # location: 0,1  0,1,2
        self.location = self.location + self.room_label*3  # combine location  and room into five types:0,1,2,3,4
        self.userid = self.group.csi_label_user[:]  # user: 0,1,2,3,4

        self.total_samples = len(self.gesture)
        self.select = np.ones(self.total_samples, dtype=np.bool)
        self.room_select = np.ones(self.total_samples, dtype=np.bool)
        self.user_select = np.ones(self.total_samples, dtype=np.bool)
        self.loc_select = np.ones(self.total_samples, dtype=np.bool)

        index_temp = np.arange(self.total_samples)

        if roomid is not None:
            self.room_select = functools.reduce(np.logical_or, [*[self.room_label == j for j in roomid]])
            self.select = np.logical_and(self.select, self.room_select)
        if userid is not None:
            self.user_select = functools.reduce(np.logical_or, [*[self.userid == j for j in userid]])
            self.select = np.logical_and(self.select, self.user_select)
        if location is not None:
            self.loc_select = functools.reduce(np.logical_or, [*[self.location == j for j in location]])
            self.select = np.logical_and(self.select, self.loc_select)

        self.index = index_temp[self.select]  # the data for a specified task
        self.flag_preload=False
        if flag_preload:
            self.preload_data=[self[i] for i in tqdm.trange(len(self))]
            self.flag_preload=True

    def __getitem__(self, index):
        if self.flag_preload:
            return self.preload_data[index]
        sample_index = self.index[index]
        if self.mode is not None:
            if self.mode == 'phase':
                pha_sample = self.group.csi_data_pha[sample_index]
                sample = pha_sample.astype(np.float32)  # shape [1800,3,114]
            elif self.mode == 'amplitude':
                amp_sample = self.group.csi_data_amp[sample_index]
                # amp_sample = self.amp[sample_index]
                sample = amp_sample.astype(np.float32)  # shape [1800,3,114]
            else:
                amp_sample = self.group.csi_data_amp[sample_index]
                pha_sample = self.group.csi_data_pha[sample_index]
                amp_sample = amp_sample.astype(np.float32)  # shape [1800,3,114]
                pha_sample = pha_sample.astype(np.float32)  # shape [1800,3,114]
                sample = np.concatenate((amp_sample, pha_sample), axis=2)  # shape [1800,3,228]
        else:
            amp_sample = self.group.csi_data_amp[sample_index]
            pha_sample = self.group.csi_data_pha[sample_index]
            amp_sample = amp_sample.astype(np.float32)  # shape [1800,3,114]
            pha_sample = pha_sample.astype(np.float32)  # shape [1800,3,114]
            sample = np.concatenate((amp_sample, pha_sample), axis=2)  # shape [1800,3,228]

        ges_label = self.gesture[sample_index]  # {0,1, 2, 3, 4, 5}
        ges_label = torch.tensor(ges_label).type(torch.LongTensor)

        loc_label = self.location[sample_index]  # {0,1, 2,}
        loc_label = torch.tensor(loc_label).type(torch.LongTensor)

        user_label = self.userid[sample_index]  # {0,1, 2, 3, 4, 5}
        user_label = torch.tensor(user_label).type(torch.LongTensor)

        sample = torch.Tensor(np.array(sample))
        sample = sample.permute(1, 2, 0).type(torch.FloatTensor)  # shape [3,114 or 228,time]
        if self.shape is not None:
            pass
        else:
            sample = sample.reshape(sample.shape[0] * sample.shape[1], -1)

        return sample, ges_label,loc_label,user_label

    def get_choose_label(self, id):
        if id == "user":
            return self.userid[self.index]
        if id == "room":
            return self.room_label[self.index]
        if id == "location":
            return self.location[self.index]
        if id == "gesture":
            return self.gesture[self.index]

    def __len__(self):
        return len(self.index)


class CSIDA():
    def __init__(self, root, roomid=None,userid=None,location=None,mode=None,shape=None,*args, **kwargs):
        self.full_data = Mydata(root, roomid=roomid,userid=userid,location=location,mode=mode,shape=shape,*args, **kwargs)
        train_size = int(0.8 * len(self.full_data))
        test_size = len(self.full_data) - train_size
        self.train, self.test = torch.utils.data.random_split(self.full_data, [train_size, test_size])


if __name__ == "__main__":
    root = Path("E:/csi_301")
    data = CSIDA(root=root, roomid=None, userid=None, location=None, mode="amplitude")
    a = data.train
    b = data.test

    print("room:", np.unique(data.full_data.get_choose_label("room")))
    print("gesture:", np.unique(data.full_data.get_choose_label("gesture")))
    print("user:", np.unique(data.full_data.get_choose_label("user")))
    print("location:", np.unique(data.full_data.get_choose_label("location")))

    print(len(a))
    print(len(b))
    print(a[0][0].shape)

    # room: [0 1]
    # gesture: [0 1 2 3 4 5]
    # user: [0 1 2 3 4]
    # location: [0 1 2 3 4]
    # 2275
    # 569
    # torch.Size([342, 1800])