from __future__ import absolute_import, print_function
"""
ARILdata
"""
import torch
import torch.utils.data as data
from pathlib import Path
import os
import sys
import numpy as np


def default_loader(root):   #root is the data storage path  eg. path = D:/CSI_Data/signfi_matlab2numpy/
    train_label_activity_path = root / "datatrain_activity_label.npy"
    train_amp_path = root / "datatrain_data.npy"
    train_label_loc_path = root /"datatrain_location_label.npy"
    test_label_activity_path = root / "datatest_activity_label.npy"
    test_amp_path = root / "datatest_data.npy"
    test_label_loc_path = root / "datatest_location_label.npy"

    train_label_activity = np.load(train_label_activity_path)
    train_amp = np.load(train_amp_path)
    train_label_loc = np.load(train_label_loc_path)
    test_label_activity = np.load(test_label_activity_path)
    test_amp = np.load(test_amp_path)
    test_label_loc = np.load(test_label_loc_path)

    all_label_activity = np.concatenate((train_label_activity,test_label_activity))
    all_amp = np.concatenate((train_amp,test_amp))
    all_label_location = np.concatenate((train_label_loc, test_label_loc))
    return all_label_activity,all_label_location,all_amp


class MyData(data.Dataset):
    def __init__(self, root, loader=default_loader,*args, **kwargs ):
        self.root = root
        self.load = loader
        self.act_label, self.loc_label, self.amp = self.load(self.root)

    def __getitem__(self, index):
        activity_label_index,loc_label_index, data_index = self.act_label[index],self.loc_label[index], self.amp[index]
        activity_label_index_tensor = torch.from_numpy(activity_label_index).type(torch.FloatTensor)
        loc_label_index_tensor = torch.from_numpy(loc_label_index).type(torch.FloatTensor)
        data_index_tensor = torch.from_numpy(data_index).type(torch.FloatTensor)
        return data_index_tensor,activity_label_index_tensor,loc_label_index_tensor,

    def __len__(self):
        return len(self.loc_label)


class ARIL:
    def __init__(self, root, *args, **kwargs):
        self.full_data = MyData(root,*args, **kwargs)
        train_size = int(0.8 * len(self.full_data))
        test_size = len(self.full_data) - train_size
        self.train, self.test = torch.utils.data.random_split(self.full_data, [train_size, test_size])


def test_ARIL(root):
    print("hahahahhahahha")
    print(ARIL.__name__)
    data = ARIL(root)
    print("the length of training set:{}".format(len(data.train)))
    print("the length of test set:{}".format(len(data.test)))
    print("the data of train[0]:{}".format(data.train[0]))
    print("the shape of test[10][2]:{}".format(data.test[:][2].shape))
    print("the shape of test[10][0]:{}".format(data.test[:][0].shape))


if __name__ == "__main__":
    test_ARIL("../../Datasets/ARIL")
