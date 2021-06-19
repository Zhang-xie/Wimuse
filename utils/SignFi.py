from __future__ import absolute_import, print_function
"""
SignFidata
"""
import torch
import torch.utils.data as data

import os
import sys
import numpy as np
import math
from collections import defaultdict


def default_loader(root):   #root is the data storage path  eg. path = D:/CSI_Data/signfi_matlab2numpy/
    # 150 gesture for 5 users in lab every user has 1500 instances
    label_path = root + "/labels.npy"    #storage path : label of Sign
    amp_path = root + "/amp_datas.npy"   #storage path : the amplitude of CSI
    phase_path = root + "/phase_datas.npy" #storage path :  the phase value of CSI
    label_all = np.load(label_path)     #label of Sign
    amp_all = np.load(amp_path)
    phase_all = np.load(phase_path)

    # the 276 gesture data for one user in lab and home
    data_home_dl_276_amp_path = root + "/data_home_dl_276_amp.npy"
    data_home_dl_276_pha_path = root + "/data_home_dl_276_pha.npy"
    label_home_276_path = root + "/label_home_276.npy"
    data_lab_dl_276_amp_path = root + "/data_lab_dl_276_amp.npy"
    data_lab_dl_276_pha_path = root + "/data_lab_dl_276_pha.npy"
    label_lab_276_path = root + "/label_lab_276.npy"
    data_home_dl_276_amp = np.load(data_home_dl_276_amp_path)
    data_home_dl_276_pha = np.load(data_home_dl_276_pha_path)
    label_home_276 = np.load(label_home_276_path)
    data_lab_dl_276_amp = np.load(data_lab_dl_276_amp_path)
    data_lab_dl_276_pha = np.load(data_lab_dl_276_pha_path)
    label_lab_276 = np.load(label_lab_276_path)
    return label_all,amp_all,phase_all,data_home_dl_276_amp,data_home_dl_276_pha,label_home_276,data_lab_dl_276_amp,data_lab_dl_276_pha,label_lab_276


def split_array_bychunk(array, chunksize, include_residual=True):
    len_ = len(array) // chunksize * chunksize
    array, array_residual = array[:len_], array[len_:]
    # array = np.split(array, len_ // chunksize)
    array = [
        array[i * chunksize: (i + 1) * chunksize]
        for i in range(len(array) // chunksize)
    ]
    if include_residual:
        if len(array_residual) == 0:
            return array
        else:
            return array + [
                array_residual,
            ]
    else:
        if len(array_residual) == 0:
            return array, None
        else:
            return array, array_residual


class MyData(data.Dataset):
    def __init__(self, root, user, subset, loader=default_loader,model=None,chunk_size=None):
        # when subset==150: we return the data of user in SignFi150 dataset, user in {0,1,2,3,4}
        # when subset==276: we reuse the mark (user) as user==0: return SignFi276 Lab data; user==1: return SignFi276 Home data
        self.root = root
        self.load = loader
        self.model = model
        self.chunk_size = chunk_size
        label_all,amp_all,phase_all, data_home_dl_276_amp,data_home_dl_276_pha,label_home_276,data_lab_dl_276_amp,data_lab_dl_276_pha,label_lab_276 = self.load(root)
        if subset == 150:  # return the data of user in SignFi150 Sub-dataset
            self.label = label_all[user * 1500:(user + 1) * 1500]
            self.label_env = np.ones(1500) * user  # the env_label of lab data is 7
            self.amp = amp_all[:, :, :, user * 1500:(user + 1) * 1500]
            self.phase = phase_all[:, :, :, user * 1500:(user + 1) * 1500]
        else:  # return the data in SignFi276 Sub-dataset
            if user == 0:  # return Lab data
                self.label = label_lab_276 - 1
                self.label_env = np.ones(5520) * 5  # the env_label of lab data is 5
                self.amp = data_lab_dl_276_amp
                self.phase = data_lab_dl_276_pha
            else:   # return Home data
                self.label = label_home_276 - 1
                self.label_env = np.ones(2760) * 6  # the env_label of home data is 6
                self.amp = data_home_dl_276_amp
                self.phase = data_home_dl_276_pha

    def __getitem__(self, index):
        label_index,label_env_index, amp_index, phase_index = self.label[index],self.label_env[index], self.amp[:,:,:,index], self.phase[:,:,:,index]

        if self.model is not None:
            amp_sample = amp_index.astype(np.float32)  # shape [200,30,3]
            amp, amp_res = split_array_bychunk(amp_sample, self.chunk_size,
                                               include_residual=False)  # shape list{[chunk,30,3],repeat}
            amp += [
                amp_sample[-self.chunk_size:],
            ]
            amp = torch.Tensor(np.array(amp))

            pha_sample = phase_index.astype(np.float32)  # shape [200,30,3]
            pha, pha_res = split_array_bychunk(pha_sample, self.chunk_size,
                                               include_residual=False)  # shape list{[chunk,30,3],repeat}
            pha += [
                pha_sample[-self.chunk_size:],
            ]
            pha = torch.Tensor(np.array(pha))   # shape [repeat,chunk,30,3]

            data_amp_phase = torch.cat(
                [amp, pha], dim=2).permute(0, 3, 1, 2).type(torch.FloatTensor)  # shape [repeat,3,chunk,60]
        else:
            # the initial amp.shape is [200,30,3], we concat amp and pha into a numpy array with shape: [200,60,3]
            # and then, transpose to [3,60,200], reshape to [180,200]
            amp_index = torch.from_numpy(amp_index).type(torch.FloatTensor)
            phase_index = torch.from_numpy(phase_index).type(torch.FloatTensor)
            data_amp_phase = torch.cat(
                [amp_index, phase_index], dim=1).permute(2, 1, 0).reshape(180, 200)
        label_env_index = label_env_index.astype(np.int)
        label_env_index_tensor = torch.tensor(label_env_index).type(torch.FloatTensor)
        label_index = label_index.astype(np.int)
        label_index_tensor = torch.tensor(label_index).type(torch.FloatTensor)
        return data_amp_phase,label_index_tensor,label_env_index_tensor

    def __len__(self):
        return len(self.label)


class SignFi:
    def __init__(self, root, mode,model=None,chunk_size=None):
        # label/activity lable:{0,...,149} or {0,...,275}
        # label_env:{0,...,6}
        # if mode in {1,2,3,4,5} means that: mode=j(using user-j_SignFi150 for training and testing)
        # if mode in {6,7} means that using Lab_SignFi276 or Home_SignFi276 for training and testing)\
        # if mode>10 means that using mode=int(mode/10) data for training, mode=mode%10 data for testing.(cross-domain)
        if mode < 10:  # return in-domain data
            if mode == 6: # using SignFi276 Lab sub-dataset for training and testing
                self.full_data = MyData(root,user=0, subset=276,model=model,chunk_size=chunk_size)
                train_size = int(0.8 * len(self.full_data))
                test_size = len(self.full_data) - train_size
                self.train, self.test = torch.utils.data.random_split(self.full_data, [train_size, test_size])
            elif mode == 7:
                self.full_data = MyData(root, user=1, subset=276,model=model,chunk_size=chunk_size)
                train_size = int(0.8 * len(self.full_data))
                test_size = len(self.full_data) - train_size
                self.train, self.test = torch.utils.data.random_split(self.full_data, [train_size, test_size])
            else:
                self.full_data = MyData(root, user= mode-1, subset=150,model=model,chunk_size=chunk_size)
                train_size = int(0.8 * len(self.full_data))
                test_size = len(self.full_data) - train_size
                self.train, self.test = torch.utils.data.random_split(self.full_data, [train_size, test_size])
        else:   # return cross domain data
            mode1 = math.floor(mode/10)  # this is for training
            mode2 = mode % 10            # this is for testing
            if mode1 == 6:
                self.train = MyData(root,user=0, subset=276,model=model,chunk_size=chunk_size)
            elif mode1 == 7:
                self.train = MyData(root, user=1, subset=276,model=model,chunk_size=chunk_size)
            else:
                self.train = MyData(root, user=mode1-1, subset=150,model=model,chunk_size=chunk_size)

            if mode2 == 6:
                self.test = MyData(root, user=0, subset=276,model=model,chunk_size=chunk_size)
            elif mode2 == 7:
                self.test = MyData(root, user=1, subset=276,model=model,chunk_size=chunk_size)
            else:
                self.test = MyData(root, user=mode2-1, subset=150,model=model,chunk_size=chunk_size)


def test_SignFi(root,mode):
    print("hahahahhahahha")
    print(SignFi.__name__)
    data = SignFi(root,mode)
    if mode<10:
        # training set:
        print("the length of training set:{}".format(len(data.train)))
        print("the type of train[0]:{}".format(type(data.train[0])))
        print("shape of train[0][0]:{}".format(data.train[0][0].size()))
        print("shape of train[0][1]:{}".format(data.train[0][1].size()))
        print("shape of train[0][2]:{}".format(data.train[0][2].size()))
        print("train[0][0]:{}".format(data.train[0][0]))
        print("train[0][1]:{}".format(data.train[0][1]))

        # print(data.train[0])
        # testing set:
        print("the length of testing set:{}".format(len(data.test)))
        print("the type of test[0]:{}".format(type(data.test[0])))
        print("the shape of test[0][0]:{}".format(data.test[0][0].shape))
        print("the shape of test[0][1]:{}".format(data.test[0][1].shape))
        print("the shape of test[0][2]:{}".format(data.test[0][2].shape))
        print("test[0][0]:{}".format(data.test[0][0]))
        print("test[0][1]:{}".format(data.test[0][1]))
    else:
        # domain1 set:
        print("the length of domain1 set:{}".format(len(data.domain1)))
        print("the type of domain1[0]:{}".format(type(data.domain1[0])))
        print("shape of domain1[0][0]:{}".format(data.domain1[0][0].size()))
        print("shape of domain1[0][1]:{}".format(data.domain1[0][1].size()))
        print("shape of domain1[0][2]:{}".format(data.domain1[0][2].size()))
        print("domain1[0][0]:{}".format(data.domain1[0][0]))
        print("domain1[0][1]:{}".format(data.domain1[0][1]))

        # print(data.train[0])
        # domain2 set:
        print("the length of domain2 set:{}".format(len(data.domain2)))
        print("the type of domain2[0]:{}".format(type(data.domain2[0])))
        print("the shape of domain2[0][0]:{}".format(data.domain2[0][0].shape))
        print("the shape of domain2[0][1]:{}".format(data.domain2[0][1].shape))
        print("the shape of domain2[0][2]:{}".format(data.domain2[0][2].shape))
        print("domain2[0][0]:{}".format(data.domain2[0][0]))
        print("domain2[0][1]:{}".format(data.domain2[0][1]))


if __name__ == "__main__":
    print("mode 2:\t")
    test_SignFi("../../Datasets/SignFi",2)
    print("mode 5:\t")
    test_SignFi("../../Datasets/SignFi", 5)
    print("mode 6:\t")
    test_SignFi("../../Datasets/SignFi", 6)
    print("mode 7:\t")
    test_SignFi("../../Datasets/SignFi", 7)
    print("mode 12:\t")
    test_SignFi("../../Datasets/SignFi", 12)
    print("mode 67:\t")
    test_SignFi("../../Datasets/SignFi", 67)
    print("mode 76:\t")
    test_SignFi("../../Datasets/SignFi", 76)
    # default_loader("../../Datasets/SignFi")
