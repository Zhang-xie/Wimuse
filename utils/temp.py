import zipfile
from pathlib import Path
import numpy as np
import io
import torch.utils.data as data
import torch
import os
import itertools,functools
from torch.utils.data import TensorDataset, DataLoader


def load_npy_from_bytes(bytes_data):
    return np.load(io.BytesIO(bytes_data))


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


class Mydata(data.Dataset):
    def __init__(self, root, roomid=None,userid=None,location=None,orientation=None,receiverid=None,model=None,chunk_size=None,batch_size=50,sampleid=None,mode=None):
        super(Mydata, self).__init__()
        self.root = root
        self.model = model
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.batch_idx = 0
        self.num_class = 6
        self.mode = mode

        self.rm_info_all = np.load(root/"rm_info_all.npy")  # this is the label of the deleted wrong data

        multi_label = np.load(root/"multi_label.npy")
        # total number: 98789
        self.room_label = multi_label["roomid"]  # {1, 2, 3}
        self.userid = multi_label["userid"]      # {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}
        self.total_samples = len(multi_label)
        temp_gesture = multi_label["gesture"]    # {1, 2, 3, 4, 6, 9}
        for i,data in enumerate(temp_gesture):
            if temp_gesture[i] == 9:
                temp_gesture[i] = 5
        self.gesture = temp_gesture  # {1, 2, 3, 4, 5, 6}
        self.location = multi_label["location"]  # {1, 2, 3, 4, 5}
        self.orientation = multi_label["face_orientation"]  # {1, 2, 3, 4, 5}
        self.sampleid = multi_label["sampleid"]   # {1, 2, 3, 4, 5}
        self.receiverid = multi_label["receiverid"]  # {1, 2, 3, 4, 5, 6}

        self.f_amp = zipfile.ZipFile(self.root/"amp.zip",mode="r")
        self.f_pha = zipfile.ZipFile(self.root /"pha.zip", mode="r")

        self.select = np.ones(self.total_samples, dtype=np.bool)
        self.room_select = np.ones(self.total_samples, dtype=np.bool)
        self.user_select = np.ones(self.total_samples, dtype=np.bool)
        self.loc_select = np.ones(self.total_samples, dtype=np.bool)
        self.ori_select = np.ones(self.total_samples, dtype=np.bool)
        self.receiver_select = np.ones(self.total_samples, dtype=np.bool)
        self.sample_select = np.ones(self.total_samples, dtype=np.bool)
        index_temp = np.arange(self.total_samples)

        if roomid is not None:
            self.room_select = functools.reduce(np.logical_or,[*[self.room_label == j for j in roomid]])
            self.select = np.logical_and(self.select,self.room_select)
        if userid is not None:
            self.user_select = functools.reduce(np.logical_or,[ *[self.userid == j for j in userid]])
            self.select = np.logical_and(self.select, self.user_select)
        if location is not None:
            self.loc_select = functools.reduce(np.logical_or,[*[self.location == j for j in location]])
            self.select = np.logical_and(self.select, self.loc_select)
        if orientation is not None:
            self.ori_select = functools.reduce(np.logical_or,[ *[self.orientation == j for j in orientation]])
            self.select = np.logical_and(self.select, self.ori_select)
        if receiverid is not None:
            self.receiver_select = functools.reduce(np.logical_or,[ *[self.receiverid == j for j in receiverid]])
            self.select = np.logical_and(self.select, self.receiver_select)
        if sampleid is not None:
            self.sample_select = functools.reduce(np.logical_or, [*[self.sampleid == j for j in sampleid]])
            self.select = np.logical_and(self.select, self.sample_select)

        self.index = index_temp[self.select]  # the data for a specified task

        # self.data_check()

        choosed_label = self.gesture[self.index]
        num_sample_per_class = []
        self.sample_index_per_class = []
        for i in range(1,self.num_class+1):
            temp = self.index[np.where(choosed_label == i)]
            num_sample_per_class.append(len(temp))
            self.sample_index_per_class.append(temp)
        self.min_num_sample_class = min(num_sample_per_class)  # find the minimal number of samples of all classes
        self.num_batch = self.min_num_sample_class // self.batch_size

    def data_check(self):
        pass

    def __getitem__(self, sample_index):
        if self.mode is not None:
            amp_sample = load_npy_from_bytes(self.f_amp.read(str(sample_index)))
            pha_sample = load_npy_from_bytes(self.f_pha.read(str(sample_index)))
            amp_sample = amp_sample.astype(np.float32)  # shape [time,3,30]
            pha_sample = pha_sample.astype(np.float32)  # shape [time,3,30]
            sample = np.concatenate((amp_sample,pha_sample),axis=2)
        else:
            amp_sample = load_npy_from_bytes(self.f_amp.read(str(sample_index)))
            sample = amp_sample.astype(np.float32)  # shape [time,3,30]

        ges_label = self.gesture[sample_index] - 1    # {0,1, 2, 3, 4, 5}
        ges_label = torch.tensor(ges_label).type(torch.LongTensor)
        loc_label = self.location[sample_index] - 1   # {0,1, 2, 3, 4, 5}
        loc_label = torch.tensor(loc_label).type(torch.LongTensor)
        room_label = self.room_label[sample_index] - 1
        room_label = torch.tensor(room_label).type(torch.LongTensor)
        ori_label = self.orientation[sample_index] - 1
        ori_label = torch.tensor(ori_label).type(torch.LongTensor)
        user_label = self.userid[sample_index] - 1
        user_label = torch.tensor(user_label).type(torch.LongTensor)

        if self.model is not None:
            samp, samp_res = split_array_bychunk(sample,self.chunk_size,include_residual=False)  # shape list{[chunk,3,30],repeat}
            samp += [sample[-self.chunk_size:],]
            sample = torch.Tensor(np.array(samp))
            sample = sample.permute(0,2,1,3).type(torch.FloatTensor)  # shape [repeat,3,chunk,30 or 60]
        else:
            sample = torch.from_numpy(sample).type(torch.FloatTensor)
            sample = sample.permute(1,2,0).reshape(90, -1)
        return sample,ges_label,loc_label,room_label,ori_label,user_label

    def get_choose_label(self,id):
        if id == "user":
            return self.userid[self.index]
        if id =="room":
            return self.room_label[self.index]
        if id =="location":
            return self.location[self.index]
        if id =="orientation":
            return self.orientation[self.index]
        if id =="gesture":
            return self.gesture[self.index]
        if id =="receiver":
            return self.receiverid[self.index]
        if id =="sampleid":
            return self.sampleid[self.index]

    def __len__(self):
        return len(self.index)

    def __del__(self):
        self.f_pha.close()
        self.f_amp.close()


class Widar():
    def __init__(self, root, roomid=None,userid=None,location=None,orientation=None,receiverid=None,model=None,chunk_size=None,batch_size=50,sampleid=None,mode=None):
        self.full_data = Mydata(root, roomid=roomid,userid=userid,location=location,orientation=orientation,receiverid=receiverid,
                                model=model,chunk_size=chunk_size,batch_size=batch_size,sampleid=sampleid,mode=mode)
        train_size = int(0.8 * len(self.full_data))
        test_size = len(self.full_data) - train_size
        self.train, self.test = torch.utils.data.random_split(self.full_data, [train_size, test_size])


if __name__ == "__main__":

    def collate(x):
        return list(zip(*x))

    def pp(a):
        print("room:", np.unique(a.full_data.get_choose_label("room")))
        print("gesture:", np.unique(a.full_data.get_choose_label("gesture")))
        print("orientation:", np.unique(a.full_data.get_choose_label("orientation")))
        print("user:", np.unique(a.full_data.get_choose_label("user")))
        print("location:", np.unique(a.full_data.get_choose_label("location")))
        print("receiver:", np.unique(a.full_data.get_choose_label("receiver")))
        print("sampleid:", np.unique(a.full_data.get_choose_label("sampleid")))

        print(len(a.train))
        print(len(a.test))
        print(len(a.train) + len(a.test))
        print(type(a.train[0]))
        print(a.train[0])
        print(a.train[0][2].shape)

    root = Path("F:/Widar3.0ReleaseData/np_f_denoise/Widar")

    a4 = Widar(root, roomid=[1], receiverid=[1], userid=None, location=None, orientation=None, model="LSTM",
                      chunk_size=50,  sampleid=[1,2],batch_size=5,mode="amp_pha")
    pp(a4)

    print('DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD')

    a3 = Widar(root, roomid=[1], receiverid=[1], userid=None, location=None, orientation=None, model="LSTM",
               chunk_size=50, sampleid=[1, 2], batch_size=5, mode=None)
    pp(a3)




