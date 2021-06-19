from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable


class Pair_metric(nn.Module):
    def __init__(self, metric_method,average=True, norm_flag = False):
        super(Pair_metric, self).__init__()
        self.metric_method = metric_method
        self.average = average
        self.norm_flag = norm_flag

    def forward(self, input, target):
        # input size: [batch, channel, ...] ; output size: [batch, channel, ...];
        batch_size = input.shape[0]
        input_2d = torch.reshape(input,(batch_size,-1))
        target_2d = torch.reshape(target,(batch_size,-1))

        if self.norm_flag:
            input_norm = torch.linalg.norm(input_2d,ord = 2, dim = 1, keepdim=True )
            target_norm = torch.linalg.norm(target_2d,ord = 2, dim = 1,keepdim=True )
            input_2d = input_2d / input_norm.expand_as(input_2d)
            target_2d = target_2d / target_norm.expand_as(target_2d)
        if self.average:
            temp = torch.pow(input_2d - target_2d, 2)
            temp2 = torch.sum(temp,dim=1)
            return torch.mean(temp2,dim=0)
        return torch.pow(input_2d - target_2d, 2).sum()


if __name__ == '__main__':
    input1 = torch.Tensor([[[1,2],[2,2],[2,3]],[[3,2],[3,2],[4,6]]])
    input2 = torch.Tensor([[[1,2],[3,4],[4,2]],[[2,1],[3,3],[2,8]]])
    metr1 = Pair_metric('dd',)
    metr2 = Pair_metric('dd', average=False)
    metr3 = Pair_metric('dd', norm_flag=True)
    metr4 = Pair_metric('dd', average=False, norm_flag=True)
    print(metr1(input1,input2))
    print(metr2(input1,input2))
    print(metr3(input1,input2))
    print(metr4(input1,input2))
