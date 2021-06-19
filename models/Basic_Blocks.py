import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
from torchsummary import summary
import pytorch_lightning as pl


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ShallowExtractor(nn.Module):
    def __init__(self,inchannel,groups):
        super(ShallowExtractor,self).__init__()
        self.groups = groups
        self.ouchannel = 128 * self.groups
        self.conv1 = nn.Conv1d(inchannel, self.ouchannel, kernel_size=7, stride=2, padding=3, bias=False,
                               groups=self.groups)
        self.bn1 = nn.BatchNorm1d(self.ouchannel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class DeepExtractor(nn.Module):
    def __init__(self,layers,strides,block,inchannel):
        super(DeepExtractor,self).__init__()
        self.inplanes = inchannel
        layer = []
        for i, element in enumerate(layers):
            layer.append(self._make_layer(block, self.inplanes, element, stride=strides[i]))
        self.layer = nn.Sequential(*layer)

    def _make_layer(self, block, planes, num_layer, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_layer):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        c = self.layer(x)
        return c


class Classifier(nn.Module):
    def __init__(self,inchannel,num_categories):
        super(Classifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(inchannel, inchannel * 2, kernel_size=3, stride=1,
                      padding=0, bias=False),
            nn.BatchNorm1d(inchannel * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.result = nn.Linear(inchannel * 2, num_categories)

    def forward(self,x):
        c = self.layer(x)
        c = c.view(c.size(0),-1)
        result = self.result(c)
        return result


class MTLbasic(nn.Module):
    def __init__(self,inchannel_SE, groups_SE, layers_DE, strides_DE,block_DE, inchannel_DE, inchannel_Cl,
                 num_categories, num_tasks,device):
        '''
        inchannel_SE and groups_SE arer the input channels (num. subcarriers) and the groups (links) of grouping convolution, respectively.
        layers_DE, strides_DE,block_DE, inchannel_DE: the first two are the number of layers and the strides of convolution.
                                                      layers_DE, strides_DE are lists. block_DE is the type of convolution blocks
                                                      in this model: Bottleneck or BasicBlock.
        inchannel_Cl,num_categories, num_tasks: num_categories is a list, and the element is the categories of the corresponding task.
        '''
        super(MTLbasic, self).__init__()
        self.SE = ShallowExtractor(inchannel=inchannel_SE,groups=groups_SE).to(device)
        self.DE = DeepExtractor(layers=layers_DE,strides=strides_DE,block=block_DE,inchannel=inchannel_DE).to(device)
        self.classifiers = []
        for i in range(num_tasks):
            self.classifiers.append(Classifier(inchannel=inchannel_Cl,num_categories=num_categories[i]).to(device))

    def forward(self, x):
        shallow_feature = self.SE(x)
        deep_feature = self.DE(shallow_feature)
        results = []
        for element in self.classifiers:
            results.append(element(deep_feature))
        return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    # SE = ShallowExtractor(inchannel=52,groups=1).to(device)
    # summary(SE, (52, 192))
    # DE = DeepExtractor(inchannel=128,layers=[1,2,2],strides=[1,2,2],block=BasicBlock).to(device)
    # summary(DE, (128, 48))
    # Cl = Classifier(inchannel=128,num_categories=6).to(device)
    # summary(Cl, (128, 12))

    x = torch.rand([2,52,192]).to(device)

    MTL_basic = MTLbasic(inchannel_SE=52, groups_SE=1, layers_DE=[1,2,2], strides_DE=[1,2,2],block_DE=BasicBlock,
                         inchannel_DE=128, inchannel_Cl=128,num_categories=[6,12], num_tasks=2,device=device).to(device)
    # summary(MTL_basic, (52, 192))
    result = MTL_basic(x)
    print(len(result))
    print(len(result[0]))
    print(len(result[0][0]))
    print(len(result[0][1]))
    print(len(result[1][0]))
    print(len(result[1][1]))

    '''
    2
    2
    6
    6
    12
    12
    '''


