import torch
import torch.nn as nn
import torchvision
import sys
import math

'''
    The encoder of ASTER, which is composed of Resnet like conv network, and a multi-layer Bidirectional LSTM network
    to enlarge the feature context, capturing long-range dependencies in both directions.
'''

def conv3x3(in_planes, out_planes, stride=1):
    # 3x3 conv with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x1(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class ResNet_ASTER(nn.Module):
    def __init__(self, with_lstm=True, n_group=1):
        super(ResNet_ASTER, self).__init__()
        self.with_lstm = with_lstm
        self.n_group = n_group

        in_channels = 1
        self.layer0 = nn.Sequential(     # 32 x 140
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.inplanes = 32
        #self.linear = nn.Linear(256*2, 256)
        self.layer1 = self._make_layer(32, 3, [2, 2])  # 16x70
        self.layer2 = self._make_layer(64, 4, [2, 2])  # 8x35
        self.layer3 = self._make_layer(128, 6, [2, 1])  #4x35
        self.layer4 = self._make_layer(256, 6, [2, 1])  # 2x35
        self.layer5 = self._make_layer(512, 3, [2, 1])  #1x35
        self.linear = nn.Linear(512, 512)
       
        if with_lstm:
            self.rnn = nn.LSTM(512, hidden_size=256, bidirectional=True, num_layers=1, batch_first=True)
            self.out_planes = 512
        else:
            self.out_planes = 512

        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != [1, 1] or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes)
            )
        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.size(0)
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        cnn_feature = x5.squeeze(2)   # x5 (N,c,h,w) --> (N,c,w)
        cnn_feature = cnn_feature.transpose(2, 1)  # (N,c,w) --> (N,w,c)  (N,35,512)
        T = cnn_feature.size(1)
        #print('cnn_feature size: ', cnn_feature.size())
                                                                                                                                

        if self.with_lstm:
            rnn_output, (rnn_hidden,c) = self.rnn(cnn_feature)   # [b, T, 512]
            rnn_hidden = self.linear(rnn_output[:,-1,:])
            rnn_output = self.linear(rnn_output)
            
            return rnn_output, rnn_hidden
        
        else:
            return cnn_feature

if __name__ == '__main__':
    x = torch.randn(3, 1, 32, 140)
    model = ResNet_ASTER()
    rnn_feature, rnn_hidden = model(x)
    print(rnn_feature.size())
    print(rnn_hidden.size())
