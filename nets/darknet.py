from typing import OrderedDict
import torch
import torch.nn as nn
import math

def convSet(in_channels, out_channels, kernel_size, stride=1, padding=1):
    '''help bulid network
    args:
        inChannel -
        outChannel - 
        kernel_size -
        stride -
    '''
    conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                     stride, padding, bias=False)
    bn = nn.BatchNorm2d(out_channels)
    relu = nn.LeakyReLU(0.1)
    return conv, bn, relu

class ResidualBlock(nn.Module):
    '''build residual block for helping bulid darknet53
    '''
    def __init__(self, CiCo):
        '''
        args:
            inChannels - 
            outChannels - a list
        '''
        super(ResidualBlock, self).__init__()
        self.conv1, self.bn1, self.relu1 = convSet(CiCo[1], CiCo[0],
                                                   kernel_size=1, padding=0)
        self.conv2, self.bn2, self.relu2 = convSet(CiCo[0], CiCo[1],
                                                   kernel_size=3)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out += residual
        return out

class Darknet53(nn.Module):
    '''Bulid darknet53 network
    '''
    def __init__(self):
        super(Darknet53, self).__init__()
        self.conv1, self.bn1, self.relu1 = convSet(3, 32, kernel_size=3)
        self.block1 = self._make_block([32,64], 1)
        self.block2 = self._make_block([64,128], 2)
        self.block3 = self._make_block([128,256], 8)
        self.block4 = self._make_block([256,512], 8)
        self.block5 = self._make_block([512,1024], 4)
        self.out_chan = [256, 512, 1024]

        # 权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_block(self, CiCo, blocks):
        layers = []
        conv, bn, relu = convSet(CiCo[0], CiCo[1], kernel_size=3, stride=2)
        layers.append(('res_conv', conv))
        layers.append(('res_bn', bn))
        layers.append(('res_relu', relu))
        for i in range(blocks):
            layers.append(('residual{}'.format(i), ResidualBlock(CiCo)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.block1(x)
        x = self.block2(x)
        out2 = self.block3(x)
        out1 = self.block4(out2)
        out0 = self.block5(out1)
        return out2, out1, out0

if __name__ == '__main__':
    '''Pass data through the model
    '''
    model = Darknet53()
    random_data = torch.rand((1, 3, 416, 416))
    result3, result2, result1 = model(random_data)
    print(result3.size(), result2.size() ,result1.size())
    
    # print('model.modules()')
    # for i, module in enumerate(model.modules()):
    #     print(i, module)
