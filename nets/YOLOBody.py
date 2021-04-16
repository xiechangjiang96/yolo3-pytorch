from math import fabs
from typing import OrderedDict

import torch
from nets.darknet import Darknet53
import torch.nn as nn

class YOLO3(nn.Module):
    def __init__(self):
        super(YOLO3, self).__init__()
        self.backbone = Darknet53()
        self.out_chan = self.backbone.out_chan # [256, 512, 1024]
        self.small_branch = self._conv2d_set(self.out_chan[-1], 512, 1024, 75)
        self.conv1 = self._conv2d_group(512, 256, 1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.medium_branch = self._conv2d_set(768, 256, 512, 75)
        self.conv2 = self._conv2d_group(256, 128, 1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.large_branch = self._conv2d_set(384, 128, 256, 75)

    def forward(self, x):
        out2, out1, out0 = self.backbone(x)
        small_prediction, small_branch_out = self._branch_sequential(out0, self.small_branch)
        tem = self.conv1(small_branch_out)
        tem = self.upsample1(tem)
        in1 = torch.cat([out1, tem], 1)
        medium_prediction, medium_branch_out = self._branch_sequential(in1, self.medium_branch)
        tem = self.conv2(medium_branch_out)
        tem = self.upsample2(tem)
        in2 = torch.cat([out2, tem], 1)
        large_prediction, drop = self._branch_sequential(in2, self.large_branch)
        return small_prediction, medium_prediction, large_prediction

    def _branch_sequential(self, x, branch):
        for i, m in enumerate(branch):
            x = m(x)
            if i == 4:
                branch_out = x
        return x, branch_out

    def _conv2d_set(self, in_chan, medium_chan, out_chan, cla_chan):
        return nn.ModuleList([
            self._conv2d_group(in_chan, medium_chan, 1),
            self._conv2d_group(medium_chan, out_chan, 3),
            self._conv2d_group(out_chan, medium_chan, 1),
            self._conv2d_group(medium_chan, out_chan, 3),
            self._conv2d_group(out_chan, medium_chan, 1),
            # The result of upper layer will be restored to participate
            # in concatenatiion of medium (or large) branch 
            self._conv2d_group(medium_chan, out_chan, 3),
            nn.Conv2d(out_chan, cla_chan, kernel_size=1, stride=1, padding=0, bias=False)
        ])

    def _conv2d_group(self, in_chan, out_chan, kernel_size):
        assert kernel_size in [1,3]
        padding = 1 if kernel_size==3 else 0
        return nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_chan, out_chan, kernel_size, stride=1, padding=padding, bias=False)),
            ('bn', nn.BatchNorm2d(out_chan)),
            ('relu', nn.LeakyReLU(0.1))
        ]))

if __name__ == '__main__':
    '''test whether the model is normal. Please copy this code and paste it in
    the upper directory, and then run it. Maybe there are some code need you 
    supplement.
    '''
    model = YOLO3()
    rand_data = torch.rand([1, 3, 416, 416])
    small, medium, large = model(rand_data)
    print(small.shape, medium.shape, large.shape)