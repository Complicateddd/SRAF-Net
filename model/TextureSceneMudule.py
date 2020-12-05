import torch.nn as nn
from .config import DefaultConfig
import math
def conv3x3(in_planes, out_planes, stride=1,dilation1=None):
    """3x3 convolution with padding"""
    if dilation1:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1,bias=False)
    else:

        return nn.Conv2d(in_planes, out_planes, kernel_size=3, dilation=dilation1,stride=stride,padding=1,bias=False)


class SceneRefineBlock(nn.Module):
    def __init__(self,inplanes, planes, stride=1, downsample=None,use_GN=False):
        super(SceneRefineBlock, self).__init__()
        self.downsample=None
        if inplanes!=planes:
            if use_GN:
                self.downsample=nn.Sequential(
                nn.Conv2d(inplanes,planes,kernel_size=1,stride=stride,bias=False),
                nn.GroupNorm(32,planes),)
            else:
                self.downsample=nn.Sequential(
                nn.Conv2d(inplanes,planes,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(planes),)
        if use_GN:
            self.bn1=nn.GroupNorm(32,planes)
            self.bn2=nn.GroupNorm(32,planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
        self.conv1 = conv3x3(inplanes,planes,dilation1=3,stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes,planes,dilation1=1)
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


class TextureSceneMudule(nn.Module):
    def __init__(self,input_channel=2048,output_channel=1024,pooling_size=3,output_class=3,scene_use_GN=False):
        super(TextureSceneMudule, self).__init__()
        self.input_channel=input_channel
        self.pooling_size=pooling_size
        self.output_channel=output_channel
        self.output_class=output_class
        self.resblock1=SceneRefineBlock(self.input_channel,self.output_channel,None,scene_use_GN)
        self.resblock2=SceneRefineBlock(self.output_channel,self.output_channel,None,scene_use_GN)
        self.pooling=nn.AdaptiveMaxPool2d((pooling_size,pooling_size))
        self.linaer=nn.Linear(self.output_channel*self.pooling_size*self.pooling_size,self.output_class)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def init_conv_RandomNormal(self,module,std=0.01):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    def forward(self,x):
        
        output1=self.resblock1(x)
        feature=self.resblock2(output1)
        output3=self.pooling(feature)
        output4=self.linaer(output3.view(-1,self.output_channel*self.pooling_size*self.pooling_size))
        return feature,output4