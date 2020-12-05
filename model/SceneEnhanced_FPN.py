'''
@Author: xxxmy
@Github: github.com/VectXmy
@Date: 2019-09-26
@Email: xxxmy@foxmail.com
'''

import torch.nn as nn
import torch.nn.functional as F
import math
from model.config import DefaultConfig
from model.TextureSceneMudule import TextureSceneMudule

class FPN(nn.Module):
    '''only for resnet50,101,152'''
    
    def __init__(self,features=256,use_p5=True):
        super(FPN,self).__init__()
        self.prj_5 = nn.Conv2d(2048+512, features, kernel_size=1)
        self.prj_4 = nn.Conv2d(1024+512, features, kernel_size=1)
        self.prj_3 = nn.Conv2d(1024, features, kernel_size=1)
        self.conv_5 =nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_4 =nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_3 =nn.Conv2d(features, features, kernel_size=3, padding=1)
        if use_p5:
            self.conv_out6 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        else:
            self.conv_out6 = nn.Conv2d(2048, features, kernel_size=3, padding=1, stride=2)
        self.conv_out7 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        self.use_p5=use_p5
        self.apply(self.init_conv_kaiming)
    def upsamplelike(self,inputs):
        src,target=inputs
        return F.interpolate(src, size=(target.shape[2], target.shape[3]),
                    mode='nearest')
    
    def init_conv_kaiming(self,module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, a=1)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self,x):
        C3,C4,C5=x
        P5 = self.prj_5(C5)
        P4 = self.prj_4(C4)
        P3 = self.prj_3(C3)
        P4 = P4 + self.upsamplelike([P5,C4])
        P3 = P3 + self.upsamplelike([P4,C3])
        P3 = self.conv_3(P3)
        P4 = self.conv_4(P4)
        P5 = self.conv_5(P5)
        P5 = P5 if self.use_p5 else C5
        P6 = self.conv_out6(P5)
        P7 = self.conv_out7(F.relu(P6))
        return [P3,P4,P5,P6,P7]



class SE_FPN(nn.Module):
    def __init__(self,config=None):
        super(SE_FPN,self).__init__()
        self.TextureSceneMudule=TextureSceneMudule(config.input_channel,config.output_channel,
            config.pooling_size,config.output_class,config.scene_use_GN)
        self.config=config
        self.fpn=FPN(config.fpn_out_channels,use_p5=config.use_p5)

    def forword(self,C3,C4,C5):

        S3,SL_pred=self.TextureSceneMudule(C3)

        AMP_4=nn.AdaptiveMaxPool2d((S3.shape[2]//2,S3.shape[3]//2))
        AMP_5=nn.AdaptiveMaxPool2d((S3.shape[2]//4,S3.shape[3]//4))

        S4=AMP_4(S3)
        S5=AMP_5(S4)

        F3=torch.cat((C3,S3),dim=1)
        F4=torch.cat((C4,S4),dim=1)
        F5=torch.cat((C5,S5),dim=1)

        P_list=self.fpn([F3,F4,F5])
        return P_list,SL_pred


