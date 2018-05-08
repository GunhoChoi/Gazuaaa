import torch.nn as nn
import math
from .layers.attention_layer import *

class VGG(nn.Module):

    def __init__(self, features, num_classes=7, init_weights=True,inter_channel=64,channel_list=[64,128,256,512,512]):
        super(VGG, self).__init__()
        self.features = features
        self.num_classes = num_classes
        self.inter_channel = inter_channel
        self.channel_list = channel_list
        self.GAP_layer = nn.AdaptiveAvgPool3d(output_size=(1,1,1))
        
        self.feature_1 = nn.Sequential(*list(self.features.children())[:7])
        self.feature_2 = nn.Sequential(*list(self.features.children())[7:14])
        self.feature_3 = nn.Sequential(*list(self.features.children())[14:27])
        self.feature_4 = nn.Sequential(*list(self.features.children())[27:40])
        self.feature_5 = nn.Sequential(*list(self.features.children())[40:])

        self.classifier_1 = nn.Sequential(
            nn.Linear(sum(self.channel_list[2:]),100),
            nn.ReLU(inplace=True),
            #nn.BatchNorm1d(100),
            nn.Linear(100,self.num_classes),
        )

        if init_weights:
            self._initialize_weights()


    def forward(self, x):
        x = self.feature_1(x)
        x = self.feature_2(x)
        x = self.feature_3(x)
        out_3 = x   # [batch, 256, 12, 12, 8]
        x = self.feature_4(x)
        out_4 = x   # [batch, 512, 6, 6, 4]
        x = self.feature_5(x)
        out_5 = x   # [batch, 512, 3, 3, 2]

        self.attention_1 = Attention_Layer_3D(out_3.size(),out_5.size(),self.inter_channel)
        self.attention_2 = Attention_Layer_3D(out_4.size(),out_5.size(),self.inter_channel)

        attention_1,weighted_feature_1 = self.attention_1(out_3,out_5)
        attention_2,weighted_feature_2 = self.attention_2(out_4,out_5)
        gap_out = self.GAP_layer(out_5)

        concat_x = torch.cat((weighted_feature_1,weighted_feature_2,gap_out),dim=1)
        concat_x = concat_x.view(concat_x.size()[0],sum(self.channel_list[2:]))
        x = self.classifier_1(concat_x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal(m.weight, 0, 0.01)
                nn.init.constant(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool3d(kernel_size=2, stride=2)]
        else:
            conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv3d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, **kwargs):
    model = VGG(make_layers(cfg['A']), **kwargs)
    return model


def vgg11_bn(pretrained=False, **kwargs):
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model


def vgg13(pretrained=False, **kwargs):
    model = VGG(make_layers(cfg['B']), **kwargs)
    return model


def vgg13_bn(pretrained=False, **kwargs):
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model


def vgg16(pretrained=False, **kwargs):
    model = VGG(make_layers(cfg['D']), **kwargs)
    return model


def vgg16_bn(pretrained=False, **kwargs):
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model


def vgg19(pretrained=False, **kwargs):
    model = VGG(make_layers(cfg['E']), **kwargs)
    return model


def vgg19_bn(pretrained=False, **kwargs):
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model