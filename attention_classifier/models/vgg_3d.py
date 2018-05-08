import torch.nn as nn
import math


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3 * 2, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            #nn.Linear(4096, 4096),
            #nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(1000, num_classes),
        )
        if init_weights:
            self._initialize_weights()

        #print(self.features)



    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
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