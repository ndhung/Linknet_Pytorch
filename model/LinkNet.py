import torch
import torch.nn as nn
import torchvision.models as models
import math


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, st, adjs):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(nn.Conv2d(in_channels, in_channels // 4, 1), 
                                     nn.BatchNorm2d(in_channels // 4, 1e-3),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, st, 1, adjs),
                                     nn.BatchNorm2d(in_channels // 4, 1e-3),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels // 4, out_channels, 1),
                                     nn.BatchNorm2d(out_channels, 1e-3),
                                     nn.ReLU())
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x):
        return self.decoder(x)


class LastBlock(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(LastBlock, self).__init__()
        self.last_block = nn.Sequential(nn.ConvTranspose2d(in_channels, 32, 3, 2, 1, 1),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(32, 32, 3, 1, 1),
                                        nn.BatchNorm2d(32, 1e-3),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(32, num_classes, 2, 2))
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.last_block(x)


class LinkNet(nn.Module):
    def __init__(self, num_classes):
        super(LinkNet, self).__init__()
        # using pretrained resnet18 model
        resnet18 = list(models.resnet18(pretrained=True).children())
        self.initial_block = nn.Sequential(resnet18[0], resnet18[1], resnet18[2], resnet18[3])
        self.layer1 = resnet18[4]
        self.layer2 = resnet18[5]
        self.layer3 = resnet18[6]
        self.layer4 = resnet18[7]

        # Decoder section with bypassed information
        self.decoder1 = Decoder(64, 64, 1, 0)
        self.decoder2 = Decoder(128, 64, 2, 1)
        self.decoder3 = Decoder(256, 128, 2, 1)
        self.decoder4 = Decoder(512, 256, 2, 1)

        # Decoder section without bypassed information
        self.last_block = LastBlock(64, num_classes)

    def forward(self, x):
        x = self.initial_block(x)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        x = self.layer4(out3)
        x = self.decoder4(x)
        x = x + out3
        x = self.decoder3(x)
        x = x + out2
        x = self.decoder2(x)
        x = x + out1
        x = self.decoder1(x)
        x = self.last_block(x)
        return x
