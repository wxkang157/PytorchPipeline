import math
import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable


class YourBackbone(nn.Module):
    def __init__(self, pretrained):
        super(YourBackbone, self).__init__()
        self.model = models.resnet50(pretrained=False)
        self.pool = nn.AvgPool2d(2, 2)

        self._initialize_weights(pretrained)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.pool(x)

        return x

    def _initialize_weights(self, pretrained):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        if pretrained is not None:
            pretrained_dict = torch.load(pretrained)
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)


class YourNeck(nn.Module):
    def __init__(self, cfg):
        super(YourNeck, self).__init__()
        self.conv1 = nn.Conv2d(1024, 512, 3, 1, 3//2)
        # Omit a bunch of operations and add them according to your needs
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        # Omit a bunch of operations and add them according to your needs
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Head(nn.Module):
    def __init__(self, numclasses):
        super(Head, self).__init__()
        self.classifier1 = nn.Linear(1024 * 8 * 8, 2048)
        self.classifier2 = nn.Linear(2048, numclasses)
        self._initialize_weights()

    def forward(self, x):
        x = self.classifier1(x)
        x = self.classifier2(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class YourModel(nn.Module):
    def __init__(self, pretrained, numclasses):
        super(YourModel, self).__init__()
        self.backbone = YourBackbone(pretrained)
        self.neck = YourNeck()
        self.flatten = Flatten()
        self.head = Head(numclasses)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.flatten(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    pretrained = None
    numclasses = 10
    input = Variable(torch.FloatTensor(2, 3, 256, 256))
    model = YourModel(pretrained, numclasses)
    out = model(input)
