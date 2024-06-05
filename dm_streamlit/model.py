import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        # self.conv = nn.Conv2d(3, 1, 3, 1, 1)
        self.fc = nn.Linear(3 * 32 * 32, 32 * 32)

        self.fc1 = nn.Linear(32 * 32, 512)
        self.ac = nn.LeakyReLU()
        self.fc2 = nn.Linear(512, 100)
        self.drop = nn.Dropout(0.01)


    def forward(self, x):
        B,_, _, _ = x.shape
        y = self.drop(self.ac(self.fc(x.view(B, -1))))
        y = self.drop(self.ac(self.fc1(y)))
        return self.fc2(y)


class SE(nn.Module):

    def __init__(self, in_chnls, ratio):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return x*torch.sigmoid(out)


class SENet(nn.Module):
    def __init__(self, num_classes=100):
        super(SENet, self).__init__()
        self.base_model = models.resnet50(pretrained=False)
        self.se_module = SE(2048, 16)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)

        x = self.se_module(x)
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x



