import torch
from torch import nn
import torch.nn.functional as F

import timm


class BaseTransformer(nn.Module):
    def __init__(self, name="vit_base_patch16_224", pretrained=True, n_classes=1):
        super(BaseTransformer, self).__init__()
        self.model = timm.create_model(name, pretrained=pretrained)
        self.n_classes = n_classes
        # self.model.head = nn.Sequential(nn.Dropout(p =  dropout), nn.Linear(self.model.head.in_features, 1))
        self.model.head = nn.Linear(self.model.head.in_features, n_classes)

    def forward(self, x):
        x = self.model(x)
        if self.n_classes == 1:
            return x.ravel()
        else:
            return x


class BaseCNN_resnext(nn.Module):
    def __init__(self, name="ig_resnext101_32x8d", pretrained=True, n_classes=1):
        super(BaseCNN_resnext, self).__init__()
        self.model = timm.create_model(name, pretrained=pretrained)
        self.n_classes = n_classes
        self.n_features = self.model.num_features
        self.model.fc = nn.Identity()
        self.fc = nn.Linear(self.n_features, self.n_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        if self.n_classes == 1:
            return x.ravel()
        else:
            return x


class BaseCNN_effnet(nn.Module):
    def __init__(self, name="tf_efficientnet_b4_ns", pretrained=True, n_classes=1):
        super(BaseCNN_effnet, self).__init__()
        self.model = timm.create_model(name, pretrained=pretrained)
        self.n_classes = n_classes
        self.n_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.fc = nn.Linear(self.n_features, self.n_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        if self.n_classes == 1:
            return x.ravel()
        else:
            return x


class BCE_scaled(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, target):
        return F.binary_cross_entropy_with_logits(preds, target / 100)


class MSE_scaled(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, target):
        return F.mse_loss(sigmoid(preds) * 100, target)


class CE_scaled(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, preds, target):
        return F.cross_entropy(preds, target, label_smoothing=self.label_smoothing)


class MSE_CE_scaled(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, preds, target, n_classes):
        return F.mse_loss((torch.argmax(preds, dim=1) + 0.5) * 100 / self.n_classes, target)


class MSE_CE_scaled_averaging(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, preds, target):
        middles = torch.arange(0, 100, 100 / self.n_classes) + 100 / self.n_classes
        middles = middles.cuda()
        return F.mse_loss(torch.sum(F.softmax(preds, dim=1) * middles, dim=1), target)


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))
