import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


# 定義類神經網路模型
class ResNet50(nn.Module):
    def __init__(self, out_num=4):
        super(ResNet50, self).__init__()

        # 載入 ResNet18 類神經網路結構
        self.model = models.resnet50(pretrained=True)

        # 鎖定 ResNet18 預訓練模型參數
        # for param in self.model.parameters():
        #    param.requires_grad = False

        # 修改輸出層輸出數量
        self.model.fc = nn.Linear(2048, out_num)

    def forward(self, x):
        logits = self.model(x)
        return F.softmax(logits, dim=-1)


class AlexNet(nn.Module):
    def __init__(self, out_num=4):
        super(AlexNet, self).__init__()
        self.model = models.alexnet(pretrained=True)
        self.model.classifier[-1] = nn.Linear(4096, out_num)

    def forward(self, x):
        logits = self.model(x)
        return F.softmax(logits, dim=-1)
