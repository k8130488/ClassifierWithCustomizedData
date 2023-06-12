import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=[.25, .25, .25], gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, output, target):
        y_pred = output
        y_true = target
        alpha = torch.cuda.FloatTensor(self.alpha)
        gamma = self.gamma
        ce = -torch.multiply(y_true, torch.log(y_pred))
        factor = torch.pow((torch.ones_like(y_true) - y_pred), gamma)
        fl = torch.matmul(torch.multiply(factor, ce), alpha)
        focal_loss = fl.mean()
        return focal_loss
