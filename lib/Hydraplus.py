import torch
import torch.nn as nn
import torch.nn.functional as F
from AF import AF
from MNet import MNet

class HP(nn.Module):

    def __init__(self, num_classes=26):
        super(HP, self).__init__()
        self.MNet = MNet(feat_out=True)
        self.AF1 = AF(feat_out=True, af_name="AF1")
        self.AF2 = AF(feat_out=True, af_name="AF2")
        self.AF3 = AF(feat_out=True, af_name="AF3")

        self.fc = nn.Linear(512 * 73, num_classes)

    def forward(self, x):
        _, _, _, F0 = self.MNet(x)
        F1 = self.AF1(x)
        F2 = self.AF2(x)
        F3 = self.AF3(x)

        ret = torch.cat((F0, F1, F2, F3), dim=1)
        # 9 x 9 x (512x(24x3 + 1))

        ret = F.avg_pool2d(ret, kernel_size=9, stride=1)

        # 1 x 1 x (512 x 73)

        ret = F.dropout(ret, training=self.training)
        # 1 x 1 x (512 x 73)
        ret = ret.view(ret.size(0), -1)
        # 512 x 73

        ret = self.fc(ret)
        # (num_classes)

        return ret