import torch.nn as nn
import torch

# Defining your CNN model
# We have defined the baseline model
class baseline_Net(nn.Module):

    def __init__(self, classes):
        super(baseline_Net, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(96, 256, 5),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(256, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.b4 = nn.Sequential(
            nn.Conv2d(384, 192, 3),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.b5 = nn.Sequential(
            nn.Conv2d(192, 256, 1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d((1,1)),
            nn.ReLU(inplace=True)
        )
        # self.b6 = nn.Sequential(
        #     nn.Conv2d(128, 256, 1),
        #     nn.BatchNorm2d(256),
        #     nn.MaxPool2d((1, 1)),
        #     nn.ReLU(inplace=True)
        # )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.Dropout(),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, classes),
        )

    def forward(self, x):
        out1 = self.b2(self.b1(x))
        out2 = self.b4(self.b3(out1))
        out3 = self.b5(out2)
        # out4 = self.b6(out3)
        out_avg = self.avg_pool(out3)
        out_flat = out_avg.view(-1, 256)
        out4 = self.fc2(self.fc1(out_flat))

        return out4

    def accuracy(self, x, y):
        # total correct predictions / total samples
        correct = (x == y)
        correct = torch.sum(correct).item()
        return correct/y.shape[0]
