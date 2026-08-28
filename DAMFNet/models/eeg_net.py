import torch
import torch.nn as nn
import torch.nn.functional as F

class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()

        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, 16, kernel_size=(1,1)))

        self.branch3 = nn.Sequential(nn.Conv2d(in_channels, 16, kernel_size=(1,1)),
                                     nn.Conv2d(16, 24, kernel_size=(3,1), padding='same'),
                                     # nn.Conv2d(24, 24, kernel_size=(3,1), padding='same')
                                     )

        self.branch5 = nn.Sequential(nn.Conv2d(in_channels, 16, kernel_size=(1,1)),
                                     nn.Conv2d(16, 24, kernel_size=(5,1), padding='same')
                                     )

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=(1,1))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch5 = self.branch5(x)
        branch3 = self.branch3(x)
        branch_pool = F.avg_pool2d(x, kernel_size=(3,1), stride=(1,1), padding=(1,0))
        branch_pool = self.branch_pool(branch_pool)

        return torch.cat((branch1, branch5, branch3, branch_pool), dim=1)

#in_channel=1,out_channels=[16, 32, 32, 64],
class MYEEG(nn.Module):
    def __init__(self,in_channel,out_channels,hidden_size,output_size,drop_prob):
        super(MYEEG,self).__init__()

        self.convs1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channels[0], kernel_size=(1, 4),padding='same'),
            # 公开数据集1 kernel_size=(1, 4)  自己数据集2 kernel_size=(1, 22),
            # nn.Conv2d(in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=(1, 2),padding='same'),
            nn.BatchNorm2d(out_channels[0]),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=(1, 4),padding='same'),
            # 公开数据集1 kernel_size=(1, 4)  自己数据集2 kernel_size=(1, 9),
            # nn.Conv2d(in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=(1, 2),padding='same'),
            nn.BatchNorm2d(out_channels[1]),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )

        self.convs2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=(4, 1),padding='same'),
            # nn.Conv2d(in_channels=out_channels[2], out_channels=out_channels[2], kernel_size=(4, 1),padding='same'),
            nn.BatchNorm2d(out_channels[2]),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)),
            nn.Conv2d(in_channels=out_channels[2], out_channels=out_channels[3], kernel_size=(4,1),padding='same'),
            # nn.Conv2d(in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=(4, 1),padding='same'),
            nn.BatchNorm2d(out_channels[3]),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        )

        self.inception = Inception(32)

        self.convs3 = nn.Sequential(
            nn.Conv2d(in_channels=88, out_channels=32, kernel_size=(4,1),padding='same'),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        )
        self.convs4 = nn.Sequential(
            nn.Conv2d(in_channels=88, out_channels=64, kernel_size=(4,1),padding='same'),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        )
        self.dropout = nn.Dropout(drop_prob)
        self.fc0 = nn.Linear(64*37*2, hidden_size)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.fc2 = nn.Linear(output_size, 2)  # 数据集1 二分类 数据集2 四分类

    def forward(self,x):
        """
        x: (batch,h*w*channels)
        eeg: 600*8*1
        """
        # x = torch.from_numpy(x)
        x = x.to(torch.float32)
        x = x.view(x.shape[0], 1, 600, x.shape[-1])
        x = self.convs1(x)   #1:  32*600*2  2:32*600*12
        # x = self.convs2(x)   #  64*37*4
        x = self.inception(x)
        x = self.convs3(x)   #1: 32*150*2  2: 32*150*12
        x = self.inception(x)
        x = self.convs4(x) #1: 64*37*3  2: 64*37*11
        #64*37*2
        print(x.shape)
        x = x.view(x.shape[0],-1)
        # print(x.shape)
        x = F.relu(self.fc0(x))
        features = self.fc1(x)
        out = self.fc2(self.dropout(features))
        return out, features

if __name__ == "__main__":
    pass