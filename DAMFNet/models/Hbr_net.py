import torch
import torch.nn as nn
import torch.nn.functional as F
#out_channels = [8,16] kernel_size1,kernel_size3,kernel_size5,kernel_size
#(1,1),(3,1)(5,1),(7,1),(1,1)
class Inception(nn.Module):
    def __init__(self, in_channel,out_channels,kernel_size1,kernel_size3,kernel_size5,kernel_size):
        super(Inception, self).__init__()

        self.branch1 = nn.Sequential(nn.Conv2d(in_channel, out_channels[0], kernel_size=kernel_size1))

        self.branch3 = nn.Sequential(nn.Conv2d(in_channel, out_channels[0], kernel_size=kernel_size1),
                                     nn.Conv2d(out_channels[0], out_channels[1], kernel_size=kernel_size3, padding='same'),
                                     # nn.Conv2d(24, 24, kernel_size=(5,1), padding='same')
                                     )

        self.branch5 = nn.Sequential(nn.Conv2d(in_channel, out_channels[0], kernel_size=kernel_size1),
                                     nn.Conv2d(out_channels[0], out_channels[1], kernel_size=kernel_size5, padding='same')
                                     )

        self.branch_pool = nn.Conv2d(in_channel, out_channels[1], kernel_size=kernel_size)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch5 = self.branch5(x)
        branch3 = self.branch3(x)
        branch_pool = F.avg_pool2d(x, kernel_size=(3,1), stride=(1,1), padding=(1,0))
        branch_pool = self.branch_pool(branch_pool)

        return torch.cat((branch1, branch5, branch3,branch_pool), dim=1)

#in_channel=1,out_channels=[16, 16, 32],
class MYHBR(nn.Module):
    def __init__(self,in_channel,out_channels,hidden_size,drop_prob):
        super(MYHBR,self).__init__()

        self.convs1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channels[0], kernel_size=(1, 12),padding='same'),
            # 数据集1  (1, 12)  数据集2 （1, 20）/(1, 17)/(1, 25),
            nn.BatchNorm2d(out_channels[0]),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),   #数据集1 (1,1)  数据集2 (1,2)
        )
        self.inception1 = Inception(16,[8,16],(1,1),(1,3),(1,5),(1,1))

        self.convs1_1 = nn.Sequential(
            nn.Conv2d(in_channels=56, out_channels=16, kernel_size=(1, 8), padding='same'),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )
        self.convs2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=(10, 1),padding='same'),
            nn.BatchNorm2d(out_channels[2]),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        )
        # cnn_output: 50*6*64
        self.inception2 = Inception(32,[16, 32],(1,1),(3,1),(5,1),(1,1))

        self.convs3 = nn.Sequential(
            nn.Conv2d(in_channels=112, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )

        # self.lstm = nn.LSTM(input_size=input_size,
        #                     hidden_size=hidden_size,
        #                     batch_first=True,
        #                     num_layers=3)

        self.dropout = nn.Dropout(drop_prob)
        self.fc0 = nn.Linear(64*7*12, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 2)  #Dataset 1: 2  Dataset 2:4
    def forward(self,x):
        """
        x: (batch,h*w*channels)
        hbr: 30*24*1     数据集2：30*51*1
        """
        # x = torch.from_numpy(x)
        x = x.to(torch.float32)
        x = x.view(x.shape[0], 1, 30, x.shape[-1])
        x = self.convs1(x)    #数据集1 16*30*12   数据集2 16*30*12
        x = self.inception1(x)
        x = self.convs1_1(x)
        x = self.convs2(x)    #32*15*12
        # x = self.dropout(x)
        x = self.inception2(x)
        x = self.convs3(x)  #64*7*12  Dataset 2:64*7*25
        # x = self.dropout(x)
        # print(x.shape)
        x = x.view(x.shape[0],-1)
        # print(x.shape)
        features = F.sigmoid(self.fc0(x))
        #
        out = self.fc1(self.dropout(features))
        return out, features


if __name__ == "__main__":
    pass