import torch
import torch.nn as nn
import torch.nn.functional as F
from .ST_Attention import ChannelsAttentionModule
from .ST_Attention import TemporalAttentionModule
from .ST_Attention import ResBlock_CTAM
from .ST_Attention import CTAM

class CAM(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(CAM, self).__init__()

        self.channel_attention1 = ChannelsAttentionModule(4)  # 数据集1--4 数据集2--22
        self.channel_attention2 = ChannelsAttentionModule(12) # 数据集1--12 数据集2--25
        self.convs1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=(20, 1),padding='same'),
            # nn.Conv2d(in_channels=out_channels[2], out_channels=out_channels[2], kernel_size=(4, 1),padding='same'),
            nn.BatchNorm2d(out_channels[0]),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(5, 1), stride=(5, 1)),
            nn.Conv2d(in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=(4, 1),padding='same'),
            # nn.Conv2d(in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=(4, 1),padding='same'),
            nn.BatchNorm2d(out_channels[0]),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.convs2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=(1, 6),padding='same'), # 数据集1--6 数据集2--12
            # nn.Conv2d(in_channels=out_channels[2], out_channels=out_channels[2], kernel_size=(4, 1),padding='same'),
            nn.BatchNorm2d(out_channels[1]),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1))
        )
        # self.convs22 = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=(3, 2), padding=(1,2), stride=(2,1),output_padding=(1,0)),
        #     nn.BatchNorm2d(out_channels[1])
        # )  #data2
        self.convs22 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=(3, 1), padding=(1,4), stride=(2,1),output_padding=(1,0)),
            nn.BatchNorm2d(out_channels[1]),
            nn.ELU()
        )

    def forward(self, x1,x2):
        # 32*60*4
        x1 = self.channel_attention1(x1) # 数据集1 32*600*4  数据集2:32*600*22
        s_map1 = x1
        # print(x1.shape)
        x1 = self.convs1(x1)             # 数据集1 32*60*4  数据集2 32*60*22
       # print(x1.shape)
        x2 = self.channel_attention2(x2) # 数据集1 16*30*12  数据集2  16*30*25
        # print(x2.shape)
        s_map2 = x2
        x2 = self.convs2(x2)  #数据集2  16*30*25
       # print(x2.shape)
        pre_x2 = x2
        x2 = self.convs22(x2) #32*60*4  数据集2  32*60*22
      #  print(x2.shape)
        out = torch.cat((x1, x2), 1) # 64*60*4   数据集2  64*30*25
        return out,s_map1,s_map2, pre_x2,x2

# TAM(in_channels=[64,64], out_channels=[64,64])
class TAM(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(TAM, self).__init__()

        self.spatial_attention1 = TemporalAttentionModule(37)  #数据集1 37  数据集2 37
        self.spatial_attention2 = TemporalAttentionModule(7)   #数据集1 7  数据集2 7
        self.convs1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=(8, 1),padding=0),
            # nn.Conv2d(in_channels=out_channels[2], out_channels=out_channels[2], kernel_size=(4, 1),padding='same'),
            nn.BatchNorm2d(out_channels[0]),
            nn.ELU()
        )
        self.convs2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=(1, 6),padding='same'),
            # nn.Conv2d(in_channels=out_channels[2], out_channels=out_channels[2], kernel_size=(4, 1),padding='same'),
            nn.BatchNorm2d(out_channels[1]),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1))
        )
        self.convs22 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=(5, 1), padding=(0,4), stride=(4,1),output_padding=(1,0)),
            nn.BatchNorm2d(out_channels[1]),
            nn.ELU()
            #nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )

    def forward(self, x1,x2):
        x1 = self.spatial_attention1(x1)    #64*37*4
        t_map1 = x1
        x1 = self.convs1(x1)                #64*30*4  数据集2 64*30*22
        #print(x1.shape)
        x2 = self.spatial_attention2(x2)    #64*7*12  数据集2 64*7*25
        t_map2 = x2
        # print(x2.shape)
        x2 = self.convs2(x2)             # 数据集2 64*7*25
        pre_x2 = x2
        # print(x2.shape)
        x2 = self.convs22(x2)#64*30*4  数据集2 64*30*22
        #print(x2.shape)
        out = torch.cat((x1, x2), 1)        # 128*30*22
        return out,t_map1,t_map2,pre_x2,x2




class Inception_eeg(nn.Module):
    def __init__(self, in_channels):
        super(Inception_eeg, self).__init__()

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

#out_channels = [8,16] kernel_size1,kernel_size3,kernel_size5,kernel_size
#(1,1),(3,1)(5,1),(7,1),(1,1)
class Inception_hbr(nn.Module):
    def __init__(self, in_channel,out_channels,kernel_size1,kernel_size3,kernel_size5,kernel_size):
        super(Inception_hbr, self).__init__()

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

class MYFusion(nn.Module):
    def __init__(self, in_places=256, places=64, OutResTA=True, OutTAM=False, OutCAM=False, OutScale=False):
        super(MYFusion, self).__init__()

        self.OutResTA = OutResTA
        self.OutTAM  =OutTAM
        self.OutCAM = OutCAM
        self.OutScale = OutScale
        self.in_places = in_places
        self.places = places

        self.convs1_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 22),padding='same'),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 11), padding='same'),
            nn.BatchNorm2d(32),
            nn.ELU(),
            # nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )
        # 32*600*4

        self.inception1_1 = Inception_eeg(32)

        self.convs1_3 = nn.Sequential(
            nn.Conv2d(in_channels=88, out_channels=32, kernel_size=(4,1),padding='same'),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        )  #32*150*4

        self.convs1_4 = nn.Sequential(
            nn.Conv2d(in_channels=88, out_channels=64, kernel_size=(4,1),padding='same'),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        )  #64*37*4

        self.convs2_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 25), padding='same'),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1))
        )  #16*30*24
        self.inception2_1 = Inception_hbr(16,[8,16],(1,1),(1,3),(1,5),(1,1))

        self.convs2_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=56, out_channels=16, kernel_size=(1, 12), padding='same'),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )  #16*30*12

        self.convs2_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(10, 1), padding='same'),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )  # 32*15*12
        self.inception2_2 = Inception_hbr(32,[16, 32],(1,1),(3,1),(5,1),(1,1))

        self.convs2_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=112, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        ) #64*7*12

        self.channel_attention = CAM(in_channels=[32,16], out_channels=[32,32])
        self.temporal_attention = TAM(in_channels=[64,64], out_channels=[64,64])

        self.conv_center =nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(6, 1), padding='same'),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.res_ctam = ResBlock_CTAM(in_places=self.in_places, places=self.places)


#-------支流eeg------ 29人--
        self.fc1_0 = nn.Linear(64*37*4, 1024)
        self.fc1_1 = nn.Linear(1024, 4)

 #'''   支流hbr    '''
        self.fc2_0 = nn.Linear(64*7*12, 1024)
        self.fc2_1 = nn.Linear(1024, 4)
#''' fusion '''
        self.dropout = nn.Dropout(0.5)

        self.fc0 = nn.Linear(256*30*4, 2048)
        self.fc0_OutTAM = nn.Linear(256*30*4, 2048)

        # self.fc0_OutResTA = nn.Linear(256 * 30 * 22, 2048)
        # self.fc0_OutCAM = nn.Linear(128 * 30 * 22, 2048)
        # self.fc0_OutScale = nn.Linear(256 * 30 * 22, 2048)


        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 2)


    def forward(self,x1,x2,modality1, modality2):
        """
        x: (batch,h*w*channels)
        eeg: 600*8*1
        """
        OutResTA = self.OutResTA
        OutTAM = self.OutTAM
        OutCAM = self.OutCAM
        OutScale = self.OutScale

        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)

        x1 = x1.view(x1.shape[0], 1, 600, x1.shape[-1])
        x2 = x2.view(x2.shape[0], 1, 30, x2.shape[-1])

        x1 = self.convs1_1(x1) #   数据集1 32*600*4 数据集2 32*600*22
        x2 = self.convs2_1(x2)
        x2 = self.inception2_1(x2)
        x2 = self.convs2_1_1(x2)  # 数据集1 16*30*12  数据集2 16*30*25
        # print(x1.shape,x2.shape)
        ys_x1 = x1
        ys_x2 = x2

        if modality1:
            print('Close branch1')
            x1 = torch.zeros_like(x1)
        if modality2:
            print('Close branch2')
            x2 = torch.zeros_like(x2)
        x_c,s_1,s_2,pre_ts,ts = self.channel_attention(x1, x2)  # 64*60*4   数据集2 64*60*22

        x1 = self.inception1_1(x1)
        x1 = self.convs1_3(x1)
        x1 = self.inception1_1(x1)
        x1 = self.convs1_4(x1)  #   64*37*4  数据集2 64*37*22

        x2 = self.convs2_2(x2) #   32*15*12
        x2 = self.inception2_2(x2)
        x2 = self.convs2_2_1(x2)  #64*7*12    数据集2 64*7*25
        # print(x1.shape,x2.shape)
        yt_x1 = x1
        yt_x2 = x2

        x_t,t_1,t_2,pre_tt,tt = self.temporal_attention(x1, x2)  # 128*30*4   数据集2 128*30*22
        # fusion:
        x_c = self.conv_center(x_c)          #  128*30*4  数据集2 128*30*22
        # print(x_c.shape)
        out_c =  torch.cat((x_t, x_c), 1)    #  256*30*4 数据集2 256*30*22
        # print(out_c.shape) #  10 29人数据-256*30*4  6
        fusion_feats = out_c.view(out_c.shape[0],-1)
        x = F.sigmoid(self.fc0(fusion_feats))
        # 消融实验
        if OutResTA:
            out_c = self.res_ctam(out_c)     # out_res-stam   数据集2 256*30*22
            # print(out_c.shape)
            # fusion_feats = out_c.view(out_c.shape[0], -1)
            # x = F.sigmoid(self.fc0_OutResTA(fusion_feats))
            fusion_feats = out_c.view(out_c.shape[0], -1)
            x = F.sigmoid(self.fc0(fusion_feats))
        if OutScale:
            x_t  = x_t.view(x_t.shape[0],-1)
            x_c = x_c.view(x_c.shape[0], -1)
            out_c = torch.concatenate((x_t, x_c),-1)
            # print(out_c.shape)
            # fusion_feats = out_c.view(out_c.shape[0], -1)
            # x = F.sigmoid(self.fc0_OutScale(fusion_feats))
            fusion_feats = out_c.view(out_c.shape[0], -1)
            x = F.sigmoid(self.fc0(fusion_feats))
        if OutTAM:
            out_c = self.res_ctam(x_c)
            print(out_c.shape)
            fusion_feats = out_c.view(out_c.shape[0], -1)
            x = F.sigmoid(self.fc0_OutTAM(fusion_feats))
        if OutCAM:
            out_c = self.res_ctam(x_t)
            # print(out_c.shape)
            # fusion_feats = out_c.view(out_c.shape[0], -1)
            # x = F.sigmoid(self.fc0_OutCAM(fusion_feats))
            fusion_feats = out_c.view(out_c.shape[0], -1)
            x = F.sigmoid(self.fc0_OutTAM(fusion_feats))

        x = self.fc1(self.dropout(x))
        out_fusion = self.fc2(x)
        # out_fusion =F.softmax(features)

        # print(x1.shape)
        # print(x2.shape)
        # eeg: 64*37*2
        modality1_eeg = x1
        modality1_hbr = x2
        x1 = x1.view(x1.shape[0],-1)
        x1 = F.sigmoid(self.fc1_0(x1))
        out1 = self.fc1_1(self.dropout(x1))
        # out1 = F.softmax(features1)
        # hbr: 32*15*6
        x2 = x2.view(x2.shape[0],-1)
        x2 = F.sigmoid( self.fc2_0(x2))
        out2 = self.fc2_1(self.dropout(x2))
        # out2 = F.softmax(features2)

        return out1,out2, out_fusion,ys_x1,ys_x2,s_1,s_2,pre_ts,ts,yt_x1,yt_x2,t_1,t_2,pre_tt,tt #,modality1_eeg,modality1_hbr




if __name__ == "__main__":
    pass
