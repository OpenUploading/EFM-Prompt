import torch
import torch.nn as nn

class ChannelsAttentionModule(nn.Module):
    def __init__(self, channel, ratio=2):
        super(ChannelsAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  #输出固定尺寸
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, c, t, mc = x.size()
        new_x = x.view(_, c, t, mc).permute(0, 3, 2, 1).contiguous()
        # print(new_x.shape)
        avgout = self.shared_MLP(self.avg_pool(new_x))
        # print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(new_x))

        out = self.sigmoid(avgout + maxout) * new_x
        out = out.permute(0, 3, 2, 1).contiguous()
        # print(out.shape)
        return out



class TemporalAttentionModule(nn.Module):
    def __init__(self, channel, ratio=4):
        super(TemporalAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, c, t, mc = x.size()
        new_x = x.view(_, c, t, mc).permute(0, 2, 1, 3).contiguous()
        avgout = self.shared_MLP(self.avg_pool(new_x))
        # print('avgout',avgout.shape)
        maxout = self.shared_MLP(self.max_pool(new_x))
        out = self.sigmoid(avgout + maxout) * new_x
        out = out.permute(0, 2, 1, 3).contiguous()
        return out


class CTAM(nn.Module):
    def __init__(self, channel1,channel2):
        super(CTAM, self).__init__()
        self.channel_attention = ChannelsAttentionModule(channel1)
        self.spatial_attention = TemporalAttentionModule(channel2)

    def forward(self, x):
        out = self.channel_attention(x)
     #   print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out)
        return out

#  ResBlock_CTAM(256,64)
class ResBlock_CTAM(nn.Module):
    def __init__(self,in_places, places, stride=(1,1),downsampling=False, expansion = 4):  #
        super(ResBlock_CTAM,self).__init__()
        self.expansion = in_places//places
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=(1,1),stride=(1,1), bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=(3,1), stride=stride, padding='same', bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=(1,1), stride=(1,1), bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )
        self.ctam = CTAM(channel1=4, channel2=30)  # mydata (channel1=6、4, channel2=30)

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=(1,1), stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        # print(out.shape)
        out = self.ctam(out)
        # print(out.shape)
        if self.downsampling:
            residual = self.downsample(x)
        # print(residual.shape)

        out += residual  #out_res
        out = self.relu(out)
        return out



