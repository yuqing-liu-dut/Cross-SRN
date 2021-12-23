from model import common

import torch
import torch.nn as nn

def make_model(args, parent=False):
    return Net(args)

class SEBlock(nn.Module):
    def __init__(self, input_dim=64):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // 16),
            nn.LeakyReLU(inplace=True),
            nn.Linear(input_dim // 16, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class Upsample(nn.Module):
    def __init__(self, n_channels, scale=4):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(n_channels, 3 * scale * scale, 3, padding=3 // 2)
        self.up = nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        return x

class CrossConv(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super(CrossConv, self).__init__()
        self.conv3_1_A = nn.Conv2d(in_channels, out_channels, (3, 1), padding=(1, 0))
        self.conv3_1_B = nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1))
        self.relu_1 = nn.LeakyReLU(inplace=True)
        self.conv3_2_A = nn.Conv2d(out_channels, out_channels, (3, 1), padding=(1, 0))
        self.conv3_2_B = nn.Conv2d(out_channels, out_channels, (1, 3), padding=(0, 1))
        self.adafm = nn.Conv2d(out_channels, out_channels, 3, padding=3//2, groups=out_channels)
        self.senet = SEBlock(out_channels)

    def forward(self, input):
        x = self.conv3_1_A(input) + self.conv3_1_B(input)
        x = self.relu_1(x)
        x = self.conv3_2_A(x) + self.conv3_2_B(x)
        x = self.adafm(x)+x
        x = self.senet(x)
        return input + x 

class ChannelSplitConv(nn.Module):
    def __init__(self, n_channels=64):
        super(ChannelSplitConv, self).__init__()
        self.conv_1 = CrossConv(n_channels//4*3, n_channels//4*3)
        self.conv_2 = CrossConv(n_channels//4*2, n_channels//4*2)
        self.conv_3 = CrossConv(n_channels//4*1, n_channels//4*1)
        self.conv_out = nn.Conv2d(n_channels, n_channels, 3, padding=3//2)
        self.conv_in = nn.Conv2d(n_channels, n_channels, 3, padding=3//2)
        self.senet = SEBlock(n_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        bypath = x
        c = x.size()[1] // 4
        l1 = self.conv_1(x[:,c:,:,:])
        l2 = self.conv_2(l1[:,c:,:,:])
        l3 = self.conv_3(l2[:,c:,:,:])
        x = torch.cat((x[:,:c,:,:], l1[:,:c,:,:], l2[:,:c,:,:], l3[:,:c,:,:]), 1)
        x = self.senet(x)
        x = self.conv_in(x)
        x = self.relu(x)
        x = self.conv_out(x)
        return x + bypath

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        scale = args.scale[0]
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        n_conv = 10
        n_step = 1 
        self.n_conv = n_conv
        self.n_step = n_step

        module_head = [nn.Conv2d(3, 64, 3, padding=3//2)]
        module_body = [ChannelSplitConv(64) for _ in range(n_conv)]
        module_tail = [nn.Conv2d(64, 64, 3, padding=3//2), nn.LeakyReLU(inplace=True), nn.Conv2d(64, 64, 3, padding=3//2)]
        self.head = nn.Sequential(*module_head)
        self.body = nn.Sequential(*module_body)
        self.tail = nn.Sequential(*module_tail)
        self.upscale = Upsample(64, scale=scale)

    def forward(self, x):
        x = self.sub_mean(x)
        head = self.head(x)
        body = self.body(head)
        tail = self.tail(body)+head
        x = self.upscale(tail)
        x = self.add_mean(x)
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
