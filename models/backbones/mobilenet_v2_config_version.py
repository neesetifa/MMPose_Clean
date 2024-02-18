import os
import torch
import torch.nn as nn
from collections import OrderedDict
import pdb

ACTIVATION = nn.ReLU6


# 单独定义一个相加模块, 方便量化函数打包
class Add(nn.Module):
    def __init__(self,):
        super(Add, self).__init__()

    def forward(self, a, b):
        return a+b

    
class ConvBNReLU(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size = 1, stride = 1, padding = 0, activation = nn.ReLU6):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels = input_channels,
                              out_channels = output_channels,
                              kernel_size = kernel_size,
                              stride = stride,
                              padding = padding,
                              bias = False)
        self.bn = nn.BatchNorm2d(output_channels)
        if activation is not None:
            self.activation = activation(inplace=True)
        else:
            self.activation = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
        

class LinearBottleneck(nn.Module):
    def __init__(self, input_channels, output_channels, expansion, stride = 1, activation = nn.ReLU6, expansion_channels = None):
        super(LinearBottleneck, self).__init__()
        self.stride = stride
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.expansion = expansion
        if expansion_channels is None:
            self.expansion_channels = input_channels * expansion
        else:
            self.expansion_channels = expansion_channels

        if self.expansion != 1:
            self.conv = nn.Conv2d(in_channels = input_channels,
                                  out_channels = self.expansion_channels,
                                  kernel_size = 1,
                                  stride = 1,
                                  bias = False)
            self.bn = nn.BatchNorm2d(self.expansion_channels)
            self.activation1 = activation(inplace=True)

        self.dw_conv = nn.Conv2d(in_channels = self.expansion_channels,
                                 out_channels = self.expansion_channels,
                                 kernel_size = 3,
                                 stride = stride,
                                 padding = 1,
                                 groups = self.expansion_channels,
                                 bias = False)
        self.dw_bn = nn.BatchNorm2d(self.expansion_channels)
        self.activation2 = activation(inplace=True)

        self.pw_conv = nn.Conv2d(in_channels = self.expansion_channels,
                                 out_channels = output_channels,
                                 kernel_size = 1,
                                 stride = 1,
                                 bias = False)
        self.pw_bn = nn.BatchNorm2d(output_channels)
        
        if self.stride == 1 and self.input_channels == self.output_channels:
            self.add = Add()
        

    def _forward_impl(self, x):
        if self.expansion != 1:
            x = self.conv(x)
            x = self.bn(x)
            x = self.activation1(x)

        x = self.dw_conv(x)
        x = self.dw_bn(x)
        x = self.activation2(x)
        
        x = self.pw_conv(x)
        x = self.pw_bn(x)

        return x

    def forward(self, inputs):
        x = inputs
        x = self._forward_impl(x)
        if self.stride == 1 and self.input_channels == self.output_channels:
            x = self.add(x, inputs)
        return x
        
class MobileNetV2(nn.Module):
    def __init__(self,
                 widen_factor,
                 residual_block = LinearBottleneck,
                 init_cfg = None,     # No need for export 
                 out_indices = None,  # compatible with mmpose config, not used
                 ):
        super(MobileNetV2, self).__init__()
        # Setting
        self.configs = configs[str(widen_factor)]
        self.residual_block = residual_block
        self.feature_channel = self.configs['feature_channel']
        self.init_cfg = init_cfg

        # 1. Input Block ====================
        self.input_channels = self.configs['input_channel']
        self.conv1 = ConvBNReLU(input_channels = 3,    # RGB image, input channel must be 3
                                output_channels = self.input_channels,
                                kernel_size = 3,
                                stride = 2,
                                padding = 1,
                                activation = ACTIVATION)
        self.block_0 = self.make_block(self.input_channels, self.configs['block_0'], expansion=1)

        # 2. Blocks
        self.block_1 = self.make_block(self.configs['block_0'][1][-1][-1], self.configs['block_1'])
        self.block_2 = self.make_block(self.configs['block_1'][1][-1][-1], self.configs['block_2'])
        self.block_3 = self.make_block(self.configs['block_2'][1][-1][-1], self.configs['block_3'])
        self.block_4 = self.make_block(self.configs['block_3'][1][-1][-1], self.configs['block_4'])
        self.block_5 = self.make_block(self.configs['block_4'][1][-1][-1], self.configs['block_5'])
        self.block_6 = self.make_block(self.configs['block_5'][1][-1][-1], self.configs['block_6'])

        # 3. Extractors
        self.conv2 = ConvBNReLU(input_channels = self.configs['block_6'][1][-1][-1],
                                output_channels = self.feature_channel,
                                activation = ACTIVATION)

        # # 4. Original MobileNet v2 head
        # self.dropout = nn.Dropout(0.2)
        # self.fc = nn.Linear(self.feature_channel, 1000) # [1280, 1000]
        
    def make_block(self, input_channels, configs, expansion=999, activation=ACTIVATION):
        modules = OrderedDict()
        stride, channel_configs = configs
        for i in range(len(channel_configs)):
            if i == 0:
                s = stride
            else:
                s = 1
            expansion_channels, output_channels = channel_configs[i]
            modules[str(i+1)] = LinearBottleneck(input_channels, output_channels, expansion, s, activation, expansion_channels)
            input_channels = output_channels

        return nn.Sequential(modules)

    def forward(self, inputs):
        # input = [3, 256, 192]
        # MobileNet v2 backbone
        x = self.conv1(inputs)     # [32, 128, 96]
        x = self.block_0(x)        # [16, 128, 96]  2x
        x = self.block_1(x)        # [24, 64, 48]  4x
        x = self.block_2(x)        # [32, 32, 24]  8x
        x = self.block_3(x)        # [64, 16, 12]  16x
        x = self.block_4(x)        # [96, 16, 12]   16x
        x = self.block_5(x)        # [160, 8, 6]    32x
        x = self.block_6(x)        # [320, 8, 6]    32x

        x = self.conv2(x)          # [1280, 8, 6]   32x  2223872 trainable params, 10 skip-connection

        # # Original Mobilenet v2 Head 
        # x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        # x = torch.flatten(x, 1)
        # x = self.dropout(x)
        # x = self.fc(x)             # 1281000 params, 2223872+128100 = 3504872 trainable params, match!

        return x

    def init_weights(self):
        if self.init_cfg is not None:
            if self.init_cfg['type'] == 'Pretrained':
                checkpoint = torch.load(self.init_cfg['checkpoint'], map_location='cpu')
                new_checkpoint = OrderedDict()
                for k,v in checkpoint.items():
                    if k.startswith('backbone.'):
                        k = k[9:]
                    new_checkpoint[k] = v
                self.load_state_dict(new_checkpoint)
                
                print(f'Backbone initialized with {self.init_cfg["checkpoint"]}')

configs = {'0.25': {"input_channel": 8,
                    "block_0": [1, [[8, 8]]],
                    "block_1": [2, [[48, 8], [48, 8]]],
                    "block_2": [2, [[48, 8], [48, 8], [48, 8]]],
                    "block_3": [2, [[48, 16], [96, 16], [96, 16], [96, 16]]],
                    "block_4": [1, [[96, 24], [144, 24], [144, 24]]],
                    "block_5": [2, [[144, 40], [240, 40], [240, 40]]],
                    "block_6": [1, [[240, 80]]],
                    "feature_channel": 1280},
           
           '0.5': {"input_channel": 16,
                   "block_0": [1, [[16, 8]]],
                   "block_1": [2, [[48, 16], [96, 16]]],
                   "block_2": [2, [[96, 16], [96, 16], [96, 16]]],
                   "block_3": [2, [[96, 32], [192, 32], [192, 32], [192, 32]]],
                   "block_4": [1, [[192, 48], [288, 48], [288, 48]]],
                   "block_5": [2, [[288, 80], [480, 80], [480, 80]]],
                   "block_6": [1, [[480, 160]]],
                   "feature_channel": 1280},
    
           '0.75': {'input_channel': 24,
                    'block_0': (1, [(24, 16)]),
                    'block_1': (2, [(96, 24), (144, 24)]),
                    'block_2': (2, [(144, 24), (144, 24), (144, 24)]),
                    'block_3': (2, [(144, 48), (288, 48), (288, 48), (288, 48)]),
                    'block_4': (1, [(288, 72), (432, 72), (432, 72)]),
                    'block_5': (2, [(432, 120), (720, 120), (720, 120)]),
                    'block_6': (1, [(720, 240)]),
                    'feature_channel': 1280},
    
           '1.0': {'input_channel': 32,
                   'block_0': (1, [(32, 16)]),
                   'block_1': (2, [(96, 24), (144, 24)]),
                   'block_2': (2, [(144, 32), (192, 32), (192, 32)]),
                   'block_3': (2, [(192, 64), (384, 64), (384, 64), (384, 64)]),
                   'block_4': (1, [(384, 96), (576, 96), (576, 96)]),
                   'block_5': (2, [(576, 160), (960, 160), (960, 160)]),
                   'block_6': (1, [(960, 320)]),
                   'feature_channel': 1280},
           }

    
if __name__ == "__main__":
    # To print model structure, run follow codes
    from torchsummaryX import summary
    
    model = MobileNetV2(widen_factor = 1.0)
    model.eval()
    summary(model, torch.zeros((1, 3, 256, 192))) # [C, H, W]
    pdb.set_trace()
