import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

ACTIVATION = nn.ReLU


# 单独定义一个concat模块和upsampling模块, 方便量化函数打包
class Concatenate(nn.Module):   
    def __init__(self, dim):
        super(Concatenate, self).__init__()
        self.dim = dim

    def forward(self, *x):  # concate 必须这么定义, 一定要这么送入(x1,x2,x3), 不然和qat里的模块定义会发生冲突
        return torch.cat(x, dim = self.dim)


# 以下两个实现等价
# def channel_shuffle(x, groups):
#     batchsize, num_channels, height, width = x.size()
#     channels_per_group = num_channels // groups

#     # reshape
#     x = x.view(batchsize, groups, channels_per_group, height, width)
#     x = torch.transpose(x, 1, 2).contiguous() # [bs,group,group_channel,h,w]->[bs,group_channel,group,h,w]
#     # flatten
#     x = x.view(batchsize, num_channels, height, width)

#     return x

# class ChannelShuffle(Module):
#     def __init__(self, groups: int) -> None:
#         super(ChannelShuffle, self).__init__()
#         self.groups = groups

#     def forward(self, input: Tensor) -> Tensor:
#         return F.channel_shuffle(input, self.groups)


class ConvBNReLU(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size = 1, stride = 1, padding = 0, groups = 1, activation = ACTIVATION):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels = input_channels,
                              out_channels = output_channels,
                              kernel_size = kernel_size,
                              stride = stride,
                              padding = padding,
                              groups = groups,
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
    def __init__(self, input_channels, output_channels, stride=1, reduction=False, activation=ACTIVATION):
        super(LinearBottleneck, self).__init__()
        self.stride = stride
        self.reduction = reduction
        self.input_channels = input_channels
        self.output_channels = output_channels // 2
        self.reduction_channels = self.output_channels if reduction else input_channels
        
        if self.reduction:
            self.conv = nn.Conv2d(in_channels = self.input_channels if stride > 1 else self.reduction_channels,
                                  out_channels = self.reduction_channels,
                                  kernel_size = 1,
                                  stride = 1,
                                  bias = False)
            self.bn = nn.BatchNorm2d(self.reduction_channels)
            self.activation1 = activation(inplace=True)

        self.dw_conv = nn.Conv2d(in_channels = self.reduction_channels,
                                 out_channels = self.reduction_channels,
                                 kernel_size = 3,
                                 stride = stride,
                                 padding = 1, 
                                 groups = self.reduction_channels,
                                 bias = False)
        self.dw_bn = nn.BatchNorm2d(self.reduction_channels)

        self.pw_conv = nn.Conv2d(in_channels = self.reduction_channels,
                                 out_channels = self.output_channels,
                                 kernel_size = 1,
                                 stride = 1,
                                 bias = False)
        self.pw_bn = nn.BatchNorm2d(self.output_channels)
        self.activation2 = activation(inplace=True)
        
    def forward(self, inputs):
        x = inputs
        
        if self.reduction:
            x = self.conv(x)
            x = self.bn(x)
            x = self.activation1(x)

        x = self.dw_conv(x)
        x = self.dw_bn(x)

        x = self.pw_conv(x)
        x = self.pw_bn(x)
        x = self.activation2(x)

        return x
    
    
class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        
        if stride == 2:
            self.branch_proj = LinearBottleneck(inp, oup, stride, False, ACTIVATION)
        else:
            self.branch_proj = nn.Sequential() # Empty

        self.branch_main = LinearBottleneck(inp, oup, stride, True, ACTIVATION)
        self.concat = Concatenate(dim = 1)
        self.channel_shuffle = nn.ChannelShuffle(groups=2)
            
    def forward(self, inputs):
        if self.stride==1:
            #x_proj, x = inputs.chunk(2, dim=1)
            batch, channel, h, w = inputs.shape
            x_proj, x = inputs[:,:channel//2], inputs[:,channel//2:]
            x = self.branch_main(x)
            out = self.concat(x_proj, x)
        
        elif self.stride==2:
            x_proj = self.branch_proj(inputs)
            x = self.branch_main(inputs)
            out = self.concat(x_proj, x)

        out = self.channel_shuffle(out)
        return out

    
class ShuffleNetV2(nn.Module):
    def __init__(self, width, repeat = [4, 8, 4], num_classes = 1000):
        super(ShuffleNetV2, self).__init__()

        self.stage_repeats = repeat
        self.width = str(width) if isinstance(width, float) else width
        stage_out_channels = {'0.5': [24, 48, 96, 192, 1024],
                              '1.0': [24, 116, 232, 464, 1024],
                              '1.5': [24, 176, 352, 704, 1024],
                              '2.0': [24, 244, 488, 976, 2048],
                              }
        self.stage_out_channels = stage_out_channels[self.width]

        # 1.first layer
        input_channel = self.stage_out_channels[0]
        self.first_conv = ConvBNReLU(input_channels = 3, # RGB 3, GRAY 1
                                     output_channels = input_channel,
                                     kernel_size = 3,
                                     stride = 2,
                                     padding = 1,
                                     activation = ACTIVATION)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 2.feature layer
        stage_names = ["stage2", "stage3", "stage4"]
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+1]
            stageSeq = []
            for i in range(numrepeat):
                if i == 0:
                    stageSeq.append(ShuffleV2Block(input_channel, output_channel, stride=2))
                else:
                    stageSeq.append(ShuffleV2Block(input_channel, output_channel, stride=1))
                    
                input_channel = output_channel
            setattr(self, stage_names[idxstage], nn.Sequential(*stageSeq))

        # Backbone ends here ======================================================================

        # 3.classification head
        self.conv_last = ConvBNReLU(input_channel, self.stage_out_channels[-1])
        # self.global_pooling = nn.AdaptiveAvgPool2d((1,1))
        # self.classifier = nn.Linear(self.stage_out_channels[-1], num_classes)
        

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        x = self.stage2(x)  # C1, [48, 40, 40]
        x = self.stage3(x)  # C2, [96, 20, 20]
        x = self.stage4(x)  # C3, [192,10, 10]
        # # Backbone ends here =========

        x = self.conv_last(x)
        # x = self.global_pooling(x).reshape(-1, self.stage_out_channels[-1])
        # x = self.classifier(x)

        return x
    


if __name__ == "__main__":
    from torchsummaryX import summary
    
    model = ShuffleNetV2(1.5)
    model.eval()
    summary(model, torch.zeros((1, 3, 128, 128)))   # [C, H, W]
    for i, (name, module) in enumerate(model.named_modules()):
        print(i, name, type(module).__name__)
    pdb.set_trace()
