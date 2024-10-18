import torch.nn as nn
import torch
import numpy as np
from .NAM import cbam_block


class SiLU(nn.Module):
    # SiLU激活函数
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def autopad(k, p=None, d=1):
    # kernel, padding, dilation
    # 对输入的特征层进行自动padding，按照Same原则
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    # 标准卷积+标准化+激活函数
    default_act = SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act    = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, deploy=False):
        super(RepVGGBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.deploy = deploy
        self.groups = groups

        # training_layer
        if self.in_channels == self.out_channels and self.stride == 1:
            self.bn_elective = nn.BatchNorm2d(self.in_channels, affine=True)

        self.conv1 = nn.Sequential()
        self.conv1.add_module("conv", nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                                kernel_size=1, padding=1 - 3 // 2, stride=self.stride))
        self.conv1.add_module("bn", nn.BatchNorm2d(self.out_channels, affine=True))
        # self.conv1.add_module("attention", cbam_block(self.out_channels))

        self.conv2 = nn.Sequential()
        self.conv2.add_module("conv", nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                                kernel_size=3, padding=1, stride=self.stride))
        self.conv2.add_module("bn", nn.BatchNorm2d(self.out_channels, affine=True))
        # self.conv2.add_module("attention", cbam_block(self.out_channels))

        # creation of the inference_layer
        self.layer = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3,
                               stride=self.stride, padding=1)

        self.act = nn.LeakyReLU()

    def forward(self, x):
        if self.deploy:
            output = self.layer(x)
            output = self.act(output)
            return output
        else:
            if hasattr(self, "bn_elective"):
                output1 = self.bn_elective(x)
            else:
                output1 = 0
            output2 = self.conv1(x)
            output3 = self.conv2(x)
            output = self.act(output1 + output2 + output3)
            return output

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def transform(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv2)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv1)
        kernelid, biasid = 0, 0
        if hasattr(self, "bn_elective"):
            kernelid, biasid = self._fuse_bn_tensor(self.bn_elective)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def to_deploy(self):
        if hasattr(self, 'layer') and self.deploy:
            return
        kernel, bias = self.transform()
        self.layer = nn.Conv2d(in_channels=self.conv2.conv.in_channels, out_channels=self.conv2.conv.out_channels,
                               kernel_size=self.conv2.conv.kernel_size, stride=self.conv2.conv.stride,
                               padding=self.conv2.conv.padding, dilation=self.conv2.conv.dilation,
                               groups=self.conv2.conv.groups, bias=True)
        self.layer.weight.data = kernel
        self.layer.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv2')
        self.__delattr__('conv1')
        if hasattr(self, 'bn_elective'):
            self.__delattr__('bn_elective')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

    def to_train(self):
        self.deploy = False


class RepBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, deploy=False):
        super(RepBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deploy = deploy
        self.n = n
        self.block1 = RepVGGBlock(self.in_channels, self.out_channels, stride=1, deploy=self.deploy)
        self.block_sequence = nn.Sequential(
            *(RepVGGBlock(self.out_channels, self.out_channels, stride=1, deploy=self.deploy) for _ in range(n - 1))
        ) if n > 1 else None

    def forward(self, x):
        output = self.block1(x)
        if self.n > 1:
            output = self.block_sequence(output)
        return output

    def to_deploy(self):
        self.block1.to_deploy()
        for block in self.block_sequence:
            block.to_deploy()

    def to_train(self):
        self.block1.to_train()
        for block in self.block_sequence:
            block.to_train()


class ERBlock(nn.Module):
    def __init__(self, in_channels, out_channels, number, deploy=False):
        super(ERBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deploy = deploy
        self.block1 = RepVGGBlock(self.in_channels, self.out_channels, stride=2, deploy=self.deploy)
        self.block_sequence = RepBlock(self.out_channels, self.out_channels, n=number, deploy=self.deploy)

    def forward(self, x):
        output = self.block1(x)
        output = self.block_sequence(output)
        return output

    def to_deploy(self):
        self.block1.to_deploy()
        self.block_sequence.to_deploy()

    def to_train(self):
        self.block1.to_train()
        self.block_sequence.to_train()


class SimSPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(SimSPPF, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels // 2, self.kernel_size, stride=1,
                      padding=self.kernel_size // 2,
                      bias=False),
            nn.BatchNorm2d(self.in_channels // 2, affine=True),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d((self.in_channels // 2) * 4, self.out_channels, self.kernel_size, stride=1,
                      padding=self.kernel_size // 2,
                      bias=False),
            nn.BatchNorm2d(self.out_channels, affine=True),
            nn.ReLU()
        )
        self.max_pooling = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)

    def forward(self, x):
        x = self.conv1(x)
        output1 = self.max_pooling(x)
        output2 = self.max_pooling(output1)
        output3 = self.max_pooling(output2)
        final = self.conv2(torch.cat((x, output1, output2, output3), dim=1))
        return final


class EfficientRep(nn.Module):
    def __init__(self, config):
        super(EfficientRep, self).__init__()
        self.RepVGGBlock = RepVGGBlock(3, 32, stride=2)
        self.cbam1 = cbam_block(32)
        self.ERBlock_2 = ERBlock(32, 64, config[0])
        self.cbam2 = cbam_block(64)
        self.ERBlock_3 = ERBlock(64, 128, config[1])
        self.cbam3 = cbam_block(128)
        self.ERBlock_4 = ERBlock(128, 256, config[2])
        self.cbam4 = cbam_block(256)
        self.ERBlock_5 = ERBlock(256, 512, config[3])
        self.SimSPPF = SimSPPF(512, 512)

    def forward(self, x):
        x = self.RepVGGBlock(x)

        x = self.cbam1(x)
        x = self.ERBlock_2(x)

        x = self.cbam2(x)
        feat3 = self.ERBlock_3(x)

        x = self.cbam3(feat3)
        feat4 = self.ERBlock_4(x)

        x = self.cbam4(feat4)
        x = self.ERBlock_5(x)
        feat5 = self.SimSPPF(x)
        return feat3, feat4, feat5

    def to_deploy(self):
        self.RepVGGBlock.to_deploy()
        self.ERBlock_2.to_deploy()
        self.ERBlock_3.to_deploy()
        self.ERBlock_4.to_deploy()
        self.ERBlock_5.to_deploy()

    def to_train(self):
        self.RepVGGBlock.to_train()
        self.ERBlock_2.to_train()
        self.ERBlock_3.to_train()
        self.ERBlock_4.to_train()
        self.ERBlock_5.to_train()


def backbone(pretrained=False):
    model = EfficientRep([1, 1, 1, 1])
    if pretrained:
        model.load_state_dict(torch.load("./work/__Prototype__Indoor/model_data/best_epoch_weights.pth"))
    return model
