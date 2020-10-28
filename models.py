import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet, densenet

import numpy as np

from collections import OrderedDict


class VGGNet(nn.Module):
    def __init__(self, im_shape, num_classes, max_pooling, num_stages, cnn_num_filters, conv_padding, norm_layer):
        super().__init__()

        self.cnn_filters = cnn_num_filters
        self.input_shape = list(im_shape)
        self.num_stages = num_stages
        self.num_classes = num_classes

        if max_pooling:
            self.conv_stride = 1
        else:
            print("Using strided convolutions")
            self.conv_stride = 2

        print("meta network params")

        layers = []
        in_channels = self.input_shape[1]
        for i in range(self.num_stages):
            layers.append(
                nn.Conv2d(
                    in_channels, self.cnn_filters, kernel_size=3,
                    padding=conv_padding, stride=self.conv_stride,
                )
            )
            in_channels = self.cnn_filters
            if norm_layer == 'batch_norm':
                layers.append(nn.BatchNorm2d(self.cnn_filters))
            else:
                assert False
            layers.append(nn.LeakyReLU(inplace=True))
            if max_pooling:
                layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0))

        test_in = torch.zeros(im_shape)
        out_shape = nn.Sequential(*layers)(test_in).shape
        layers += [nn.Flatten(), nn.Linear(np.prod(out_shape[1:]),  self.num_classes)]
        self.net = nn.Sequential(*layers)

        for name, param in self.named_parameters():
            print(name, param.shape)

    def forward(self, x):
        return self.net(x)


# Source: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L118
# Modified to be more like https://github.com/ElementAI/TADAM/blob/master/model/tadam.py#L333
class ResNet(nn.Module):
    def __init__(self, im_shape, layers=[1, 1, 1, 1], block=resnet.BasicBlock,
                 num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        self.input_shape = list(im_shape)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.inplanes = self.input_shape[1]
        self.layer1 = self._make_layer(block, 32, layers[0], stride=2)
        self.layer2 = self._make_layer(
            block, 64, layers[1], stride=2,
            dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(
            block, 128, layers[2], stride=2,
            dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(
            block, 256, layers[3], stride=2,
            dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                resnet.conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(
        self, im_shape, growth_rate=8, block_config=(3, 3, 3, 3),
        bn_size=2, drop_rate=0, num_classes=1000,
        memory_efficient=False
    ):
        super(DenseNet, self).__init__()

        self.input_shape = im_shape

        # First convolution
        self.features = nn.Sequential()

        # Each denseblock
        num_features = self.input_shape[1]
        for i, num_layers in enumerate(block_config):
            block = densenet._DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = densenet._Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
