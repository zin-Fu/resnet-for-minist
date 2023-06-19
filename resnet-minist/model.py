from utils import *
from data_loader import *
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1  # 基本块的扩展系数。在ResNet中，基本块的输出通道数是输入通道数的expansion倍

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):  # downsample下采样层
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)  # 3x3卷积层，它将输入特征图转换为具有out_channels通道数的特征图。步幅由stride参数指定
        self.bn1 = nn.BatchNorm2d(out_channels)  # 2D批归一化层，用于归一化卷积层的输出
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  # 可选的下采样层，用于匹配输入和输出的维度，以便能够进行残差连接
        self.stride = stride

    def forward(self, x):
        residual = x  # 将输入x保存为residual，以便后续将其添加到卷积块的输出上，形成残差连接

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)

        if self.downsample is not None:  # 检查是否存在下采样操作
            residual = self.downsample(x)  # 果存在下采样操作，对输入x进行下采样，得到residual作为残差

        y += residual  # 将残差residual与经过卷积和批归一化后的特征图y相加，实现残差连接
        y = self.relu(y)

        return y


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.in_channels = 64  # 初始输入通道数为64
        if grayscale:  # 如果grayscale为True，则将输入通道数in_dim设置为1，表示灰度图像；
            in_dim = 1
        else:  # 否则，设置为3，表示RGB图像
            in_dim = 3
        super(ResNet, self).__init__()  # 调用父类nn.Module的初始化方法
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)  # inplace=True表示在原地执行操作，节省内存
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])  # layers->残差快数量
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # 定义全连接层fc，输入特征数量为512 * block.expansion，输出特征数量为num_classes，用于分类任务

        for m in self.modules():  # 遍历ResNet模型的所有模块
            if isinstance(m, nn.Conv2d):  # 如果当前模块是nn.Conv2d类型的
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # 计算卷积核的参数数量
                m.weight.data.normal_(0, (2. / n) ** .5)  # 使用计算得到的卷积核参数数量（n）来对卷积核权重进行初始化，采用的是均值为0，标准差为(2. / n) ** .5的正态分布
            elif isinstance(m, nn.BatchNorm2d):  # 如果当前模块是nn.BatchNorm2d类型的
                m.weight.data.fill_(1)  # 将批归一化层的权重初始化为1
                m.bias.data.zero_()  # 将批归一化层的偏置项初始化为0

    def _make_layer(self, block, out_channels, blocks, stride=1):  # 定义构建残差层的过程。它接受块类型block、输出通道数out_channels、块的数量blocks和步幅stride作为参数
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:  # 如果步幅不为1或输入通道数与输出通道数不匹配
            downsample = nn.Sequential(                                        # 创建下采样层downsample，由一个1x1卷积层和一个批归一化层组成(为了解决在残差连接中输入和输出尺寸不匹配的问题)
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []  # 初始化层列表
        layers.append(block(self.in_channels, out_channels, stride, downsample))  # 将第一个残差块添加到层列表中，并更新输入通道数
        self.in_channels = out_channels * block.expansion  # 更新输入通道数为当前输出通道数乘以块的扩展系数
        '''
           为什么第一颗残差快单独考虑：

           第一个残差块的输入是经过初始卷积操作得到的特征图，输入尺寸与后续的残差块的输入尺寸不同。

           第一个残差块之后的残差块（layer1，layer2，layer3，layer4）的输入通道数与输出通道数是一致的，因为在每个残差块内部已经通过卷积操作将输入通道数进行了调整。

           而第一个残差块的输入通道数则由初始卷积操作的输出通道数决定，通常为64。

           因此，为了适应这个特殊情况，需要单独将第一个残差块添加到层列表中，并更新self.inchannels变量，使其与第一个残差块的输出通道数保持一致。

           这样在后续的残差块中，self.inchannels的值就会与输出通道数自动匹配，确保网络的连续性和正确性。
        '''
        for i in range(1, blocks):  # 迭代构建剩余的残差块
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)  # 将层列表转换为顺序容器nn.Sequential并返回

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
    # 以下通过残差层向前传播
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # because MNIST is already 1x1 here:
        # disable avg pooling
        # x = self.avgpool(x)

        x = x.view(x.size(0), -1)  # 将特征图展平为向量
        logits = self.fc(x)  # 通过全连接层计算预测的logits
        probas = F.softmax(logits, dim=1)  # 对logits进行softmax操作得到预测概率
        return logits, probas


def resnet18(num_classes):
    model = ResNet(block=BasicBlock,
                   layers=[2, 2, 2, 2],  # 有四个残差层，每个残差层都由两个块组成
                   num_classes=NUM_CLASSES,
                   grayscale=GRAYSCALE)
    return model
