from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from config import *   # 引用config里全部参数

# 定义数据增强的转换操作
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),  # 对图像进行随机裁剪，将其大小调整为指定的尺寸，随机裁剪可以提取图像的不同部分，增加数据的多样性。
    transforms.RandomHorizontalFlip(),  # 以一定的概率对图像进行随机水平翻转。通过翻转图像，可以增加数据的多样性，使模型更具鲁棒性
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # 对图像进行标准化处理，使其像素值在均值为0，标准差为1的范围内。标准化可以帮助模型更好地学习数据的分布，加速训练过程
])

train_dataset = datasets.MNIST(root='data',  # 指定数据集的保存路径
                               train=True,  # 加载训练集
                               transform=train_transform,  # 应用数据增强
                               download=True)  # 如果数据集不存在，则自动从网上下载数据集

test_dataset = datasets.MNIST(root='data',
                              train=False,  # 加载测试集
                              transform=transforms.ToTensor())  # 将数据转换为Tensor类型

# 数据加载器
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)  # 每个epoch开始时，对数据进行洗牌，以增加训练的随机性

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False)

#检查数据集
'''
for images, labels in train_loader:
    print('Image batch dimensions: ', images.shape)
    print('Image label dimensions: ', labels.shape)
    break
'''

