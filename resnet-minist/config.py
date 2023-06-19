import torch

# Hyperparameters
RANDOM_SEED = 1  # 设置随机数生成器的种子
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 100

# Architecture
NUM_FEATURES = 28*28  # 输入数据的特征数量,MNIST数据集中的图像大小为28x28，所以特征数量为28x28=784
NUM_CLASSES = 10

# Other
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRAYSCALE = True  # 输入数据是否为灰度图像

