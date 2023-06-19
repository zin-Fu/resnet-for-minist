import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels, Mystride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=Mystride,padding=1,bias=False)


def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0  # 记录正确预测的样本数和总样本数
    for i, (features, targets) in enumerate(data_loader):  # 使用enumerate遍历data_loader，以获取索引(i)和数据批次(features，targets)
        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)  # 对输入特征进行模型的前向传播，得到预测的logits和概率
        _, predicted_labels = torch.max(probas, 1)  # 在probas张量的第二个维度(dim=1)上找到最大概率的索引，表示预测的类标签。最大概率的实际值被舍弃（用_表示）
        num_examples += targets.size(0)  # 将当前批次的大小（即批次中的样本数）添加到num_examples变量中，增加总样本数
        correct_pred += (predicted_labels == targets).sum()  # 过将预测的标签与目标标签进行比较，并计算匹配的数量，统计正确预测的样本数
    return correct_pred.float() / num_examples * 100  # 通过将正确预测的样本数除以总样本数，计算准确率，并乘以100转换为百分比

