from model import *
import numpy as np
import torch
import matplotlib.pyplot as plt

def test_and_show(model):
    with torch.set_grad_enabled(False):  # 关闭梯度计算。这可以节省内存，因为在测试阶段不需要计算梯度
        print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader, device=DEVICE)))
    for batch_idx, (features, targets) in enumerate(test_loader):

        features = features
        targets = targets
        break

    nhwc_img = np.transpose(features[0], axes=(1, 2, 0))
    nhw_img = np.squeeze(nhwc_img.numpy(), axis=2)  # 从张量中挤压（squeeze）掉大小为1的维度。在这里，我们将挤压掉通道维度，得到一个形状为(H, W)的灰度图像
    plt.imshow(nhw_img, cmap='Greys')
    plt.show()

    model.eval()

    logits, probas = model(features.to(DEVICE)[0, None])  # 对第一个样本进行模型的前向传播，得到预测的logits和概率
    print('Probability 7 =  %.2f%%' % (probas[0][7] * 100))  # 获取类别7的概率值,[0] 表示取出第一个样本的预测结果,[7] 表示取出预测结果中的第 8 个元素（0在识别范围）


