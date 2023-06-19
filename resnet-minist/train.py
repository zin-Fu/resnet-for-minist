from model import *
import time
import torch
import torch.nn.functional as F

def trainModel(model, optimizer, scheduler):

    start_time = time.time()
    for epoch in range(NUM_EPOCHS):

        model.train()  #  将模型设置为训练模式，启用Batch Normalization和Dropout层
        for batch_idx, (features, targets) in enumerate(train_loader):  # 迭代训练数据集，使用enumerate(train_loader)函数返回一个可迭代的对象，
                                                                        # 其中每次迭代返回一个包含两个元素的元组(batch_idx, (features, targets))
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)

            logits, probas = model(features)  # 将输入特征features作为模型的输入，调用模型对象model进行前向传播计算
            cost = F.cross_entropy(logits, targets)
            # 清零优化器的梯度缓存
            optimizer.zero_grad()
            # 反向传播并计算梯度
            cost.backward()
            # 更新权重
            optimizer.step()

            if not batch_idx % 50:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' % (epoch+1, NUM_EPOCHS, batch_idx, len(train_loader), cost))

        model.eval()  # 将模型设置为评估模式，禁用Batch Normalization和Dropout层
        with torch.set_grad_enabled(False):  # 关闭梯度计算，节省内存
            print('Epoch: %03d/%03d | Train: %.3f%%' % (
                epoch + 1, NUM_EPOCHS,
                compute_accuracy(model, train_loader, device=DEVICE)))
        scheduler.step()  # 学习率调度器更新
        print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))
