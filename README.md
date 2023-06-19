# resnet-for-minist

###Introduction：这个项目实现了用resnet18进行minist的手写数字分类
### 项目总架构：

`main.py`：这是项目的主要入口文件，用于执行整个训练和测试流程

`model.py`：这个文件包含了ResNet模型的定义。在这里定义ResNet的网络结构、残差块等

`data_loader.py`：这个文件负责数据集的加载和预处理。可以在这里编写代码来加载MNIST数据集，对图像进行预处理并准备训练和测试数据

`train.py`：这个文件包含训练过程的代码。可以在这里编写训练循环，包括前向传播、计算损失、反向传播和参数更新等

`test.py`：这个文件包含测试过程的代码。可以在这里编写测试循环，用于评估训练好的模型在测试集上的性能。

`utils.py`：这个文件可以包含一些辅助函数或工具函数，用于数据预处理、超参数调优、结果分析等

`config.py`：配置文件 ，用于保存和管理项目的超参数、路径、模型配置等。这样可以更方便地调整和管理这些参数
