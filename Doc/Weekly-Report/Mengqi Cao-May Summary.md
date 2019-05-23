# 本月的工作

> Wang, Yulong, et al. "Interpret neural networks by identifying critical data routing paths." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2018.

原作者的主要贡献在于从神经网络可解释性的角度提出了一种对抗样本检测的方法。

我们尝试了原论文的思路，与network slimming的方法类似，在卷积层每个channel后面添加一个scalar(对应原论文中的"control gate")，并在损失函数中加入L1范数使其趋近于零。我们利用这个标量去衡量对应channel的重要性，当control gate value小于某个阈值时，对应的channel将会被剪掉。

实验使用了cifar100数据集，设置全局剪枝率，在预训练模型上保留原网络参数，为每一张图片训练一组control gate values。对于整个训练集的所有图片，将不同网络层的control gate values分别累加，全局排序所有的control gate values，按照全局剪枝率保留control gate values大的channel，control gate value小的channel将会被剪掉。对于剪枝后的网络再进行fine-tune。

目前方法存在的问题是：将每一张图片训练出来的"小"网络归并成一个"大"网络。对于预训练的VGG16（预训练模型在测试集上的准确率为0.82），如果只归并出一个剪枝率为0.9的三分类网络（比如只训练一个识别"猫"、"狗"或者"其它"）的准确率将达到0.97；归并出一个剪枝率为0.9的十分类网络的准确率为0.83，但剪枝率为0.9的全分类网络准确率只能达到0.02。

# 下月的计划

接下来端训练算法部分将由两位学生分别实现两组方案，是对目前两篇channel pruning的工作的跟进：

#### 方案一

> Gao, X., Zhao, Y., Dudziak, L., Mullins, R., & Xu, C. Z. (2018). Dynamic Channel Pruning: Feature Boosting and Suppression. *arXiv preprint arXiv:1810.05331*.

本篇工作的主要贡献在于为预训练模型的每一层卷积网络设计了一个子网络(对应原文中的channel saliency predictor) 。子网络的输出用来预测这一层被激活的卷积，从而减少inference阶段的计算量，达到网络加速。

我们将尝试在training阶段同时训练原网络参数与子网络参数。

#### 方案二

> Huang, Zehao, and Naiyan Wang. "Data-driven sparse structure selection for deep neural networks." *Proceedings of the European Conference on Computer Vision (ECCV)*. 2018.

本篇是图森科技在ECCV 2018发表的工作，剪枝的方法是在每个channel后增加一个标量参数作为衡量神经元输出，同时对于不可导的L1范数设计了Proximal Gradient算法实现反向传播。本篇工作与我们目前的control gate的想法不谋而合，而且是在网络参数训练过程中，也符合目前的任务场景。

我们将尝试对这个标量参数进行量化，把这个标量设计成一种"0-1开关"，重新设计0-1开关的反向传播算法。