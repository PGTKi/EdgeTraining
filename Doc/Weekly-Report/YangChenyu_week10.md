# 杨晨宇第十周实验周报

## 仿照韩松的down sampling实验

训练好的网络，用RL寻找每一层的最佳down sampling rate
训练网络使用downsampling论文里面的网络训练方法，现在已有
用RL输入当前层数，卷积核大小等等基本信息，然后输出当前层最佳的down sampling rate

### 实验结果

收敛速度比较喜人，几十分钟即可，可以像预期一样通过价值函数调整策略。

比如当价值函数是 top1acc-(1e-9)*flops 的时候，最优策略基本是在第二层使用0.75的dsrate



![1557157770018](/PaperReadingNotes/Images/ycyReport/1557157770018.png)

### 实验意义

这个实验本身没有什么意义，算是对韩松论文的复现，只是为了后面扩展的一小步。