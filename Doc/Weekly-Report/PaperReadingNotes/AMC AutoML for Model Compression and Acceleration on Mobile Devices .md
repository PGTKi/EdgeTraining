# AMC: AutoML for Model Compression and Acceleration on Mobile Devices 阅读

## 目标：

使用**RL**自动化地寻找每一层的**压缩率**（剪枝目标的选择仍然是传统方法）

## 方法：

####trajectory

 从头到尾一层一层，一层输入一个state输出一个action，整个网络剪完（不微调）直接测试精度

####RL state

使用某一层的基本信息，层数t,n,c,h,w,stride,k,还有这一层的FLOPS以及前一层已经省掉的FLOPS，前一层的action和网络后面剩下的FLOPS，

####RL action

 这一层的剪枝率，具体剪哪些不在考虑范围。实验中使用least magnitude 剪 fine-grained,使用 max response 选择 channel pruning

####RL reward

和目标有关，

如果是对资源敏感，直接把资源限制设置在action space里面最大化精度。
$$
R_{e r r}=-Error
$$
如果是想寻求一个取舍，表示为乘积的形式。
$$
\begin{array}{c}{R_{\mathrm{FLOPs}}=-\text {Error} \cdot \log (\mathrm{FLOPs})} \\ {R_{\mathrm{Param}}=-\text {Error} \cdot \log (\# \mathrm{Param})}\end{array}
$$
另外，对于不同step的reward，取的都是最后把整个网络剪完之后测出的reward。
“each transition in an episode is $\left(s_{t}, a_{t}, R, s_{t+1}\right)$”

## 效果

CIFAR-10 computationally efficient: the RL can finish searching within 1 hour on a single GeForce GTX TITAN Xp GPU

The result we obtain has up to 60% compression ratio with even a little higher accuracy on both validation set and test set, which might be in light of the regularization effect of pruning

![1555816222417](.\Images\AMC\1555816222417.png)

![1555816171408](.\Images\AMC\1555816171408.png)

![1555816117930](.\Images\AMC\1555816117930.png)

## 思考

虽然提到了Generalization Ability，但是好像仅仅是任务上的拓展（detection），没有说在这个RL学到了适用于不同数据集和网络的统一策略。也就是这个RL agent充当了一个针对这个网络结构自动化搜索的工具，整个网络剪枝的过程中，如何选择被剪枝的channel仍然是原来的算法，这篇工作仅仅是搜到一个更加科学的剪枝率 。从给RL agent的信息来看，压缩率搜索利用的可能更多的是不断的尝试，而不是对网络某一层本身各种信息的理解。

作者对于reward函数的设置看上去很随意，把每一个timestep的reward设置成一样的，都是最后的测试得到的reward。但是细想之下也挺有道理的。首先每个timestep孤立来看，这一步的确对最后的reward的得到产生了影响，所以把它设成最后的reward合情合理。另外，这样的reward和state transition的配合也可以让agent整体规划，因为它前面做决策时就会考虑如何避免后面reward低的状态出现

作者训练速度有些出乎意料。他的actor网络和critic网络都是两个隐层的小网络，每层300个神经元，训练CIFAR的时候使用的验证集（用来计算reward）只有5K张图片，所以算是他另外一个工作中的proxy 训练方法。也可以解释如此快速的原因。