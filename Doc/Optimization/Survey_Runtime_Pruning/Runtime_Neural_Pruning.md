# Info

> @incollection{NIPS2017_6813,
> 
> title = {Runtime Neural Pruning},
> author = {Lin, Ji and Rao, Yongming and Lu, Jiwen and Zhou, Jie},
> booktitle = {Advances in Neural Information Processing Systems 30},
> editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
> pages = {2181--2191},
> year = {2017},
> publisher = {Curran Associates, Inc.},
> url = {http://papers.nips.cc/paper/6813-runtime-neural-pruning.pdf}
> 
> }

# Abstract

In this paper, we propose a **Runtime Neural Pruning (RNP) framework** which prunes the deep neural network dynamically at the runtime. Unlike existing neural pruning methods which produce a fixed pruned model for deployment, our method preserves the full ability of the original network and conducts pruning according to the input image and current feature maps adaptively. **The pruning is performed in a bottom-up, layer-by-layer manner, which we model as a Markov decision process and use reinforcement learning for training.** Since the ability of network is fully preserved, the balance point is easily adjustable according to the available resources. Our method can be applied to off-the-shelf network structures and reach a better tradeoff between speed and accuracy, especially with a large pruning rate.

> The agent judges the importance of each convolutional kernel and conducts channel-wise pruning conditioned on different samples, where the network is pruned more when the image is easier for the task. 

#### AMC: AutoML for Model Compression

Model compression is an effective technique to efficiently deploy neural network models on mobile devices which have limited computation resources and tight power budgets. Conventional model compression techniques rely on hand-crafted features and require domain experts to explore the large design space trading off among model size, speed, and accuracy, which is usually sub-optimal and time-consuming. In this paper, we propose AutoML for Model Compression (AMC) which leverages reinforcement learning to efficiently sample the design space and can improve the model compression quality. We achieved state-of-the-art model compression results in a fully automated way without any human efforts. Under 4 × FLOPs reduction, we achieved 2.7% better accuracy than the hand-crafted model compression method for VGG-16 on ImageNet. We applied this automated, push-the-button compression pipeline to MobileNet-V1 and achieved a speedup of 1.53× on the GPU (Titan Xp) and 1.95× on an Android phone (Google Pixel 1), with negligible loss of accuracy.

#### DEEP HIDDEN ANALYSIS: A STATISTICAL FRAMEWORK TO PRUNE FEATURE MAPS

In this paper, we propose a statistical framework to prune feature maps in 1-D deep convolutional networks. SoundNet is a pre-trained deep convolutional network that accepts raw audio samples as input. The feature maps generated at various layers of SoundNet have redundancy, which can be identified by statistical analysis. These redundant feature maps can be pruned from the network with a very minor reduction in the capability of the network. The advantage of pruning feature maps, is that computational complexity can be reduced in the context of using an ensemble of classifiers on the layers of SoundNet. Our experiments on acoustic scene classification demonstrate that ignoring 89% of feature maps reduces the performance by less than 3% with 18% reduction in computational complexity

#### Dynamic Channel Pruning: Feature Boosting and Suppression

Making deep convolutional neural networks more accurate typically comes at the cost of increased computational and memory resources. In this paper, we reduce this cost by exploiting the fact that the importance of features computed by convolutional layers is highly input-dependent, and propose feature boosting and suppression (FBS), a new method to predictively amplify salient convolutional channels and skip unimportant ones at run-time. FBS introduces small auxiliary connections to existing convolutional layers. In contrast to channel pruning methods which permanently remove channels, it preserves the full network structures and accelerates convolution by dynamically skipping unimportant input and output channels. FBS-augmented networks are trained with conventional stochastic gradient descent, making it readily available for many state-of-the-art CNNs. We compare FBS to a range of existing channel pruning and dynamic execution schemes and demonstrate large improvements on ImageNet classification. Experiments show that FBS can respectively provide5×and2×savings in compute on VGG-16 and ResNet-18, both with less than0.6%top-5 accuracy loss.

#### Dynamic Runtime Feature Map Pruning【笔记见仓库，有源码】

High bandwidth requirements are an obstacle for accelerating the training and inference of deep neural networks. Most previous research focuses on reducing the size of kernel maps for inference. We analyze parameter sparsity of six popular convolutional neural networks - AlexNet, MobileNet, ResNet-50, SqueezeNet, TinyNet, and VGG16. Of the networks considered, those using ReLU (AlexNet, SqueezeNet, VGG16) contain a high percentage of 0-valued parameters and can be statically pruned. Networks with Non-ReLU activation functions in some cases may not contain any 0-valued parameters (ResNet-50, TinyNet). We also investigate runtime feature map usage and find that input feature maps comprise the majority of bandwidth requirements when depth-wise convolution and point-wise convolutions used. We introduce dynamic runtime pruning of feature maps and show that 10% of dynamic feature map execution can be removed without loss of accuracy. We then extend dynamic pruning to allow for values within an epsilon of zero and show a further 5% reduction of feature map loading with a 1% loss of accuracy in top-1.

#### Compression of Deep Convolutional Neural Networks under Joint Sparsity Constraints

We consider the optimization of deep convolutional neural networks (CNNs) such that they provide good performance while having reduced complexity if deployed on either conventional systems utilizing spatial-domain convolution or lower complexity systems designed for Winograd convolution. Furthermore, we explore the universal quantization and compression of these networks. In particular, the proposed framework produces one compressed model whose convolutional filters can be made sparse either in the spatial domain or in the Winograd domain. Hence, one compressed model can be deployed universally on any platform, without need for re-training on the deployed platform, and the sparsity of its convolutional filters can be exploited for further complexity reduction in either domain. To get a better compression ratio, the sparse model is compressed in the spatial domain which has a less number of parameters. From our experiments, we obtain24.2×,47.7×and35.4×compressed models for ResNet-18, AlexNet and CT-SRCNN, while their computational cost is also reduced by4.5×,5.1×and23.5×, respectively.

#### Runtime Network Routing for Efficient Image Classification

In this paper, we propose a generic Runtime Network Routing (RNR) framework for efficient image classification, which selects an optimal path inside the network. Unlike existing static neural network acceleration methods, our method preserves the full ability of the original large network and conducts dynamic routing at runtime according to the input image and current feature maps. The routing is performed in a bottom-up, layer-by-layer manner, where we model it as a Markov decision process and use reinforcement learning for training. The agent determines the estimated reward of each sub-path and conducts routing conditioned on different samples, where a faster path is taken when the image is easier for the task. Since the ability of network is fully preserved, the balance point is easily adjustable according to the available resources. We test our method on both multi-path residual networks and incremental convolutional channel pruning, and show that RNR consistently outperforms static methods at the same computation complexity on both the CIFAR and ImageNet datasets. Our method can also be applied to off-the-shelf neural network structures and easily extended to other application scenarios.
