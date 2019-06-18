### 后续的进展

__目标：把sparsity flow加入地更完整到TVM框架中，实现对soarsity index的全支持（包括feature map和kernel）__


#### TODO
1. 看一下现有的sparsity方法如何支持我们的想法，分析影响
2. 查阅蒋老师提供的若干paper
3. 到目前为止，对于稀疏性训练的支持还是空白
4. 结构化稀疏性相对于稀疏性的优势在于同等稀疏度下少算tile（但CSR的方法会不会带来cost？）
5. 交付华为中期报告

#### Survey
[Here](https://shimo.im/docs/xQ4baX8mRsQuG0GZ/)

-----------------------------------------------

#### 5000x model compression in DNNs; But is it truly desirable?
2019-6/8 
By Prof. Yanzhi Wang

##### some points
1. __non-structured pruning is bad for acceleration.__ In non-structured pruning, sometime the memory used to store index is higher than that used to store weight. It is not suitable for CPU, GPU or even FPGA.
3. __structured pruning is better than non-structured pruning, but it is difficult to have no accuracy lost__
4. __channel pruning__ is similar to training a small network (or structured search). But the main difference is that in different group, the pruned channel can be different. For example, in Resnet, the pruned channel in shortcut does not necessarily fit that in the main flow. This can bring robustness to the model (proven by sensetime already). __We might need to think about what's the major difference between channel pruning and training a smaller model.__
5. __column pruning__ does not fit with acceleration method like winograd. And it might be slower than channel pruning.
6. __Pattern pruning__, which is a different case of column pruning, proven to be good. For example, in a $3 \times 3$ kernel, we keep 4 weight in this kernel (the number is better if it can be divided by 4), and optimize it.
7. __Connectivity pruning__ is the general case of channel pruning. Pruning an entire channel might be too aggressive. So it just prunes the connectivity between some of input channel and some output channel.
8. __Quantization__ might not be a general solution. TFLite can increase 15%-30% inference time with 2% accuracy drop.
9. Currently, fp16 is good. But fixed point might be some problem for general hardwares like CPU or GPU. For example, a fixed point 8 times a fixed point 8 get a fixed point 16 result. And to get a fp8 result we need to divide some figure (usually they do not just keep the largest 8 bits). And the transformation is very time-consuming. Let alone transfroming fixed point number to float point number


##### some facts
1. MobileNet is not good to be trained and not robust enough. It requires some training tricks like mixed up (30% image A + 70% image B) or warm up. It might be too aggressive to reduce some many weight. 
2. In CPU, branch and shift operation is very slow and should be avoid. We should use some bit-wise operation to replace them. 
3. Batch Normalization is good for training, but it is not friendly for quantization.
4. We can use 2 power of 2 number and add them as one to represent one weight.
5. Some time we split a dense layer into several block and do pruning seperately.
6. Prof Wang has proven that pruning YOLO 8x can achieve higher accuracy than tiny-YOLO (8x smaller).
7. We should avoid doing network search because GoogLe is very productive in this field.
8. Some training tricks: cosine annealing

