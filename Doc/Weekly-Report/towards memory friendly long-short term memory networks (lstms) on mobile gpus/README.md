### Towards Memory Friendly Long-Short Term Memory Networks (LSTMs) on Mobile GPUs
#### 概述
##### 背景：
- LSTM在NLP问题（个人智能助理）中很重要
- mobile GPU上，LSTM效率很低，原因有两个：reduantant data movement & limited off-chip bandwidth

##### 解决方法:
| 问题 | 解决方案 |
| --- | --- |
| reduantant data movement |  **inter-cell level optimizations** that intelligently parallelize the originally sequentially executed LSTM cells | 
| limited offchip memory bandwidth | **intra-cell level optimizations** that dynamically skip the loads and computations of rows in the weight matrices with trivial contribution to the outputs |

#### 方法




