#### 本周进展
1. 已经完成pytorch-onnx-nnvm的通路，导入了baseline model
1. 正在对resnet18导出稀疏模型

#### 问题
1. 题目是否需要更改
2. 关于resnet18 sparse：
    - resnet-18 sparse卡在非conv层与conv层之间的维度匹配问题，可以解决
    - CCQ学长的剪枝似乎没有考虑layer与layer之间的连续性，即每个layer只有一个维度是独立的。解决方法是将部分filter填0，来匹配维度

#### 之后工作
1. 写新任务书给蒋老师
2. resnet-18搞定
2. 尽量做两个bonus point
3. 答辩完看J.I.T.
