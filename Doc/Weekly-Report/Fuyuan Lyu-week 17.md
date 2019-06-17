#### 后续的进展

##### 目标：把sparsity flow加入地更完整到TVM框架中，实现对soarsity index的全支持（包括feature map和kernel）


TODO：
1. 看一下现有的sparsity方法如何支持我们的想法，分析影响
2. 查阅蒋老师提供的若干paper
3. 到目前为止，对于稀疏性训练的支持还是空白
4. 结构化稀疏性相对于稀疏性的优势在于同等稀疏度下少算tile（但CSR的方法会不会带来cost？）
5. 交付华为中期报告
