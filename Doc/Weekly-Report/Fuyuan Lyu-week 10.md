### Week 10 Report
#### 本周任务：
- 继续完成end-to-end testing code
- __进一步细化对结构性稀疏的支持__
  - __在考虑控制的粒度时候，应该以硬件的最低粒度为主__
  - __要详细了解GPU进行运算的粒度__
  - __控制粒度要考虑fusing的问题__

#### 下周任务：
- 继续完成end-to-end testing code
- 开始对结构性稀疏进行支持

-----------------

#### end to end testing code
目前已经搞定了data部分， 但卡在lower部分。

TVM本身并没有考虑training，但已经支持到了optimizer


#### 结构性稀疏
[硬件调研](https://shimo.im/docs/tpiTssCzx00aUElq/)

一周内，希望以channel为最低单位初步实现结构化稀疏的support



