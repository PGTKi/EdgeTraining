# 端到端测试进度

本周的主要目标是让 end to end 测试跑起来

1. 修改 check_computation 的实现。让它可以通过 rpc，在rk3399上验证结果。修改后的代码放在 `demo/check_computation.py` 中
2. 补足 conv2d_grad 的 scheduling: 借鉴 mali 的 conv2d 和 arm_cpu 的 conv2d_transpose
