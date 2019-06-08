#### resnet-18 结果
在RK3399 Mali上以800MHz频率（最高）运行，重复30次

| network | mean(ms) | std(ms) |
| -----|------| ---------------- |
| resnet-18-baseline | 96.41 | 3.90 | 
| resnet-18-pruned-wos | 64.52 | 3.71 | 


在RK3399 Mali上以200MHz频率（最低）运行，重复30次

| network | mean(ms) | std(ms) |
| -----|------| ---------------- |
| resnet-18-baseline | 283.62 | 2.31 | 
| resnet-18-pruned-wos | 189.36 | 1.53 | 
