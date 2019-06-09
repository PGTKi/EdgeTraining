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


在RK3399 Mali上以800MHz频率（最高）运行，CPU锁频为最高，重复30次

| network | mean(ms) | std(ms) |
| -----|------| ---------------- |
| resnet-18-baseline | 84.69 | 2.08 | 
| resnet-18-pruned-wos | 54.19 | 1.03 | 

#### Profiling
##### CPU
![CPU](https://github.com/acada-sjtu/EdgeTraining/blob/master/Doc/Weekly-Report/CPU.jpg)
##### GPU
![GPU](https://github.com/acada-sjtu/EdgeTraining/blob/master/Doc/Weekly-Report/GPU.jpg)
