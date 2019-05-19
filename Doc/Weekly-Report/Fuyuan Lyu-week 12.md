#### Test report on mali / opencl

| Sparsity_index  | num of channel calculated | time per 100 inference |
| ------------- | ------------- | ------------- |
| 12 | 1 | 0.631 |
| 6 | 2  | 0.768 |
| 4 | 3 | 0.522 |
| 3 | 4 | 0.598 | 
| 2 | 6 | 0.553 |
| 1 | 12 | 1.076 |
| NAN | 12 | 1.211 |

#### Test report on LLVM
| Sparsity_index  | num of channel calculated | time per 10^4 inference |
| ------------- | ------------- | ------------- |
| 12 | 1 | 0.787 |
| 6 | 2 | 0.868 |
| 4 | 3 | 0.809 |
| 3 | 4 | 0.801 | 
| 2 | 6 | 0.795 |
| 1 | 12 | 0.790 |
| NAN | 12 | 1+ ~ 10+ (not optimzied) |
