import numpy as np
import nnvm
import nnvm.symbol as sym
from nnvm.testing.check_computation import check_function
import numpy as np


x = sym.Variable("x")
y = sym.max_pool2d(x, pool_size=[2,2], strides=[2,2])

def backward(head_grads, x, kernel, pad, stride, **params):
    x_grad = np.zeros(x.shape, dtype="float32")
    ishape = x.shape
    oshape = head_grads.shape
    
    for n in range(0,oshape[0]):
        for c in range(0,oshape[1]):
            for o_h in range(0,oshape[2]):
                for o_w in range(0,oshape[3]):
                    i_h_start = stride[0]*o_h-pad[0]
                    i_w_start = stride[1]*o_w-pad[1]
                    i_h_end = min(i_h_start+kernel[0], oshape[2])
                    i_w_end = min(i_w_start+kernel[1], oshape[3])
                    i_h_start = max(i_h_start,0)
                    i_w_start = max(i_w_start,0)
                    max_idx_height = i_h_start
                    max_idx_weight = i_w_start
                    max_idx = x[n][c][max_idx_height][max_idx_weight]
                    for i_h_t in range(i_h_start, i_h_end):
                        for i_w_t in range(i_w_start, i_w_end):
                            if x[n][c][i_h_t][i_w_t] >= max_idx:
                                max_idx = x[n][c][i_h_t][i_w_t]
                                max_idx_height = i_h_t
                                max_idx_weight = i_w_t
                    x_grad[n][c][max_idx_height][max_idx_weight] += head_grads[n][c][o_h][o_w]                                

    return x_grad
 
def forward(x, kernel, pad, stride):
    return np.log(x)


dtype = "float32"
shape = {'x': (1, 3, 32, 32)}
check_function(y, forward=None, backward=backward, numerical_grads=False, in_range=(0.001, 2.0), dtype=dtype, shape=shape, additional_params={'kernel':[2,2], 'pad':[0,0],'stride':[2,2]})


#for _ in range(10000):
#    check_function(y, forward, backward, in_range=(0.001, 2.0), dtype=dtype, shape=shape)


