import numpy as np
import nnvm
import nnvm.symbol as sym
from nnvm.testing.check_computation import check_function
import numpy as np
import time

x = sym.Variable("x")
conv_kernel = sym.Variable("conv_kernel")
conv_bias = sym.Variable("conv_bias")
y = sym.conv2d(data=x, weight=conv_kernel, channels=3, kernel_size=(3,3), padding=(1,1), use_bias=False, out_layout='NCHW')

def backward(head_grads, x, conv_kernel, kernel, pad, stride):
    
    ishape = x.shape
    oshape = head_grads.shape
    kshape = conv_kernel.shape
    print(ishape)
    print(oshape)
    print(kshape)
    input_grad = np.zeros(ishape, dtype="float32")
    kernel_grad = np.zeros(kshape, dtype="float32")

    for n in range(0,oshape[0]):
        for o_c in range(0,oshape[1]):
            for o_h in range(0,oshape[2]):
                for o_w in range(0,oshape[3]):
                    baseline_h = stride[0]*o_h-pad[0]
                    baseline_w = stride[1]*o_w-pad[1]
                    i_h_start = max(baseline_h,0)
                    i_w_start = max(baseline_w,0)
                    i_h_end = min(baseline_h+kernel[0],ishape[2])
                    i_w_end = min(baseline_w+kernel[1],ishape[3])
                    for i_c in range(0,ishape[1]):
                        for i_h in range(i_h_start - baseline_h, i_h_end - baseline_h):
                            for i_w in range(i_w_start - baseline_w , i_w_end - baseline_w):
                                input_grad[n][i_c][baseline_h + i_h][baseline_w + i_w] += head_grads[n][o_c][o_h][o_w] * conv_kernel[o_c][i_c][i_h][i_w]  
                                kernel_grad[o_c][i_c][i_h][i_w] += head_grads[n][o_c][o_h][o_w] * x[n][i_c][baseline_h + i_h][baseline_w + i_w]                      

    return {
        "x": input_grad,
        "conv_kernel": kernel_grad
    }


dtype = "float32"
shape = {'x': (1, 3, 16, 16)}
localtime = time.asctime( time.localtime(time.time()) )
print("Start time:" + localtime)
for _ in range(10):
    check_function(y, forward=None, backward=None, numerical_grads=True, in_range=(-2.0, 2.0), dtype=dtype, shape=shape, additional_params={'kernel':[3,3], 'pad':[1,1],'stride':[1,1]})
localtime = time.asctime( time.localtime(time.time()) )
print("End time:" + localtime)
