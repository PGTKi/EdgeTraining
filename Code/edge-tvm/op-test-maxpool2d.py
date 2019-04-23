import numpy as np
import nnvm
import nnvm.symbol as sym
from nnvm.testing.check_computation import check_function


x = sym.Variable("x")
y = sym.max_pool2d(x, pool_size=[2,2], strides=[2,2])
#y = sym.dense(sym.flatten(x), units=100)
#y = some_self_defined function


def backward(head_grads, data, out, kernel, pad, stride):
    ishape = data.shape
    oshape = out.shape
    height = ishape[2]
    width = ishape[3]
    pooled_height = oshape[2]
    pooled_width = oshape[3]
    kernel_h = kernel[0]
    kernel_w = kernel[1]
    stride_h = stride[0]
    stride_w = stride[1]
    pad_h = pad[0]
    pad_w = pad[1]
    in_offset = ishape[2] * ishape[3];
    out_offset = oshape[2] * oshape[3];
    for n in range(0,oshape[0]):
        for c in range(0,oshape[1]):
            for ph in range(0,pooled_height):
                for pw in range(0,pooled_width):
                    hstart = ph * stride_h - pad_h
                    wstart = pw * stride_w - pad_w
                    hend = min(hstart + kernel_h, height)
                    wend = min(wstart + kernel_w, width)
                    hstart = max(hstart, 0)
                    wstart = max(wstart, 0)
                    pool_index = ph * pooled_width + pw
                    max_idx = -1
                    found = False
                    for h in range(hstart, hend):
                        for w in range(wstart, wend):
                            idx = h * width + w
                            if in_data[idx] == out_data[pool_index]:
                                max_idx = idx
                                found = true
                                break
                        if found :
                            break
                    #  In the case where pad > 0 and kernel = 1, for example,
                    #  max_idx can be -1 reaching this step.
                    if max_idx >= 0:
                        in_grad[max_idx] += out_grad[pool_index]        
            in_data += in_offset
            in_grad += in_offset
            out_data += out_offset
            out_grad += out_offset
            
    '''
    const int height = ishape[2], width = ishape[3];
    const int pooled_height = oshape[2], pooled_width = oshape[3];
    const int kernel_h = kernel[0], kernel_w = kernel[1];
    const int pad_h = pad[0], pad_w = pad[1];
    const int stride_h = stride[0], stride_w = stride[1];
    const index_t in_offset = ishape[2] * ishape[3];
    const index_t out_offset = oshape[2] * oshape[3];
    for (index_t n = 0; n < oshape[0]; ++n) 
        for (index_t c = 0; c < oshape[1]; ++c) 
            for (int ph = 0; ph < pooled_height; ++ph) 
                for (int pw = 0; pw < pooled_width; ++pw)
                    int hstart = ph * stride_h - pad_h;
                    int wstart = pw * stride_w - pad_w;
                    int hend = std::min(hstart + kernel_h, height);
                    int wend = std::min(wstart + kernel_w, width);
                    hstart = std::max(hstart, 0);
                    wstart = std::max(wstart, 0);
                    const int pool_index = ph * pooled_width + pw;
                    int max_idx = -1;
                    bool found = false;
                    for (int h = hstart; h < hend; ++h)
                        for (int w = wstart; w < wend; ++w)
                            const int idx = h * width + w;
                            if (in_data[idx] == out_data[pool_index]) 
                                max_idx = idx;
                                found = true;
                                break;
                        if (found) break;
          
                    #  In the case where pad > 0 and kernel = 1, for example,
                    #  max_idx can be -1 reaching this step.
                    if (max_idx >= 0)
                        in_grad[max_idx] += out_grad[pool_index];           
            in_data += in_offset;
            in_grad += in_offset;
            out_data += out_offset;
            out_grad += out_offset;
            
    '''



def forward(x, kernel, pad, stride):
    return np.log(x)


dtype = "float32"
shape = {'x': (1, 3, 32, 32)}
#shape = {'x': (1000)}
check_function(y, forward=None, backward=backward, numerical_grads=False, in_range=(0.001, 2.0), dtype=dtype, shape=shape, additional_params={'kernel':[2,2], 'pad':[0,0],'stride':[2,2]})


#for _ in range(10000):
#    check_function(y, forward, backward, in_range=(0.001, 2.0), dtype=dtype, shape=shape)
