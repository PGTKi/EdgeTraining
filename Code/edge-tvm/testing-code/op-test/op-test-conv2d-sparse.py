import numpy as np
import nnvm
import nnvm.symbol as sym
from nnvm.testing.check_computation import check_function
from nnvm import graph
import numpy as np
import time

x = sym.Variable("x")
conv_kernel = sym.Variable("conv_kernel")
conv_bias = sym.Variable("conv_bias")
sparse_kernel = sym.Variable("sparse_kernel")
#y = sym.conv2d_sparse(data=x, weight=conv_kernel, sparsity=sparse_kernel, bias=conv_bias, channels=1, kernel_size=(3,3), padding=(0,0), use_bias=True, out_layout='NCHW')
y = sym.conv2d_sparse(data=x, weight=conv_kernel, sparsity=sparse_kernel, channels=1, kernel_size=(3,3), padding=(0,0), use_bias=False, out_layout='NCHW')


# Test Graph compilation
# Once the API is well-defined, this part will be OK
g = graph.create(y)
print("-------------Starts----------------")
print(g.json())
print("-----------------------------------")
print(g.ir())
print("--------------Ends-----------------")



# Check computation
def forward(x, conv_kernel, sparse_kernel, kernel, pad, stride, **args):
    ishape = x.shape
    return

def backward(head_grads, x, conv_kernel, sparse_kernel, kernel, pad, stride, **args):
    return

dtype = "float32"
shape = {'x': (1,1,3,3)}
localtime = time.asctime( time.localtime(time.time()) )
print("Start time:" + localtime)
for _ in range(1):
    check_function(y, forward=forward, backward=backward, numerical_grads=False, values=np.ones(shape['x'],dtype), dtype=dtype, shape=shape, additional_params={'kernel':[3,3], 'pad':[0,0],'stride':[1,1]})
localtime = time.asctime( time.localtime(time.time()) )
print("End time:" + localtime)


'''
batch_size = 1
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)

opt_level = 3
#target = tvm.target.intel_graphics()
target = tvm.target.create('opencl')
with nnvm.compiler.build_config(opt_level=opt_level):
    graph, lib, params = nnvm.compiler.build(
        net, target, shape={"data": data_shape}, params=params)
'''




