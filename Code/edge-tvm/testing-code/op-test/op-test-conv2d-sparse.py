import numpy as np
import nnvm
import nnvm.symbol as sym
from nnvm.testing.check_computation import check_function
from nnvm import graph
import numpy as np
import time
import tvm
from nnvm.testing.utils import create_workload

data = sym.Variable("data")
#conv_kernel = sym.Variable("conv_kernel")
#conv_bias = sym.Variable("conv_bias")
sparse_kernel = sym.Variable("sparse_kernel")
#y = sym.conv2d_sparse(data=x, weight=conv_kernel, sparsity=sparse_kernel, bias=conv_bias, channels=1, kernel_size=(3,3), padding=(0,0), use_bias=True, out_layout='NCHW')
y = sym.conv2d_sparse(data=data, sparsity=sparse_kernel, channels=1, kernel_size=(3,3), padding=(0,0), use_bias=False, out_layout='NCHW')
out = y

# Test Graph compilation
# Once the API is well-defined, this part will be OK
g = graph.create(y)
print("-------------Starts----------------")
print(g.json())
print("-----------------------------------")
print(g.ir())
print("--------------Ends-----------------")


# Create workload
batch_size = 1
num_class = 10
image_shape = (3, 32, 32)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)
dtype = "float32"
net, params = create_workload(out, batch_size, image_shape, dtype)
print("-------------Starts----------------")
print(net.debug_str())
print("--------------Ends-----------------")



# NNVM.compiler.build
opt_level = 0
#target = tvm.target.intel_graphics()
target = tvm.target.create('opencl')
with nnvm.compiler.build_config(opt_level=opt_level):
    graph, lib, params = nnvm.compiler.build(net, target, shape={"data": data_shape}, params=params)


# create random input
#ctx = tvm.opencl()
#data = np.random.uniform(-1, 1, size=data_shape).astype("float32")





























'''


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
    check_function(y, forward=None, backward=None, numerical_grads=False, values=np.ones(shape['x'],dtype), dtype=dtype, shape=shape, additional_params={'kernel':[3,3], 'pad':[0,0],'stride':[1,1]})
localtime = time.asctime( time.localtime(time.time()) )
print("End time:" + localtime)
'''


