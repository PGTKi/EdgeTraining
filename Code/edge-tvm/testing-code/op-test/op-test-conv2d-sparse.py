import nnvm
import nnvm.symbol as sym
from nnvm.testing.check_computation import check_function
from nnvm import graph
import numpy as np
import time
import tvm
from nnvm.testing.utils import create_sparse_workload
from tvm.contrib import graph_runtime

# Hyper-parameter define
batch_size = 1
num_class = 10
image_shape = (3, 32, 32)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)
sparse_kernel_shape = (batch_size, 3, 32)
dtype = "float32"

data = sym.Variable("data")
sparse_kernel = sym.Variable("sparse_kernel", init=np.random.randint(0, 1, sparse_kernel_shape))
y = sym.conv2d_sparse(data=data, sparsity=sparse_kernel, channels=1, kernel_size=(3,3), padding=(0,0), use_bias=False, out_layout='NCHW')
y = sym.flatten(y)
y = sym.dense(y, units=10)
y =sym.softmax(y)
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
net, params = create_sparse_workload(out, batch_size, image_shape, dtype)
print("-------------Starts2---------------")
print(net.debug_str())
print("--------------Ends2----------------")

print("-------------Starts3---------------")
# NNVM-compiler build
opt_level = 0
target = tvm.target.create('llvm')
with nnvm.compiler.build_config(opt_level=opt_level):
    graph, lib, params = nnvm.compiler.build(net, target, shape={"data": data_shape}, params=params)
print("--------------Ends3----------------")

# create random input
ctx = tvm.context("llvm", 0)
real_data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
# create module
module = graph_runtime.create(graph, lib, ctx)
# set input and parameters
module.set_input("data", real_data)
module.set_input("sparse_kernel", np.random.randint(0, 1, sparse_kernel_shape))
module.set_input(**params)
# run
module.run()
# get output
out = module.get_output(0, tvm.nd.empty(out_shape))
# convert to numpy
out.asnumpy()

# Print first 10 elements of output
print(out.asnumpy().flatten()[0:10])
