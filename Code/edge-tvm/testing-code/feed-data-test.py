import nnvm
import tvm
import nnvm.symbol as sym
import numpy as np
from nnvm.testing.utils import create_workload
from nnvm import graph
from nnvm.compiler import graph_util
from nnvm.testing.init import Xavier


batch_size = 1
num_class = 10
image_shape = (3, 32, 32)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)
dtype = "float32"

# name "data" is preferred!
data = sym.Variable("data")

# if you want to create_workload, you may have to let the system automatically generate layer kernels
# if you pass a self-defined kernel in, there will be error
#conv_kernel = sym.Variable("conv_kernel")
x = sym.conv2d(data=data, channels=1, kernel_size=(3,3), padding=(0,0), use_bias=False, out_layout='NCHW')
x = sym.flatten(data=x)
x = sym.dense(data=x, units= num_class, use_bias=False)

'''
params = {}
g = graph.create(x)
input_shapes, _ = graph_util.infer_shape(g, data=data_shape)
shape_dict = dict(zip(g.index.input_names, input_shapes))
np.random.seed(0)
initializer = Xavier()
for k, v in shape_dict.items():
    if k == 'data':
        print(k)
        continue
    print(k, end='\t')
    print(v)
    init_value = np.zeros(v).astype(dtype)
    initializer(k, init_value)
    params[k] = tvm.nd.array(init_value, ctx=tvm.opencl())
'''
    
'''
for k, v in shape_dict.items():
    if k == "data":
        continue
    init_value = np.zeros(v).astype(dtype)
    initializer(k, init_value)
    params[k] = tvm.nd.array(init_value, ctx=tvm.cpu(0))
'''


net, params = create_workload(x, batch_size, image_shape, dtype)
print(net.debug_str())

opt_level = 3
#target = tvm.target.create('opencl')
target = tvm.target.create('opencl')
with nnvm.compiler.build_config(opt_level= opt_level):
    graph, lib, params = nnvm.compiler.build(
        net, target, shape={"data": data_shape}, params= params)


