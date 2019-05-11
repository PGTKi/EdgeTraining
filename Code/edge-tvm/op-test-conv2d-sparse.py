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
y = sym.conv2d_sparse(data=x, weight=conv_kernel, sparsity=sparse_kernel, channels=1, kernel_size=(3,3), padding=(0,0), use_bias=False, out_layout='NCHW')


g = graph.create(y)
print(g.json())
print("-----------------------------------")
print(g.ir())
