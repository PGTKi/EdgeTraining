import nnvm
import nnvm.symbol as sym
from nnvm.testing.check_computation import check_function, infer_shapes_dtypes
from nnvm.compiler.graph_attr import TCODE_TO_DTYPE, DTYPE_TO_TCODE
from nnvm.compiler import graph_util
from nnvm import graph
import numpy as np
import time
import tvm
from nnvm.testing.utils import create_sparse_workload
from tvm.contrib import graph_runtime, util
from tvm import rpc


def test_one_time(one_time_length=1000, Test_sparse = True, image_shape = (3, 32, 32)):
    # Hyper-parameter define
    batch_size = 1
    num_class = 10
    data_shape = (batch_size,) + image_shape
    out_shape = (batch_size, num_class)
    sparse_kernel_shape = (batch_size, 12)
    dtype = "float32"

    data = sym.Variable("data")
    sparse_kernel = sym.Variable("sparse_kernel", init=np.random.randint(0, 2, sparse_kernel_shape).astype(dtype))
    if Test_sparse:
        y1 = sym.conv2d_sparse(data=data, sparsity=sparse_kernel, channels=12, kernel_size=(3,3), padding=(0,0), use_bias=False, out_layout='NCHW')
    else:
        y1 = sym.conv2d(data=data, channels=10, kernel_size=(3,3), padding=(0,0), use_bias=False, out_layout='NCHW')
    # y = sym.flatten(y1)
    # y = sym.dense(y, units=10, use_bias=False)
    # y = sym.softmax(y)
    out = y1

    # Test Graph compilation
    # Once the API is well-defined, this part will be OK
    # g = graph.create(out)
    # print("-------------Starts----------------")
    # print(g.json())
    # print("-----------------------------------")
    # print(g.ir())
    # print("--------------Ends-----------------")

    # Create workload
    net, params = create_sparse_workload(out, batch_size, image_shape, dtype)
    # print("-------------Starts2---------------")
    # print(net.debug_str())
    # print(params)
    # print("--------------Ends2----------------")

    # Test Forward
    # NNVM-compiler build
    opt_level = 0
    target = tvm.target.mali()
    target_host = "llvm -target=aarch64-linux-gnu"
    with nnvm.compiler.build_config(opt_level=opt_level):
        graph, lib, params = nnvm.compiler.build(net, target=target, shape={"data": data_shape}, params=params, target_host=target_host)

    tmp = util.tempdir()
    lib_fname = tmp.relpath("net.tar")
    lib.export_library(lib_fname)
    remote = rpc.connect('59.78.6.204', 9090)
    remote.upload(lib_fname)
    rlib = remote.load_module("net.tar")

    ctx = remote.cl(0)

    # create random input
    real_data = np.random.uniform(-1, 1, size=data_shape).astype(dtype)
    real_sparse_kernel = np.array(([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])).astype(dtype)
    # real_sparse_kernel = np.random.randint(0, 2, sparse_kernel_shape).astype(dtype)

    # print(real_data)
    # print(real_sparse_kernel)

    # create module
    module = graph_runtime.create(graph, rlib, ctx)
    # set input and parameters
    module.set_input("data", real_data)
    if Test_sparse:
        module.set_input("sparse_kernel", real_sparse_kernel)
        module.set_input(**params)

    # run
    # localtime = time.asctime(time.localtime(time.time()))
    # print("Start time:" + localtime)
    starttime = time.time()
    for _ in range(one_time_length):
        module.run()
    endtime = time.time()
    # localtime = time.asctime(time.localtime(time.time()))
    # print("End time:" + localtime)
    print(endtime - starttime)

    # get output
    out = module.get_output(0)
    # convert to numpy
    out.asnumpy()

    # Print first 10 elements of output
    # print("-------------Starts3---------------")
    # # print(out.asnumpy().flatten()[0:10])
    # print(out)
    # print("--------------Ends3----------------")

    return endtime-starttime


def test_case():
    test_length = 20
    total_time = 0.
    test_sparse = True
    total_log = [0. for i in range(test_length)]
    for i in range(test_length):
        this_time = test_one_time(100, test_sparse, (3, 254, 254))
        total_time += this_time
        total_log[i] = this_time
    print(total_time/test_length)


if __name__ == "__main__":
    test_case()
