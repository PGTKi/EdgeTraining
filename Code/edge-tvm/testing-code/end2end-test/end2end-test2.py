import numpy as np
from nnvm import symbol as sym
from nnvm import graph
import nnvm.compiler.optimizer as optimizer
import nnvm.compiler.lr_scheduler as lr_scheduler
import nnvm
import tvm
import topi
from nnvm.testing.utils import create_workload
from tvm.contrib import graph_runtime

from data import load_cifar_10_data


batch_size = 1
num_class = 10
image_shape = (3, 32, 32)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)
dtype = "float32"


if __name__ == "__main__":
    """show it works"""

    cifar_10_dir = 'cifar-10-batches-py'

    train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = load_cifar_10_data(cifar_10_dir)

    print("Train data: ", train_data.shape)
    print("Train filenames: ", train_filenames.shape)
    print("Train labels: ", train_labels.shape)
    print("Test data: ", test_data.shape)
    print("Test filenames: ", test_filenames.shape)
    print("Test labels: ", test_labels.shape)
    print("Label names: ", label_names.shape)

    # Don't forget that the label_names and filesnames are in binary and need conversion if used.

    a = tvm.placeholder((1, 3, 32, 32), name="a")
    b = tvm.placeholder((1, 10), name="b")

    dense_weight =sym.Variable("dense_weight",init=np.empty((900, 10), dtype=dtype))
    # define network
    data = sym.Variable("data")
    y1 = sym.conv2d(data=data, channels=1, kernel_size=(3,3), padding=(0,0), use_bias=False, out_layout='NCHW')
    y2 = sym.flatten(y1)
    #y3 = sym.dense(y2, units=10, use_bias=False)
    y3 = sym.dense(y2, weight=dense_weight, use_bias=False)
    y4 = sym.softmax(y3)
    out = y4            # This is some of the loss function

    # create workload
    net, params = create_workload(out, batch_size, image_shape, dtype)
    #print(net.debug_str())

    target = tvm.target.create('llvm')
    #target = tvm.target.create('opencl')
    with nnvm.compiler.build_config(opt_level=0):
        graph, lib, params = nnvm.compiler.build(net, target, shape={"data": data_shape}, params=params)

    # create random input
    #ctx = tvm.opencl()
    ctx = tvm.context("llvm", 0)
    #data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
    data = train_data[:1].swapaxes(1,3).swapaxes(2,3).astype("float32")
    label = train_labels[:1]
    real_label = np.zeros(10, dtype='int8')
    for i in range(10):
        if i == label:
            real_label[i] = 1
    print(real_label)

    # create module
    module = graph_runtime.create(graph, lib, ctx)
    # set input and parameters
    module.set_input("data", data)
    #print("-----------------------")
    #for i in params.items():
    #    print(i)
    new_params = {key: value for key, value in params.items() if key != 'Adam_t'}
    module.set_input(**new_params)
    # run
    module.run()
    # get output
    out = module.get_output(0, tvm.nd.empty(out_shape))
    # convert to numpy
    out.asnumpy()

    # Print first 10 elements of output
    print("----------Output----------")
    print(out.asnumpy().flatten())




    base_lr = 0.1
    lr_factor = 0.5
    rescale_grad = 0.2
    wd = 0.1
    clip_gradient = 0.25

    scheduler = lr_scheduler.FactorScheduler(base_lr=base_lr, step=1, factor=lr_factor)
    opt = optimizer.SGD(learning_rate=base_lr, lr_scheduler=scheduler,
                        rescale_grad=rescale_grad, clip_gradient=clip_gradient,
                        wd=wd)
    opt_sym = opt.minimize(tvm.ndarray.array(((real_label - out.asnumpy().flatten()) ** 2), ctx=ctx), var=params['dense0_weight'])




