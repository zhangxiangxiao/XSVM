Xiang's Kernel SVM Library in Torch 7

This library contains two SMO (sequential minimal optimization) variants for kernel support vector machines.
xsvm.simple(args): A simplified SMO algorithm[1]
xsvm.platt(args): John C. Platt's original SMO algorithm[2]

They can be used in this way:
model = xsvm.simple{C = 0.05, cache = true, kernel = kfunc} -- create model
model:train(dataset) -- Train on a dataset. The training error is returned.
model:test(dataset) -- Test on a dataset. The testing error is returned.
vmodel:f(x) -- The output function
model:g(x) -- The decision function (return -1 or 1)

xsvm.simple() and xsvm.platt() have the same protocols for controlling the parameter C (default 0.05), whether to cache (default false) and the kernel function (default is inner-product between vectors, i.e., linear model)

Generally speaking, xsvm.simple() is faster but its result is random because it does not exactly solve the problem. It also does not handle non-PSD kernels well. xsvm.platt() is the original SMO algorithm which gives consistent solution, and sometimes it will handle non-PSD kernels.

The dataset format follows the convention of Torch 7 tutorial, and I quote it here:
"A dataset is an object which implements the operator dataset[index] and implements the method dataset:size(). The size() methods returns the number of examples and dataset[i] has to return the i-th example. An example has to be an object which implements the operator example[field], where field often takes the value 1 (for input features) or 2 (for corresponding labels), i.e an example is a pair of input and output objects."
