--[[
Xiang's Kernel SVM :)
By Xiang Zhang @ New York University
10/06/2012, Version 0.3

---------------------------------------------

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

---------------------------------------------

DOCUMENTATION

This library contains SMO (sequential minimal optimization) variants for kernel
support vector machines.
xsvm.simple(args): A simplified SMO algorithm[1]
xsvm.platt(args): John C. Platt's original SMO algorithm[2]
xsvm.tweaked(args): A tweaked heuristic version of Platt's algorithm
xsvm.vectorized(args): A vectorized version of the tweaked algorithm
xsvm.fast(args): A vectorized version of Platt's original SMO algorithm

They can be used in this way:
model = xsvm.simple{C = 0.05, cache = true, kernel = kfunc} -- create model
model:train(dataset) -- Train on a dataset. The training error is returned.
model:test(dataset) -- Test on a dataset. The testing error is returned.
model:f(x) -- The output function
model:g(x) -- The decision function (return -1 or 1 as a tensor)
model:nsv() -- Query how many support vectors are there for a trained model

xsvm.simple(), xsvm.platt() and xsvm.tweaked() have the same protocols for
controlling the parameter C (default 0.05), whether to cache (default false)
and the kernel function (default is inner-product between vectors, i.e.,
linear model). You will not be able to control whether to cache in
xsvm.vectorized() and xsvm.fast(). They must enable cache for vectorization.

We recommend you to use xsvm.vectorized() if memory is not a constraint.
Otherwise, xsvm.tweaked() is probably the best option.

Generally speaking, xsvm.simple() is just an example code to understand the
algorithm. Do not use it in actual applications, because the result is random
as it does not exactly solve the problem. xsvm.platt() is the original SMO
algorithm which gives consistent solution. xsvm.tweaked() is a heuristic
version of the original SMO algorithm, which is much faster, but its optimality
may not be as good (but not bad as well). xsvm.fast() is the vectorized version
of xsvm.platt(). xsvm.vectorized() is the vectorized version of xsvm.tweaked().

The speed comparison of the algorithms are generally as the follows ('>'
represents faster than):
xsvm.vectorized() > xsvm.fast() > xsvm.tweaked() > xsvm.simple() > xsvm.platt()

The memory usage comparision of the algorithms are as follows ('>' represents
more memory usage):
xsvm.vectorized() = xsvm.fast() > xsvm.tweaked{cache = true} = xsvm.platt{
cache = true} = xsvm.simple{cache = true} > xsvm.tweaked{cache = false} =
xsvm.platt{cache = false} = xsvm.simple{cache = false}

The optimality comparison of the algorithms are as follows ('>' represents
better optimality guarantee):
xsvm.fast() = xsvm.platt() > xsvm.vectorized() = xsvm.tweaked() > xsvm.simple()

The dataset format follows the convention of Torch 7 tutorial, and I quote it
here:
"A dataset is an object which implements the operator dataset[index] and
implements the method dataset:size(). The size() methods returns the number of
examples and dataset[i] has to return the i-th example. An example has to be
an object which implements the operator example[field], where field often
takes the value 1 (for input features) or 2 (for corresponding labels), i.e
an example is a pair of input and output objects."

The support vectors are stored in the table model.a, where if model.a[i] = nil
it means the dual variable with respect to the i training sample is 0. Support
vectors and their labels are stored in table model.x and model.y respectively,
following the same convention as model.a. The bias parameter is stored at
model.b.

---------------------------------------------

REFERENCES

[1] Andrew Ng. The Simplified SMO Algorithm. Lecture Notes for Stanford CS229
in Autumn 2009. http://cs229.stanford.edu/materials/smo.pdf
[2] John C. Platt. Sequential minimal optimization: A fast algorithm for
training support vector machines. Technical Report MSR-TR-98-14 April 21, 1998

]]

-- The namespace
xsvm = {}

-- Create a new xsvm simplified trainer with kernel
-- kernel: the kernel function (callable); C: regularization parameter
-- tol: Numerical tolerance; maxi_passes: max number of times to iterate over a's without changing
-- ratio_alpha_changes: a ratio heuristic for stopping the computation.
-- cache: whether to cache the kernel
function xsvm.simple(args)
   local model = {a = {}, x = {}, y = {}, b = 0}
   args = args or {}
   -- If kernel undefined, using linear kernel
   local kernel = args.kernel or function (x1,x2) return torch.dot(x1,x2) end
   -- Default C is 1
   local C = args.C or 0.05
   -- Default cache is false
   local cache = args.cache or false
   -- Default tolerance is 1e-3
   local tol = args.tol or 1e-3
   -- Default max_passes is 3
   local max_passes = args.max_passes or 3
   -- Default ratio alpha changes is 1%
   local ratio_alpha_changes = args.ratio_alpha_changes or 0.01
   -- Cache helper
   local kcache = torch.zeros(1)
   local kcflag = torch.zeros(1):byte()
   -- Initializing the kernel cache
   local function kcache_init(dataset)
      -- Allocate cache tensors
      kcache = torch.zeros(dataset:size()*(dataset:size()+1)/2)
      kcflag = torch.zeros(dataset:size()*(dataset:size()+1)/2):byte()
   end
   -- Query the cached value
   local function kcache_query(dataset, i, j)
      -- Make sure i is larger
      if j > i then i,j = j,i end
      -- The index
      local ind = i*(i-1)/2+j
      -- Test the flags and compute the kernel
      if kcflag[ind] == 0 then
	 kcache[ind] = kernel(dataset[i][1], dataset[j][1])
	 kcflag[ind] = 1
      end
      -- Return the cache
      return kcache[ind]
   end
   -- kcache utilized f function
   local function kcache_f(dataset, i)
      local result = torch.zeros(1)
      -- Iterate over all pairs
      for k,v in pairs(model.a) do
	 result[1] = result[1] + model.y[k]*model.a[k]*kcache_query(dataset, k, i)
      end
      result[1] = result[1] + model.b
      -- Return result as a tensor
      return result
   end
   -- Clean up the cache
   local function kcache_clean()
      kcache = torch.zeros(1)
      kcflag = torch.zeros(1):byte()
   end
   -- A cached version of the simplified SMO algorithm (private function)
   local function train_cache(dataset)
      -- Allocate training variable a
      local a = torch.zeros(dataset:size())
      -- Set bias variable
      model.b = 0
      -- Just for the easiness of programming
      local y = torch.zeros(dataset:size())
      for i = 1, dataset:size() do
	 y[i] = dataset[i][2][1]
      end
      -- Start the algorithm
      kcache_init(dataset)
      local passes = 0
      local e = torch.zeros(dataset:size())
      while passes < max_passes do
	 local num_changed_alphas = 0
	 for i = 1,dataset:size() do
	    e[i] = kcache_f(dataset, i)[1] - y[i]
	    -- Violates KKT condition
	    if ((y[i]*e[i] < -tol and a[i] < C) or (y[i]*e[i] > tol and a[i] > 0)) then
		-- Sample j
		local j = math.random(1,dataset:size())
		while i == j do j = math.random(1,dataset:size()) end
		-- Calculate e(j)
		e[j] = kcache_f(dataset,j)[1] - y[j]
		-- Save old a
		local ai = a[i]
		local aj = a[j]
		-- Compute L and H
		local L = 0
		local H = C
		if y[i] == y[j] then
		   L = math.max(0, ai+aj-C)
		   H = math.min(C, ai+aj)
		else
		   L = math.max(0, aj-ai)
		   H = math.min(C, C+aj-ai)
		end
		-- Decide feasibility
		local feasible = true
		if L == H then feasible = false end
		local eta = 0
		local kij = 0
		local kii = 0
		local kjj = 0
		-- Compute eta update
		if feasible == true then
		   kij = kcache_query(dataset,i,j)
		   kii = kcache_query(dataset,i,i)
		   kjj = kcache_query(dataset,j,j)
		   eta = 2*kij-kii-kjj
		   if eta >= 0 then feasible = false end
		end
		-- Compute aj update
		if feasible == true then
		   a[j] = a[j] - y[j]*(e[i]-e[j])/eta
		   if a[j] > H then
		      a[j] = H
		   elseif a[j] < L then
		      a[j] = L
		   end
		   if math.abs(aj-a[j]) < 1e-5 then
		      feasible = false
		      a[j] = aj
		   end
		end
		-- Compute ai update and b
		local b1 = 0
		local b2 = 0
		if feasible == true then
		   a[i] = a[i] + y[i]*y[j]*(aj-a[j])
		   b1 = model.b - e[i] - y[i]*(a[i]-ai)*kii - y[j]*(a[j]-aj)*kij
		   b2 = model.b - e[j] - y[i]*(a[i]-ai)*kij - y[j]*(a[j]-aj)*kjj
		   if a[i] > 0 and a[i] < C then
		      model.b = b1
		   elseif a[j] > 0 and a[j] < C then
		      model.b = b2
		   else
		      model.b = (b1+b2)/2
		   end
		   num_changed_alphas = num_changed_alphas + 1
		end
		-- Add to support vector list
		if feasible == true then
		   -- Add/Remove to support vector list
		   if a[i] > 0 then
		      model.a[i] = a[i]
		      model.x[i] = dataset[i][1]
		      model.y[i] = y[i]
		   else
		      model.a[i] = nil
		      model.x[i] = nil
		      model.y[i] = nil
		   end
		   -- Add/Remove to support vector list
		   if a[j] > 0 then
		      model.a[j] = a[j]
		      model.x[j] = dataset[j][1]
		      model.y[j] = y[j]
		   else
		      model.a[j] = nil
		      model.x[j] = nil
		      model.y[j] = nil
		   end
		end
	    end
	 end
	 -- Decide whether to stop
	 if num_changed_alphas <= dataset:size()*ratio_alpha_changes then
	    passes = passes + 1
	 else
	    passes = 0
	 end
      end
      -- Cleanup the caching
      kcache_clean()
   end
   -- A non-cached version of the simplified SMO algorithm (private function)
   local function train_ncache(dataset)
      -- Allocate training variable a
      local a = torch.zeros(dataset:size())
      -- Set bias variable
      model.b = 0
      -- Just for the easiness of programming
      local y = torch.zeros(dataset:size())
      for i = 1, dataset:size() do
	 y[i] = dataset[i][2][1]
      end
      -- Start the algorithm
      local passes = 0
      local e = torch.zeros(dataset:size())
      while passes < max_passes do
	 local num_changed_alphas = 0
	 for i = 1,dataset:size() do
	    e[i] = model:f(dataset[i][1])[1] - y[i]
	    -- Violates KKT condition
	    if ((y[i]*e[i] < -tol and a[i] < C) or (y[i]*e[i] > tol and a[i] > 0)) then
		-- Sample j
		local j = math.random(1,dataset:size())
		while i == j do j = math.random(1,dataset:size()) end
		-- Calculate e(j)
		e[j] = model:f(dataset[j][1])[1] - y[j]
		-- Save old a
		local ai = a[i]
		local aj = a[j]
		-- Compute L and H
		local L = 0
		local H = C
		if y[i] == y[j] then
		   L = math.max(0, ai+aj-C)
		   H = math.min(C, ai+aj)
		else
		   L = math.max(0, aj-ai)
		   H = math.min(C, C+aj-ai)
		end
		-- Decide feasibility
		local feasible = true
		if L == H then feasible = false end
		local eta = 0
		local kij = 0
		local kii = 0
		local kjj = 0
		-- Compute eta update
		if feasible == true then
		   kij = kernel(dataset[i][1], dataset[j][1])
		   kii = kernel(dataset[i][1], dataset[i][1])
		   kjj = kernel(dataset[j][1], dataset[j][1])
		   eta = 2*kij-kii-kjj
		   if eta >= 0 then feasible = false end
		end
		-- Compute aj update
		if feasible == true then
		   a[j] = a[j] - y[j]*(e[i]-e[j])/eta
		   if a[j] > H then
		      a[j] = H
		   elseif a[j] < L then
		      a[j] = L
		   end
		   if math.abs(aj-a[j]) < 1e-5 then
		      feasible = false
		      a[j] = aj
		   end
		end
		-- Compute ai update and b
		local b1 = 0
		local b2 = 0
		if feasible == true then
		   a[i] = a[i] + y[i]*y[j]*(aj-a[j])
		   b1 = model.b - e[i] - y[i]*(a[i]-ai)*kii - y[j]*(a[j]-aj)*kij
		   b2 = model.b - e[j] - y[i]*(a[i]-ai)*kij - y[j]*(a[j]-aj)*kjj
		   if a[i] > 0 and a[i] < C then
		      model.b = b1
		   elseif a[j] > 0 and a[j] < C then
		      model.b = b2
		   else
		      model.b = (b1+b2)/2
		   end
		   num_changed_alphas = num_changed_alphas + 1
		end
		-- Add to support vector list
		if feasible == true then
		   -- Add/Remove to support vector list
		   if a[i] > 0 then
		      model.a[i] = a[i]
		      model.x[i] = dataset[i][1]
		      model.y[i] = y[i]
		   else
		      model.a[i] = nil
		      model.x[i] = nil
		      model.y[i] = nil
		   end
		   -- Add/Remove to support vector list
		   if a[j] > 0 then
		      model.a[j] = a[j]
		      model.x[j] = dataset[j][1]
		      model.y[j] = y[j]
		   else
		      model.a[j] = nil
		      model.x[j] = nil
		      model.y[j] = nil
		   end
		end
	    end
	 end
	 -- Decide whether to stop
	 if num_changed_alphas <= dataset:size()*ratio_alpha_changes then
	    passes = passes + 1
	 else
	    passes = 0
	 end
      end
   end
   -- Train on a dataset
   function model:train(dataset)
      if cache == true then
	 train_cache(dataset)
      else
	 train_ncache(dataset)
      end
      return model:test(dataset)
   end
   -- Test on a dataset
   function model:test(dataset)
      -- Counter for wrong classification
      local error = 0
      for i = 1,dataset:size() do
	 -- Iterate error rate computation
	 if torch.sum(torch.ne(model:g(dataset[i][1]), dataset[i][2])) == 0 then
	    error = error*(i-1)/i
	 else
	    error = (error*i-error+1)/i
	 end
      end
      return error
   end
   -- The decision function
   function model:f(x)
      local result = torch.zeros(1)
      -- Iterate over all pairs
      for k,v in pairs(model.a) do
	 result[1] = result[1] + model.y[k]*model.a[k]*kernel(model.x[k],x)
      end
      result[1] = result[1] + model.b
      -- Return result as a tensor
      return result
   end
   -- The indicator function
   function model:g(x)
      local result = model:f(x)
      if result[1] >= 0 then
	 return torch.ones(1)
      else
	 return -torch.ones(1)
      end
   end
   -- The number of support vectors
   function model:nsv()
      local count = 0
      for k,v in pairs(model.a) do
	 count = count + 1
      end
      return count
   end
   -- Return the object
   return model
end

-- Platt's original svm algorithm
-- C: the regularization parameter; cache: whether to cache the kernel function
-- kernel: the kernel function; tol: tolerance on violation of KKT conditions
-- eps: the eps to detect change; maxIterRatio: maximum number of iterations is this number times dataset
function xsvm.platt(args)
   local model = {a = {}, x = {}, y = {}, b = 0}
   args = args or {}
   -- If kernel undefined, using linear kernel
   local kernel = args.kernel or function (x1,x2) return torch.dot(x1,x2) end
   -- Default C is 1
   local C = args.C or 0.05
   -- Default cache is false
   local cache = args.cache or false
   -- Default tolerance is 1e-3
   local tol = args.tol or 1e-3
   -- Default eps (round-off error on Mercer condition) is 1e-3
   local eps = args.eps or 1e-3
   -- Maximum number of iterations is this number times dataset size
   local maxIterRatio = args.maxIterRatio or 20
   -- Recording the number of non-zero & non-C alpha
   local nbound = 0
   -- The error cache
   local ecache = torch.zeros(1)
   -- Cache helper
   local kcache = torch.zeros(1)
   local kcflag = torch.zeros(1):byte()
   -- Initializing the kernel cache
   local function kcache_init(dataset)
      -- Allocate cache tensors
      kcache = torch.zeros(dataset:size()*(dataset:size()+1)/2)
      kcflag = torch.zeros(dataset:size()*(dataset:size()+1)/2):byte()
   end
   -- Query the cached value
   local function kcache_query(dataset, i, j)
      -- Make sure i is larger
      if j > i then i,j = j,i end
      -- The index
      local ind = i*(i-1)/2+j
      -- Test the flags and compute the kernel
      if kcflag[ind] == 0 then
	 kcache[ind] = kernel(dataset[i][1], dataset[j][1])
	 kcflag[ind] = 1
      end
      -- Return the cache
      return kcache[ind]
   end
   -- kcache utilized f function
   local function kcache_f(dataset, i)
      local result = torch.zeros(1)
      -- Iterate over all pairs
      for k,v in pairs(model.a) do
	 result[1] = result[1] + model.y[k]*model.a[k]*kcache_query(dataset, k, i)
      end
      result[1] = result[1] + model.b
      -- Return result as a tensor
      return result
   end
   -- Clean up the cache
   local function kcache_clean()
      kcache = torch.zeros(1)
      kcflag = torch.zeros(1):byte()
   end
   -- Cached helper takeStep function
   local function takeStep_cache(dataset, i1, i2, E1, E2)
      if i1 == i2 then
	 return 0
      end
      -- Allocating values
      local alph1 = model.a[i1] or 0
      local alph2 = model.a[i2] or 0
      local y1 = dataset[i1][2][1]
      local y2 = dataset[i2][2][1]
      local s = y1*y2
      local L = 0
      local H = C
      if y1 == y2 then
	 L = math.max(0, alph2 + alph1 - C)
	 H = math.min(C, alph2 + alph1)
      else
	 L = math.max(0, alph2 - alph1)
	 H = math.min(C, C + alph2 - alph1)
      end
      -- Compute the kernel values and step size
      local k11 = kcache_query(dataset,i1,i1)
      local k12 = kcache_query(dataset,i1,i2)
      local k22 = kcache_query(dataset,i2,i2)
      local eta = k11 + k22 - 2*k12
      local a1 = alph1
      local a2 = alph2
      -- Check feasibility of eta (Mercer kernel)
      if eta > 0 then
	 -- Compute the new value of a2
	 a2 = alph2 + y2*(E1-E2)/eta
	 if a2 < L then
	    a2 = L
	 elseif a2 > H then
	    a2 = H
	 end
      else
	 local f1 = y1*(E1-model.b)-alph1*k11-s*alph2*k12
	 local f2 = y1*(E2-model.b)-s*alph1*k12-alph2*k22
	 local L1 = alph1 + s*(alph2 - L)
	 local H1 = alph1 + s*(alph2 - H)
	 local Lobj = L1*f1 + L*f2 + L1*L1*k11/2 + L*L*k22/2 + s*L*L1*k12
	 local Hobj = H1*f1 + H*f2 + H1*H1*k11/2 + H*H*k22/2 + s*H*H1*k12
	 -- Determine the new value of a2
	 if Lobj < Hobj - eps then
	    a2 = L
	 elseif Lobj > Hobj + eps then
	    a2 = H
	 else
	    a2 = alph2
	 end
      end
      -- Do we change alph2 enough?
      if math.abs(a2-alph2) < eps*(a2 + alph2 + eps) then
	 return 0
      end
      -- Update a1
      a1 = alph1 + s*(alph2 - a2)
      -- Update multiplier a1
      if a1 > 0 then
	 model.a[i1] = a1
	 model.x[i1] = dataset[i1][1]
	 model.y[i1] = dataset[i1][2][1]
	 -- Keep track of nbound values
	 if a1 < C and (alph1 >= C or alph1 <= 0) then
	    nbound = nbound + 1
	 elseif a1 >= C and alph1 < C and alph1 > 0 then
	    nbound = nbound - 1
	 end
      else
	 model.a[i1] = nil
	 model.x[i1] = nil
	 model.y[i1] = nil
	 if alph1 < C and alph1 > 0 then
	    nbound = nbound - 1
	 end
      end
      -- Update multiplier a2
      if a2 > 0 then
	 model.a[i2] = a2
	 model.x[i2] = dataset[i2][1]
	 model.y[i2] = dataset[i2][2][1]
	 -- Keep track of nbound values
	 if a2 < C and (alph2 >= C or alph2 <= 0) then
	    nbound = nbound + 1
	 elseif a2 >= C and alph2 < C and alph2 > 0 then
	    nbound = nbound - 1
	 end
	 -- Keep track of ecache
      else
	 model.a[i2] = nil
	 model.x[i2] = nil
	 model.y[i2] = nil
	 if alph2 < C and alph2 > 0 then
	    nbound = nbound - 1
	 end
      end
      -- Update the bias b
      local b = model.b
      local b1 = model.b - E1 - y1*(a1-alph1)*k11 - y2*(a2-alph2)*k12
      local b2 = model.b - E2 - y1*(a1-alph1)*k12 - y2*(a2-alph2)*k22
      if a1 > 0 and a1 < C then
	 model.b = b1
      elseif a2 > 0 and a2 < C then
	 model.b = b2
      elseif L ~= H then
	 model.b = (b1+b2)/2
      end
      -- Update ecache
      local kk1 = 0
      local kk2 = 0
      -- Update ecache for i1
      if a1 > 0 and a1 < C then
	 if alph1 <= 0 or alph1 >= C then
	    ecache[i1] = kcache_f(dataset,i1)[1] - y1
	 else
	    ecache[i1] = ecache[i1] + (model.b - b) + (a1-alph1)*y1*k11 + (a2-alph2)*y2*k12
	 end
      end
      -- Update cache for i2
      if a2 > 0 and a2 < C then
	 if alph2 <= 0 or alph2 >= C then
	    ecache[i2] = kcache_f(dataset,i2)[1] - y2
	 else
	    ecache[i2] = ecache[i2] + (model.b - b) + (a1-alph1)*y1*k12 + (a2-alph2)*y2*k22
	 end
      end
      -- Update cache for everybody we care about
      for k, v in pairs(model.a) do
	 if model.a[k] < C and k ~= i1 and k ~= i2 then
	    kk1 = kcache_query(dataset,i1, k)
	    kk2 = kcache_query(dataset,i2, k)
	    ecache[k] = ecache[k] + (model.b - b) + (a1-alph1)*y1*kk1 + (a2-alph2)*y2*kk2
	 end
      end
      return 1
   end
   -- Examing an example cached version
   local function examineExample_cache(dataset, i2, E2)
      local y2 = dataset[i2][2][1]
      local alph2 = model.a[i2] or 0
      local r2 = E2*y2
      local E1 = 0
      local i1 = 0
      if (r2 < -tol and alph2 < C) or (r2 > tol and alph2 > 0) then
	 -- Second heuristics
	 if nbound > 1 then
	    if E2 > 0 then
	       E1 = math.huge
	       for k,v in pairs(model.a) do
		  if model.a[k] < C then
		     if ecache[k] < E1 then
			E1 = ecache[k]
			i1 = k
		     end
		  end
	       end
	    else
	       E1 = -math.huge
	       for k,v in pairs(model.a) do
		  if model.a[k] < C then
		     if ecache[k] > E1 then
			E1 = ecache[k]
			i1 = k
		     end
		  end
	       end
	    end
	    if takeStep_cache(dataset, i1, i2, E1, E2) > 0 then
	       return 1
	    end
	 end
	 -- Heuristic hierarchy
	 for i1,v in pairs(model.a) do
	    if model.a[i1] < C then
	       E1 = ecache[i1]
	       if takeStep_cache(dataset, i1, i2, E1, E2) > 0 then
		  return 1
	       end
	    end
	 end
	 for i1 = 1, dataset:size() do
	    E1 = kcache_f(dataset,i1)[1] - dataset[i1][2][1]
	    if takeStep_cache(dataset, i1, i2, E1, E2) > 0 then
	       return 1
	    end
	 end
      end
      return 0
   end
   -- A cached version of the platt's SMO algorithm (private function)
   local function train_cache(dataset)
      nbound = 0
      kcache_init(dataset)
      ecache = torch.zeros(dataset:size())
      model.a = {}
      model.b = 0
      model.x = {}
      model.y = {}
      local numChanged = 0
      local examineAll = 1
      local maxIter = maxIterRation*dataset:size()
      local numIter = 0
      while numIter < maxIter and (numChanged > 0 or examineAll == 1) do
	 numChanged = 0
	 -- Examine all
	 if examineAll == 1 then
	    for i2 = 1,dataset:size() do
	       numChanged = numChanged + examineExample_cache(dataset, i2, kcache_f(dataset,i2)[1]-dataset[i2][2][1])
	    end
	 else
	    for i2, v in pairs(model.a) do
	       if model.a[i2] < C then
		  numChanged = numChanged + examineExample_cache(dataset, i2, ecache[i2])
	       end
	    end
	 end
	 -- Loop value
	 if examineAll == 1 then
	    examineAll = 0
	 elseif numChanged == 0 then
	    examineAll = 1
	 end
	 numIter = numIter + 1
      end
      ecache = torch.zeros(1)
      kcache_clean()
   end
   -- Non-cached helper takeStep function
   local function takeStep_ncache(dataset, i1, i2, E1, E2)
      if i1 == i2 then
	 return 0
      end
      -- Allocating values
      local alph1 = model.a[i1] or 0
      local alph2 = model.a[i2] or 0
      local y1 = dataset[i1][2][1]
      local y2 = dataset[i2][2][1]
      local s = y1*y2
      local L = 0
      local H = C
      if y1 == y2 then
	 L = math.max(0, alph2 + alph1 - C)
	 H = math.min(C, alph2 + alph1)
      else
	 L = math.max(0, alph2 - alph1)
	 H = math.min(C, C + alph2 - alph1)
      end
      -- Compute the kernel values and step size
      local k11 = kernel(dataset[i1][1], dataset[i1][1])
      local k12 = kernel(dataset[i1][1], dataset[i2][1])
      local k22 = kernel(dataset[i2][1], dataset[i2][1])
      local eta = k11 + k22 - 2*k12
      local a1 = alph1
      local a2 = alph2
      -- Check feasibility of eta (Mercer kernel)
      if eta > 0 then
	 -- Compute the new value of a2
	 a2 = alph2 + y2*(E1-E2)/eta
	 if a2 < L then
	    a2 = L
	 elseif a2 > H then
	    a2 = H
	 end
      else
	 local f1 = y1*(E1-model.b)-alph1*k11-s*alph2*k12
	 local f2 = y1*(E2-model.b)-s*alph1*k12-alph2*k22
	 local L1 = alph1 + s*(alph2 - L)
	 local H1 = alph1 + s*(alph2 - H)
	 local Lobj = L1*f1 + L*f2 + L1*L1*k11/2 + L*L*k22/2 + s*L*L1*k12
	 local Hobj = H1*f1 + H*f2 + H1*H1*k11/2 + H*H*k22/2 + s*H*H1*k12
	 -- Determine the new value of a2
	 if Lobj < Hobj - eps then
	    a2 = L
	 elseif Lobj > Hobj + eps then
	    a2 = H
	 else
	    a2 = alph2
	 end
      end
      -- Do we change alph2 enough?
      if math.abs(a2-alph2) < eps*(a2 + alph2 + eps) then
	 return 0
      end
      -- Update a1
      a1 = alph1 + s*(alph2 - a2)
      -- Update multiplier a1
      if a1 > 0 then
	 model.a[i1] = a1
	 model.x[i1] = dataset[i1][1]
	 model.y[i1] = dataset[i1][2][1]
	 -- Keep track of nbound values
	 if a1 < C and (alph1 >= C or alph1 <= 0) then
	    nbound = nbound + 1
	 elseif a1 >= C and alph1 < C and alph1 > 0 then
	    nbound = nbound - 1
	 end
      else
	 model.a[i1] = nil
	 model.x[i1] = nil
	 model.y[i1] = nil
	 if alph1 < C and alph1 > 0 then
	    nbound = nbound - 1
	 end
      end
      -- Update multiplier a2
      if a2 > 0 then
	 model.a[i2] = a2
	 model.x[i2] = dataset[i2][1]
	 model.y[i2] = dataset[i2][2][1]
	 -- Keep track of nbound values
	 if a2 < C and (alph2 >= C or alph2 <= 0) then
	    nbound = nbound + 1
	 elseif a2 >= C and alph2 < C and alph2 > 0 then
	    nbound = nbound - 1
	 end
	 -- Keep track of ecache
      else
	 model.a[i2] = nil
	 model.x[i2] = nil
	 model.y[i2] = nil
	 if alph2 < C and alph2 > 0 then
	    nbound = nbound - 1
	 end
      end
      -- Update the bias b
      local b = model.b
      local b1 = model.b - E1 - y1*(a1-alph1)*k11 - y2*(a2-alph2)*k12
      local b2 = model.b - E2 - y1*(a1-alph1)*k12 - y2*(a2-alph2)*k22
      if a1 > 0 and a1 < C then
	 model.b = b1
      elseif a2 > 0 and a2 < C then
	 model.b = b2
      elseif L ~= H then
	 model.b = (b1+b2)/2
      end
      -- Update ecache
      local kk1 = 0
      local kk2 = 0
      -- Update ecache for i1
      if a1 > 0 and a1 < C then
	 if alph1 <= 0 or alph1 >= C then
	    ecache[i1] = model:f(dataset[i1][1])[1] - y1
	 else
	    ecache[i1] = ecache[i1] + (model.b - b) + (a1-alph1)*y1*k11 + (a2-alph2)*y2*k12
	 end
      end
      -- Update cache for i2
      if a2 > 0 and a2 < C then
	 if alph2 <= 0 or alph2 >= C then
	    ecache[i2] = model:f(dataset[i2][1])[1] - y2
	 else
	    ecache[i2] = ecache[i2] + (model.b - b) + (a1-alph1)*y1*k12 + (a2-alph2)*y2*k22
	 end
      end
      -- Update cache for everybody we care about
      for k, v in pairs(model.a) do
	 if model.a[k] < C and k ~= i1 and k ~= i2 then
	    kk1 = kernel(dataset[i1][1], dataset[k][1])
	    kk2 = kernel(dataset[i2][1], dataset[k][1])
	    ecache[k] = ecache[k] + (model.b - b) + (a1-alph1)*y1*kk1 + (a2-alph2)*y2*kk2
	 end
      end
      return 1
   end
   -- Examing an example noncached version
   local function examineExample_ncache(dataset, i2, E2)
      local y2 = dataset[i2][2][1]
      local alph2 = model.a[i2] or 0
      local r2 = E2*y2
      local E1 = 0
      local i1 = 0
      if (r2 < -tol and alph2 < C) or (r2 > tol and alph2 > 0) then
	 -- Second heuristics
	 if nbound > 1 then
	    if E2 > 0 then
	       E1 = math.huge
	       for k,v in pairs(model.a) do
		  if model.a[k] < C then
		     if ecache[k] < E1 then
			E1 = ecache[k]
			i1 = k
		     end
		  end
	       end
	    else
	       E1 = -math.huge
	       for k,v in pairs(model.a) do
		  if model.a[k] < C then
		     if ecache[k] > E1 then
			E1 = ecache[k]
			i1 = k
		     end
		  end
	       end
	    end
	    if takeStep_ncache(dataset, i1, i2, E1, E2) > 0 then
	       return 1
	    end
	 end
	 -- Heuristic hierarchy
	 for i1,v in pairs(model.a) do
	    if model.a[i1] < C then
	       E1 = ecache[i1]
	       if takeStep_ncache(dataset, i1, i2, E1, E2) > 0 then
		  return 1
	       end
	    end
	 end
	 for i1 = 1, dataset:size() do
	    E1 = model:f(dataset[i1][1])[1] - dataset[i1][2][1]
	    if takeStep_ncache(dataset, i1, i2, E1, E2) > 0 then
	       return 1
	    end
	 end
      end
      return 0
   end
   -- A non-cached version of the platt's SMO algorithm (private function)
   local function train_ncache(dataset)
      nbound = 0
      ecache = torch.zeros(dataset:size())
      model.a = {}
      model.b = 0
      model.x = {}
      model.y = {}
      local numChanged = 0
      local examineAll = 1
      local maxIter = maxIterRatio * dataset:size()
      local numIter = 0
      while numIter < maxIter and (numChanged > 0 or examineAll == 1) do
	 numChanged = 0
	 -- Examine all
	 if examineAll == 1 then
	    for i2 = 1,dataset:size() do
	       numChanged = numChanged + examineExample_ncache(dataset, i2, model:f(dataset[i2][1])[1]-dataset[i2][2][1])
	    end
	 else
	    for i2, v in pairs(model.a) do
	       if model.a[i2] < C then
		  numChanged = numChanged + examineExample_ncache(dataset, i2, ecache[i2])
	       end
	    end
	 end
	 -- Loop value
	 if examineAll == 1 then
	    examineAll = 0
	 elseif numChanged == 0 then
	    examineAll = 1
	 end
	 numIter = numIter + 1
      end
      ecache = torch.zeros(1)
   end
   -- Train on a dataset
   function model:train(dataset)
      if cache == true then
	 train_cache(dataset)
      else
	 train_ncache(dataset)
      end
      return model:test(dataset)
   end
   -- Test on a dataset
   function model:test(dataset)
      -- Counter for wrong classification
      local error = 0
      for i = 1,dataset:size() do
	 -- Iterate error rate computation
	 if torch.sum(torch.ne(model:g(dataset[i][1]), dataset[i][2])) == 0 then
	    error = error*(i-1)/i
	 else
	    error = (error*i-error+1)/i
	 end
      end
      return error
   end
   -- The decision function
   function model:f(x)
      local result = torch.zeros(1)
      -- Iterate over all pairs
      for k,v in pairs(model.a) do
	 result[1] = result[1] + model.y[k]*model.a[k]*kernel(model.x[k],x)
      end
      result[1] = result[1] + model.b
      -- Return result as a tensor
      return result
   end
   -- The indicator function
   function model:g(x)
      local result = model:f(x)
      if result[1] >= 0 then
	 return torch.ones(1)
      else
	 return -torch.ones(1)
      end
   end
   -- The number of support vectors
   function model:nsv()
      local count = 0
      for k,v in pairs(model.a) do
	 count = count + 1
      end
      return count
   end
   -- Return the object
   return model
end

-- Tweaked Platt's SMO algorithm
-- C: the regularization parameter; cache: whether to cache the kernel function
-- kernel: the kernel function; tol: tolerance on violation of KKT conditions
-- eps: the eps to detect change; ratio_alpha_changed: a heuristic for stopping
-- maxIterRation: maximum number of iterations is this number time dataset size
-- maxAllIter: maximum number of iterations through all the data points.
function xsvm.tweaked(args)
   local model = {a = {}, x = {}, y = {}, b = 0}
   args = args or {}
   -- If kernel undefined, using linear kernel
   local kernel = args.kernel or function (x1,x2) return torch.dot(x1,x2) end
   -- Default C is 1
   local C = args.C or 0.05
   -- Default cache is false
   local cache = args.cache or false
   -- Default tolerance is 1e-3
   local tol = args.tol or 1e-3
   -- Default eps (round-off error on Mercer condition) is 1e-3
   local eps = args.eps or 1e-3
   -- Default ratio alpha changes is 1%
   local ratio_alpha_changes = args.ratio_alpha_changes or 0.01
   -- Maximum number of iterations is this ratio times dataset size
   local maxIterRatio = args.maxIterRatio or 20
   -- Maximum number of all iterations
   local maxAllIter = args.maxAllIter or 20
   -- Recording the number of non-zero & non-C alpha
   local nbound = 0
   -- The error cache
   local ecache = torch.zeros(1)
   -- Cache helper
   local kcache = torch.zeros(1)
   local kcflag = torch.zeros(1):byte()
   -- Initializing the kernel cache
   local function kcache_init(dataset)
      -- Allocate cache tensors
      kcache = torch.zeros(dataset:size()*(dataset:size()+1)/2)
      kcflag = torch.zeros(dataset:size()*(dataset:size()+1)/2):byte()
   end
   -- Query the cached value
   local function kcache_query(dataset, i, j)
      -- Make sure i is larger
      if j > i then i,j = j,i end
      -- The index
      local ind = i*(i-1)/2+j
      -- Test the flags and compute the kernel
      if kcflag[ind] == 0 then
	 kcache[ind] = kernel(dataset[i][1], dataset[j][1])
	 kcflag[ind] = 1
      end
      -- Return the cache
      return kcache[ind]
   end
   -- kcache utilized f function
   local function kcache_f(dataset, i)
      local result = torch.zeros(1)
      -- Iterate over all pairs
      for k,v in pairs(model.a) do
	 result[1] = result[1] + model.y[k]*model.a[k]*kcache_query(dataset, k, i)
      end
      result[1] = result[1] + model.b
      -- Return result as a tensor
      return result
   end
   -- Clean up the cache
   local function kcache_clean()
      kcache = torch.zeros(1)
      kcflag = torch.zeros(1):byte()
   end
   -- Cached helper takeStep function
   local function takeStep_cache(dataset, i1, i2, E1, E2)
      if i1 == i2 then
	 return 0
      end
      -- Allocating values
      local alph1 = model.a[i1] or 0
      local alph2 = model.a[i2] or 0
      local y1 = dataset[i1][2][1]
      local y2 = dataset[i2][2][1]
      local s = y1*y2
      local L = 0
      local H = C
      if y1 == y2 then
	 L = math.max(0, alph2 + alph1 - C)
	 H = math.min(C, alph2 + alph1)
      else
	 L = math.max(0, alph2 - alph1)
	 H = math.min(C, C + alph2 - alph1)
      end
      -- Compute the kernel values and step size
      local k11 = kcache_query(dataset,i1,i1)
      local k12 = kcache_query(dataset,i1,i2)
      local k22 = kcache_query(dataset,i2,i2)
      local eta = k11 + k22 - 2*k12
      local a1 = alph1
      local a2 = alph2
      -- Check feasibility of eta (Mercer kernel)
      if eta > 0 then
	 -- Compute the new value of a2
	 a2 = alph2 + y2*(E1-E2)/eta
	 if a2 < L then
	    a2 = L
	 elseif a2 > H then
	    a2 = H
	 end
      else
	 local f1 = y1*(E1-model.b)-alph1*k11-s*alph2*k12
	 local f2 = y1*(E2-model.b)-s*alph1*k12-alph2*k22
	 local L1 = alph1 + s*(alph2 - L)
	 local H1 = alph1 + s*(alph2 - H)
	 local Lobj = L1*f1 + L*f2 + L1*L1*k11/2 + L*L*k22/2 + s*L*L1*k12
	 local Hobj = H1*f1 + H*f2 + H1*H1*k11/2 + H*H*k22/2 + s*H*H1*k12
	 -- Determine the new value of a2
	 if Lobj < Hobj - eps then
	    a2 = L
	 elseif Lobj > Hobj + eps then
	    a2 = H
	 else
	    a2 = alph2
	 end
      end
      -- Do we change alph2 enough?
      if math.abs(a2-alph2) < eps*(a2 + alph2 + eps) then
	 return 0
      end
      -- Update a1
      a1 = alph1 + s*(alph2 - a2)
      -- Update multiplier a1
      if a1 > 0 then
	 model.a[i1] = a1
	 model.x[i1] = dataset[i1][1]
	 model.y[i1] = dataset[i1][2][1]
	 -- Keep track of nbound values
	 if a1 < C and (alph1 >= C or alph1 <= 0) then
	    nbound = nbound + 1
	 elseif a1 >= C and alph1 < C and alph1 > 0 then
	    nbound = nbound - 1
	 end
      else
	 model.a[i1] = nil
	 model.x[i1] = nil
	 model.y[i1] = nil
	 if alph1 < C and alph1 > 0 then
	    nbound = nbound - 1
	 end
      end
      -- Update multiplier a2
      if a2 > 0 then
	 model.a[i2] = a2
	 model.x[i2] = dataset[i2][1]
	 model.y[i2] = dataset[i2][2][1]
	 -- Keep track of nbound values
	 if a2 < C and (alph2 >= C or alph2 <= 0) then
	    nbound = nbound + 1
	 elseif a2 >= C and alph2 < C and alph2 > 0 then
	    nbound = nbound - 1
	 end
	 -- Keep track of ecache
      else
	 model.a[i2] = nil
	 model.x[i2] = nil
	 model.y[i2] = nil
	 if alph2 < C and alph2 > 0 then
	    nbound = nbound - 1
	 end
      end
      -- Update the bias b
      local b = model.b
      local b1 = model.b - E1 - y1*(a1-alph1)*k11 - y2*(a2-alph2)*k12
      local b2 = model.b - E2 - y1*(a1-alph1)*k12 - y2*(a2-alph2)*k22
      if a1 > 0 and a1 < C then
	 model.b = b1
      elseif a2 > 0 and a2 < C then
	 model.b = b2
      elseif L ~= H then
	 model.b = (b1+b2)/2
      end
      -- Update ecache
      local kk1 = 0
      local kk2 = 0
      -- Update ecache for i1
      if a1 > 0 and a1 < C then
	 ecache[i1] = kcache_f(dataset,i1)[1] - y1
      end
      -- Update cache for i2
      if a2 > 0 and a2 < C then
	 ecache[i2] = kcache_f(dataset,i2)[1] - y2
      end
      return 1
   end
   -- Examing an example cached version
   local function examineExample_cache(dataset, i2, E2, examineAll)
      local y2 = dataset[i2][2][1]
      local alph2 = model.a[i2] or 0
      local r2 = E2*y2
      local E1 = 0
      local i1 = 0
      if examineAll == 1 then
	 if (r2 < -tol and alph2 < C) or (r2 > tol and alph2 > 0) then
	    -- Second heuristics
	    if nbound > 1 then
	       if E2 > 0 then
		  E1 = math.huge
		  for k,v in pairs(model.a) do
		     if model.a[k] < C then
			if ecache[k] < E1 then
			   i1 = k
			end
		     end
		  end
	       else
		  E1 = -math.huge
		  for k,v in pairs(model.a) do
		     if model.a[k] < C then
			if ecache[k] > E1 then
			   i1 = k
			end
		     end
		  end
	       end
	       E1 = kcache_f(dataset,i1)[1] - dataset[i1][2][1]
	       if takeStep_cache(dataset, i1, i2, E1, E2) > 0 then
		  return 1
	       end
	    end
	    -- Heuristic hierarchy
	    for i1,v in pairs(model.a) do
	       if model.a[i1] < C then
		  E1 = kcache_f(dataset,i1)[1] - dataset[i1][2][1]
		  if takeStep_cache(dataset, i1, i2, E1, E2) > 0 then
		     return 1
		  end
	       end
	    end
	    for i1 = 1, dataset:size() do
	       E1 = kcache_f(dataset,i1)[1] - dataset[i1][2][1]
	       if takeStep_cache(dataset, i1, i2, E1, E2) > 0 then
		  return 1
	       end
	    end
	 end
      else
	 if (r2 < -tol and alph2 < C) or (r2 > tol and alph2 > 0) then
	    -- Second heuristics
	    if nbound > 1 then
	       if E2 > 0 then
		  E1 = math.huge
		  for k,v in pairs(model.a) do
		     if model.a[k] < C then
			if ecache[k] < E1 then
			   i1 = k
			end
		     end
		  end
	       else
		  E1 = -math.huge
		  for k,v in pairs(model.a) do
		     if model.a[k] < C then
			if ecache[k] > E1 then
			   i1 = k
			end
		     end
		  end
	       end
	       E1 = kcache_f(dataset,i1)[1] - dataset[i1][2][1]
	       if takeStep_cache(dataset, i1, i2, E1, E2) > 0 then
		  return 1
	       end
	    end
	 end
      end
      return 0
   end
   -- A cached version of the platt's SMO algorithm (private function)
   local function train_cache(dataset, stopChanged)
      nbound = 0
      kcache_init(dataset)
      ecache = torch.zeros(dataset:size())
      model.a = {}
      model.b = 0
      model.x = {}
      model.y = {}
      local numChanged = 0
      local examineAll = 1
      local maxIter = maxIterRatio * dataset:size()
      local numIter = 0
      local numAllIter = 0
      while numAllIter < maxAllIter and numIter < maxIter and (numChanged > stopChanged or examineAll == 1) do
	 numChanged = 0
	 -- Examine all
	 if examineAll == 1 then
	    for i2 = 1,dataset:size() do
	       numChanged = numChanged + examineExample_cache(dataset, i2, kcache_f(dataset,i2)[1]-dataset[i2][2][1], examineAll)
	    end
	    numAllIter = numAllIter + 1
	 else
	    for i2, v in pairs(model.a) do
	       if model.a[i2] < C then
		  numChanged = numChanged + examineExample_cache(dataset, i2, kcache_f(dataset,i2)[1]-dataset[i2][2][1], examineAll)
	       end
	    end
	 end
	 -- Loop value
	 if examineAll == 1 then
	    examineAll = 0
	 elseif numChanged <= stopChanged then
	    examineAll = 1
	 end
	 numIter = numIter + 1
      end
      ecache = torch.zeros(1)
      kcache_clean()
   end
   -- Non-cached helper takeStep function
   local function takeStep_ncache(dataset, i1, i2, E1, E2)
      if i1 == i2 then
	 return 0
      end
      -- Allocating values
      local alph1 = model.a[i1] or 0
      local alph2 = model.a[i2] or 0
      local y1 = dataset[i1][2][1]
      local y2 = dataset[i2][2][1]
      local s = y1*y2
      local L = 0
      local H = C
      if y1 == y2 then
	 L = math.max(0, alph2 + alph1 - C)
	 H = math.min(C, alph2 + alph1)
      else
	 L = math.max(0, alph2 - alph1)
	 H = math.min(C, C + alph2 - alph1)
      end
      -- Compute the kernel values and step size
      local k11 = kernel(dataset[i1][1], dataset[i1][1])
      local k12 = kernel(dataset[i1][1], dataset[i2][1])
      local k22 = kernel(dataset[i2][1], dataset[i2][1])
      local eta = k11 + k22 - 2*k12
      local a1 = alph1
      local a2 = alph2
      -- Check feasibility of eta (Mercer kernel)
      if eta > 0 then
	 -- Compute the new value of a2
	 a2 = alph2 + y2*(E1-E2)/eta
	 if a2 < L then
	    a2 = L
	 elseif a2 > H then
	    a2 = H
	 end
      else
	 local f1 = y1*(E1-model.b)-alph1*k11-s*alph2*k12
	 local f2 = y1*(E2-model.b)-s*alph1*k12-alph2*k22
	 local L1 = alph1 + s*(alph2 - L)
	 local H1 = alph1 + s*(alph2 - H)
	 local Lobj = L1*f1 + L*f2 + L1*L1*k11/2 + L*L*k22/2 + s*L*L1*k12
	 local Hobj = H1*f1 + H*f2 + H1*H1*k11/2 + H*H*k22/2 + s*H*H1*k12
	 -- Determine the new value of a2
	 if Lobj < Hobj - eps then
	    a2 = L
	 elseif Lobj > Hobj + eps then
	    a2 = H
	 else
	    a2 = alph2
	 end
      end
      -- Do we change alph2 enough?
      if math.abs(a2-alph2) < eps*(a2 + alph2 + eps) then
	 return 0
      end
      -- Update a1
      a1 = alph1 + s*(alph2 - a2)
      -- Update multiplier a1
      if a1 > 0 then
	 model.a[i1] = a1
	 model.x[i1] = dataset[i1][1]
	 model.y[i1] = dataset[i1][2][1]
	 -- Keep track of nbound values
	 if a1 < C and (alph1 >= C or alph1 <= 0) then
	    nbound = nbound + 1
	 elseif a1 >= C and alph1 < C and alph1 > 0 then
	    nbound = nbound - 1
	 end
      else
	 model.a[i1] = nil
	 model.x[i1] = nil
	 model.y[i1] = nil
	 if alph1 < C and alph1 > 0 then
	    nbound = nbound - 1
	 end
      end
      -- Update multiplier a2
      if a2 > 0 then
	 model.a[i2] = a2
	 model.x[i2] = dataset[i2][1]
	 model.y[i2] = dataset[i2][2][1]
	 -- Keep track of nbound values
	 if a2 < C and (alph2 >= C or alph2 <= 0) then
	    nbound = nbound + 1
	 elseif a2 >= C and alph2 < C and alph2 > 0 then
	    nbound = nbound - 1
	 end
	 -- Keep track of ecache
      else
	 model.a[i2] = nil
	 model.x[i2] = nil
	 model.y[i2] = nil
	 if alph2 < C and alph2 > 0 then
	    nbound = nbound - 1
	 end
      end
      -- Update the bias b
      local b = model.b
      local b1 = model.b - E1 - y1*(a1-alph1)*k11 - y2*(a2-alph2)*k12
      local b2 = model.b - E2 - y1*(a1-alph1)*k12 - y2*(a2-alph2)*k22
      if a1 > 0 and a1 < C then
	 model.b = b1
      elseif a2 > 0 and a2 < C then
	 model.b = b2
      elseif L ~= H then
	 model.b = (b1+b2)/2
      end
      -- Update ecache for i1
      if a1 > 0 and a1 < C then
	 ecache[i1] = model:f(dataset[i1][1])[1] - y1
      end
      -- Update cache for i2
      if a2 > 0 and a2 < C then
	 ecache[i2] = model:f(dataset[i2][1])[1] - y2
      end
      return 1
   end
   -- Examing an example noncached version
   local function examineExample_ncache(dataset, i2, E2, examineAll)
      local y2 = dataset[i2][2][1]
      local alph2 = model.a[i2] or 0
      local r2 = E2*y2
      local E1 = 0
      local i1 = 0
      if examineAll == 1 then
	 if (r2 < -tol and alph2 < C) or (r2 > tol and alph2 > 0) then
	    -- Second heuristics
	    if nbound > 1 then
	       if E2 > 0 then
		  E1 = math.huge
		  for k,v in pairs(model.a) do
		     if model.a[k] < C then
			if ecache[k] < E1 then
			   i1 = k
			end
		     end
		  end
	       else
		  E1 = -math.huge
		  for k,v in pairs(model.a) do
		     if model.a[k] < C then
			if ecache[k] > E1 then
			   i1 = k
			end
		     end
		  end
	       end
	       E1 = model:f(dataset[i1][1])[1] - dataset[i1][2][1]
	       if takeStep_ncache(dataset, i1, i2, E1, E2) > 0 then
		  return 1
	       end
	    end
	    -- Heuristic hierarchy
	    for i1,v in pairs(model.a) do
	       if model.a[i1] < C then
		  E1 = model:f(dataset[i1][1])[1] - dataset[i1][2][1]
		  if takeStep_ncache(dataset, i1, i2, E1, E2) > 0 then
		     return 1
		  end
	       end
	    end
	    for i1 = 1, dataset:size() do
	       E1 = model:f(dataset[i1][1])[1] - dataset[i1][2][1]
	       if takeStep_ncache(dataset, i1, i2, E1, E2) > 0 then
		  return 1
	       end
	    end
	 end
      else
	 if (r2 < -tol and alph2 < C) or (r2 > tol and alph2 > 0) then
	    -- Second heuristics
	    if nbound > 1 then
	       if E2 > 0 then
		  E1 = math.huge
		  for k,v in pairs(model.a) do
		     if model.a[k] < C then
			if ecache[k] < E1 then
			   i1 = k
			end
		     end
		  end
	       else
		  E1 = -math.huge
		  for k,v in pairs(model.a) do
		     if model.a[k] < C then
			if ecache[k] > E1 then
			   i1 = k
			end
		     end
		  end
	       end
	       E1 = model:f(dataset[i1][1])[1] - dataset[i1][2][1]
	       if takeStep_ncache(dataset, i1, i2, E1, E2) > 0 then
		  return 1
	       end
	    end
	 end
      end
      return 0
   end
   -- A non-cached version of the platt's SMO algorithm (private function)
   local function train_ncache(dataset, stopChanged)
      nbound = 0
      ecache = torch.zeros(dataset:size())
      model.a = {}
      model.b = 0
      model.x = {}
      model.y = {}
      local numChanged = math.huge
      local numIter = 0
      local examineAll = 1
      local maxIter = maxIterRatio * dataset:size()
      local numAllIter = 0
      while numAllIter < maxAllIter and numIter < maxIter and (numChanged > stopChanged or examineAll == 1) do
	 numChanged = 0
	 -- Examine all
	 if examineAll == 1 then
	    for i2 = 1,dataset:size() do
	       numChanged = numChanged + examineExample_ncache(dataset, i2, model:f(dataset[i2][1])[1]-dataset[i2][2][1], examineAll)
	    end
	    numAllIter = numAllIter + 1
	 else
	    for i2, v in pairs(model.a) do
	       if model.a[i2] < C then
		  numChanged = numChanged + examineExample_ncache(dataset, i2, model:f(dataset[i2][1])[1]-dataset[i2][2][1], examineAll)
	       end
	    end
	 end
	 -- Loop value
	 if examineAll == 1 then
	    examineAll = 0
	 elseif numChanged <= stopChanged then
	    examineAll = 1
	 end
	 numIter = numIter + 1
      end
      ecache = torch.zeros(1)
   end
   -- Train on a dataset
   function model:train(dataset)
      if cache == true then
	 train_cache(dataset, ratio_alpha_changes*dataset:size())
      else
	 train_ncache(dataset, ratio_alpha_changes*dataset:size())
      end
      return model:test(dataset)
   end
   -- Test on a dataset
   function model:test(dataset)
      -- Counter for wrong classification
      local error = 0
      for i = 1,dataset:size() do
	 -- Iterate error rate computation
	 if torch.sum(torch.ne(model:g(dataset[i][1]), dataset[i][2])) == 0 then
	    error = error*(i-1)/i
	 else
	    error = (error*i-error+1)/i
	 end
      end
      return error
   end
   -- The decision function
   function model:f(x)
      local result = torch.zeros(1)
      -- Iterate over all pairs
      for k,v in pairs(model.a) do
	 result[1] = result[1] + model.y[k]*model.a[k]*kernel(model.x[k],x)
      end
      result[1] = result[1] + model.b
      -- Return result as a tensor
      return result
   end
   -- The indicator function
   function model:g(x)
      local result = model:f(x)
      if result[1] >= 0 then
	 return torch.ones(1)
      else
	 return -torch.ones(1)
      end
   end
   -- The number of support vectors
   function model:nsv()
      local count = 0
      for k,v in pairs(model.a) do
	 count = count + 1
      end
      return count
   end
   -- Return the object
   return model
end

-- Vectorized Tweaked Platt's SMO algorithm
-- C: the regularization parameter
-- kernel: the kernel function; tol: tolerance on violation of KKT conditions
-- eps: the eps to detect change; ratio_alpha_changed: a heuristic for stopping
-- maxIterRatio: maximum number of iterations to stop is this number times dataset size
-- maxAllIter: maximum number of all iterations
function xsvm.vectorized(args)
   local model = {a = {}, x = {}, y = {}, b = 0}
   args = args or {}
   -- If kernel undefined, using linear kernel
   local kernel = args.kernel or function (x1,x2) return torch.dot(x1,x2) end
   -- Default C is 1
   local C = args.C or 0.05
   -- Default tolerance is 1e-3
   local tol = args.tol or 1e-3
   -- Default eps (round-off error on Mercer condition) is 1e-3
   local eps = args.eps or 1e-3
   -- Default ratio alpha changes is 1%
   local ratio_alpha_changes = args.ratio_alpha_changes or 0.01
   -- Recording the number of non-zero & non-C alpha
   local nbound = 0
   -- Maximum number of iterations is this ratio times dataset size
   local maxIterRatio = args.maxIterRatio or 20
   -- Maximum number of all iteration iterations
   local maxAllIter = args.minStabIter or 20
   -- Cache helper
   local ecache = torch.zeros(1)
   local kcache = torch.zeros(1)
   local ycache = torch.zeros(1)
   local acache = torch.zeros(1)
   -- Initializing the kernel cache
   local function kcache_init(dataset)
      -- Allocate cache tensors
      ycache = torch.zeros(dataset:size())
      acache = torch.zeros(dataset:size())
      ecache = torch.zeros(dataset:size())
      kcache = torch.zeros(dataset:size(), dataset:size())
      for i = 1, dataset:size() do
	 ycache[i] = dataset[i][2][1]
	 for j = i, dataset:size() do
	    local k = kernel(dataset[i][1],dataset[j][1])
	    kcache[i][j] = k
	    kcache[j][i] = k
	 end
      end
   end
   -- Query the cached value
   local function kcache_query(dataset, i, j)
      -- Return the cache
      return kcache[i][j]
   end
   -- kcache utilized f function
   local function kcache_f(dataset, i)
      -- Return result as a tensor
      return torch.ones(1)*(torch.dot(torch.cmul(acache,ycache), kcache[i]) + model.b);
   end
   -- Clean up the cache
   local function kcache_clean()
      kcache = torch.zeros(1)
      ycache = torch.zeros(1)
      acache = torch.zeros(1)
      ecache = torch.zeros(1)
   end
   -- Cached helper takeStep function
   local function takeStep_cache(dataset, i1, i2, E1, E2)
      if i1 == i2 then
	 return 0
      end
      -- Allocating values
      local alph1 = model.a[i1] or 0
      local alph2 = model.a[i2] or 0
      local y1 = dataset[i1][2][1]
      local y2 = dataset[i2][2][1]
      local s = y1*y2
      local L = 0
      local H = C
      if y1 == y2 then
	 L = math.max(0, alph2 + alph1 - C)
	 H = math.min(C, alph2 + alph1)
      else
	 L = math.max(0, alph2 - alph1)
	 H = math.min(C, C + alph2 - alph1)
      end
      -- Compute the kernel values and step size
      local k11 = kcache_query(dataset,i1,i1)
      local k12 = kcache_query(dataset,i1,i2)
      local k22 = kcache_query(dataset,i2,i2)
      local eta = k11 + k22 - 2*k12
      local a1 = alph1
      local a2 = alph2
      -- Check feasibility of eta (Mercer kernel)
      if eta > 0 then
	 -- Compute the new value of a2
	 a2 = alph2 + y2*(E1-E2)/eta
	 if a2 < L then
	    a2 = L
	 elseif a2 > H then
	    a2 = H
	 end
      else
	 local f1 = y1*(E1-model.b)-alph1*k11-s*alph2*k12
	 local f2 = y1*(E2-model.b)-s*alph1*k12-alph2*k22
	 local L1 = alph1 + s*(alph2 - L)
	 local H1 = alph1 + s*(alph2 - H)
	 local Lobj = L1*f1 + L*f2 + L1*L1*k11/2 + L*L*k22/2 + s*L*L1*k12
	 local Hobj = H1*f1 + H*f2 + H1*H1*k11/2 + H*H*k22/2 + s*H*H1*k12
	 -- Determine the new value of a2
	 if Lobj < Hobj - eps then
	    a2 = L
	 elseif Lobj > Hobj + eps then
	    a2 = H
	 else
	    a2 = alph2
	 end
      end
      -- Do we change alph2 enough?
      if math.abs(a2-alph2) < eps*(a2 + alph2 + eps) then
	 return 0
      end
      -- Update a1
      a1 = alph1 + s*(alph2 - a2)
      -- Update multiplier a1
      if a1 > 0 then
	 model.a[i1] = a1
	 acache[i1] = a1
	 model.x[i1] = dataset[i1][1]
	 model.y[i1] = dataset[i1][2][1]
	 -- Keep track of nbound values
	 if a1 < C and (alph1 >= C or alph1 <= 0) then
	    nbound = nbound + 1
	 elseif a1 >= C and alph1 < C and alph1 > 0 then
	    nbound = nbound - 1
	 end
      else
	 model.a[i1] = nil
	 acache[i1] = 0
	 model.x[i1] = nil
	 model.y[i1] = nil
	 if alph1 < C and alph1 > 0 then
	    nbound = nbound - 1
	 end
      end
      -- Update multiplier a2
      if a2 > 0 then
	 model.a[i2] = a2
	 acache[i2] = a2
	 model.x[i2] = dataset[i2][1]
	 model.y[i2] = dataset[i2][2][1]
	 -- Keep track of nbound values
	 if a2 < C and (alph2 >= C or alph2 <= 0) then
	    nbound = nbound + 1
	 elseif a2 >= C and alph2 < C and alph2 > 0 then
	    nbound = nbound - 1
	 end
	 -- Keep track of ecache
      else
	 model.a[i2] = nil
	 acache[i2] = 0
	 model.x[i2] = nil
	 model.y[i2] = nil
	 if alph2 < C and alph2 > 0 then
	    nbound = nbound - 1
	 end
      end
      -- Update the bias b
      local b = model.b
      local b1 = model.b - E1 - y1*(a1-alph1)*k11 - y2*(a2-alph2)*k12
      local b2 = model.b - E2 - y1*(a1-alph1)*k12 - y2*(a2-alph2)*k22
      if a1 > 0 and a1 < C then
	 model.b = b1
      elseif a2 > 0 and a2 < C then
	 model.b = b2
      elseif L ~= H then
	 model.b = (b1+b2)/2
      end
      -- Update ecache
      local kk1 = 0
      local kk2 = 0
      -- Update ecache for i1
      if a1 > 0 and a1 < C then
	 ecache[i1] = kcache_f(dataset,i1)[1] - y1
      end
      -- Update cache for i2
      if a2 > 0 and a2 < C then
	 ecache[i2] = kcache_f(dataset,i2)[1] - y2
      end
      return 1
   end
   -- Examing an example cached version
   local function examineExample_cache(dataset, i2, E2, examineAll)
      local y2 = dataset[i2][2][1]
      local alph2 = model.a[i2] or 0
      local r2 = E2*y2
      local E1 = 0
      local i1 = 0
      if examineAll == 1 then
	 -- Use full heuristic hierarchy
	 if (r2 < -tol and alph2 < C) or (r2 > tol and alph2 > 0) then
	    -- Second heuristics
	    if nbound > 1 then
	       if E2 > 0 then
		  E1 = math.huge
		  for k,v in pairs(model.a) do
		     if model.a[k] < C then
			if ecache[k] < E1 then
			   i1 = k
			end
		     end
		  end
	       else
		  E1 = -math.huge
		  for k,v in pairs(model.a) do
		     if model.a[k] < C then
			if ecache[k] > E1 then
			   i1 = k
			end
		     end
		  end
	       end
	       E1 = kcache_f(dataset,i1)[1] - dataset[i1][2][1]
	       if takeStep_cache(dataset, i1, i2, E1, E2) > 0 then
		  return 1
	       end
	    end
	    -- Heuristic hierarchy
	    for i1,v in pairs(model.a) do
	       if model.a[i1] < C then
		  E1 = kcache_f(dataset,i1)[1] - dataset[i1][2][1]
		  if takeStep_cache(dataset, i1, i2, E1, E2) > 0 then
		     return 1
		  end
	       end
	    end
	    for i1 = 1, dataset:size() do
	       E1 = kcache_f(dataset,i1)[1] - dataset[i1][2][1]
	       if takeStep_cache(dataset, i1, i2, E1, E2) > 0 then
		  return 1
	       end
	    end
	 end
      else
	 -- Using partial heuristic hierarchy concerning current support vectors only
	 if (r2 < -tol and alph2 < C) or (r2 > tol and alph2 > 0) then
	    -- Second heuristics
	    if nbound > 1 then
	       if E2 > 0 then
		  E1 = math.huge
		  for k,v in pairs(model.a) do
		     if model.a[k] < C then
			if ecache[k] < E1 then
			   i1 = k
			end
		     end
		  end
	       else
		  E1 = -math.huge
		  for k,v in pairs(model.a) do
		     if model.a[k] < C then
			if ecache[k] > E1 then
			   i1 = k
		     end
		     end
		  end
	       end
	       E1 = kcache_f(dataset,i1)[1] - dataset[i1][2][1]
	       if takeStep_cache(dataset, i1, i2, E1, E2) > 0 then
		  return 1
	       end
	    end
	 end
      end
      return 0
   end
   -- A cached version of the platt's SMO algorithm (private function)
   local function train_cache(dataset, stopChanged)
      nbound = 0
      kcache_init(dataset)
      model.a = {}
      model.b = 0
      model.x = {}
      model.y = {}
      local numChanged = math.huge
      local examineAll = 1
      local numIter = 0
      local maxIter = maxIterRatio*dataset:size()
      local numAllIter = 0
      while numAllIter < maxAllIter and numIter < maxIter and (numChanged > stopChanged or examineAll == 1) do
	 numChanged = 0
	 -- Examine all
	 if examineAll == 1 then
	    for i2 = 1,dataset:size() do
	       numChanged = numChanged + examineExample_cache(dataset, i2, kcache_f(dataset,i2)[1]-dataset[i2][2][1], examineAll)
	    end
	    numAllIter = numAllIter + 1
	 else
	    for i2, v in pairs(model.a) do
	       if model.a[i2] < C then
		  numChanged = numChanged + examineExample_cache(dataset, i2, kcache_f(dataset,i2)[1]-dataset[i2][2][1], examineAll)
	       end
	    end
	 end
	 -- Loop value
	 if examineAll == 1 then
	    examineAll = 0
	 elseif numChanged <= stopChanged then
	    examineAll = 1
	 end
	 numIter = numIter + 1
      end
      kcache_clean()
   end
   function model:train(dataset)
      train_cache(dataset, ratio_alpha_changes*dataset:size())
      return model:test(dataset)
   end
   -- Test on a dataset
   function model:test(dataset)
      -- Counter for wrong classification
      local error = 0
      for i = 1,dataset:size() do
	 -- Iterate error rate computation
	 if torch.sum(torch.ne(model:g(dataset[i][1]), dataset[i][2])) == 0 then
	    error = error*(i-1)/i
	 else
	    error = (error*i-error+1)/i
	 end
      end
      return error
   end
   -- The decision function
   function model:f(x)
      local result = torch.zeros(1)
      -- Iterate over all pairs
      for k,v in pairs(model.a) do
	 result[1] = result[1] + model.y[k]*model.a[k]*kernel(model.x[k],x)
      end
      result[1] = result[1] + model.b
      -- Return result as a tensor
      return result
   end
   -- The indicator function
   function model:g(x)
      local result = model:f(x)
      if result[1] >= 0 then
	 return torch.ones(1)
      else
	 return -torch.ones(1)
      end
   end
   -- The number of support vectors
   function model:nsv()
      local count = 0
      for k,v in pairs(model.a) do
	 count = count + 1
      end
      return count
   end
   -- Return the object
   return model
end

-- Vectorized Platt's original svm algorithm
-- C: the regularization parameter;
-- kernel: the kernel function; tol: tolerance on violation of KKT conditions
-- eps: the eps to detect change; maxIterRatio: maximum number of iterations is this number times dataset size
function xsvm.fast(args)
   local model = {a = {}, x = {}, y = {}, b = 0}
   args = args or {}
   -- If kernel undefined, using linear kernel
   local kernel = args.kernel or function (x1,x2) return torch.dot(x1,x2) end
   -- Default C is 1
   local C = args.C or 0.05
   -- Default cache is false
   local cache = args.cache or false
   -- Default tolerance is 1e-3
   local tol = args.tol or 1e-3
   -- Default eps (round-off error on Mercer condition) is 1e-3
   local eps = args.eps or 1e-3
   -- Maximum number of iterations is this ratio times dataset size
   local maxIterRatio = args.maxIterRatio or 20
   -- Recording the number of non-zero & non-C alpha
   local nbound = 0
   -- Cache helper
   local ecache = torch.zeros(1)
   local kcache = torch.zeros(1)
   local ycache = torch.zeros(1)
   local acache = torch.zeros(1)
   -- Initializing the kernel cache
   local function kcache_init(dataset)
      -- Allocate cache tensors
      ycache = torch.zeros(dataset:size())
      acache = torch.zeros(dataset:size())
      ecache = torch.zeros(dataset:size())
      kcache = torch.zeros(dataset:size(), dataset:size())
      for i = 1, dataset:size() do
	 ycache[i] = dataset[i][2][1]
	 for j = i, dataset:size() do
	    local k = kernel(dataset[i][1],dataset[j][1])
	    kcache[i][j] = k
	    kcache[j][i] = k
	 end
      end
   end
   -- Query the cached value
   local function kcache_query(dataset, i, j)
      -- Return the cache
      return kcache[i][j]
   end
   -- kcache utilized f function
   local function kcache_f(dataset, i)
      -- Return result as a tensor
      return torch.ones(1)*(torch.dot(torch.cmul(acache,ycache), kcache[i]) + model.b);
   end
   -- Clean up the cache
   local function kcache_clean()
      kcache = torch.zeros(1)
      ycache = torch.zeros(1)
      acache = torch.zeros(1)
      ecache = torch.zeros(1)
   end
   -- Cached helper takeStep function
   local function takeStep_cache(dataset, i1, i2, E1, E2)
      if i1 == i2 then
	 return 0
      end
      -- Allocating values
      local alph1 = model.a[i1] or 0
      local alph2 = model.a[i2] or 0
      local y1 = dataset[i1][2][1]
      local y2 = dataset[i2][2][1]
      local s = y1*y2
      local L = 0
      local H = C
      if y1 == y2 then
	 L = math.max(0, alph2 + alph1 - C)
	 H = math.min(C, alph2 + alph1)
      else
	 L = math.max(0, alph2 - alph1)
	 H = math.min(C, C + alph2 - alph1)
      end
      -- Compute the kernel values and step size
      local k11 = kcache_query(dataset,i1,i1)
      local k12 = kcache_query(dataset,i1,i2)
      local k22 = kcache_query(dataset,i2,i2)
      local eta = k11 + k22 - 2*k12
      local a1 = alph1
      local a2 = alph2
      -- Check feasibility of eta (Mercer kernel)
      if eta > 0 then
	 -- Compute the new value of a2
	 a2 = alph2 + y2*(E1-E2)/eta
	 if a2 < L then
	    a2 = L
	 elseif a2 > H then
	    a2 = H
	 end
      else
	 local f1 = y1*(E1-model.b)-alph1*k11-s*alph2*k12
	 local f2 = y1*(E2-model.b)-s*alph1*k12-alph2*k22
	 local L1 = alph1 + s*(alph2 - L)
	 local H1 = alph1 + s*(alph2 - H)
	 local Lobj = L1*f1 + L*f2 + L1*L1*k11/2 + L*L*k22/2 + s*L*L1*k12
	 local Hobj = H1*f1 + H*f2 + H1*H1*k11/2 + H*H*k22/2 + s*H*H1*k12
	 -- Determine the new value of a2
	 if Lobj < Hobj - eps then
	    a2 = L
	 elseif Lobj > Hobj + eps then
	    a2 = H
	 else
	    a2 = alph2
	 end
      end
      -- Do we change alph2 enough?
      if math.abs(a2-alph2) < eps*(a2 + alph2 + eps) then
	 return 0
      end
      -- Update a1
      a1 = alph1 + s*(alph2 - a2)
      -- Update multiplier a1
      if a1 > 0 then
	 model.a[i1] = a1
	 acache[i1] = a1
	 model.x[i1] = dataset[i1][1]
	 model.y[i1] = dataset[i1][2][1]
	 -- Keep track of nbound values
	 if a1 < C and (alph1 >= C or alph1 <= 0) then
	    nbound = nbound + 1
	 elseif a1 >= C and alph1 < C and alph1 > 0 then
	    nbound = nbound - 1
	 end
      else
	 model.a[i1] = nil
	 acache[i1] = 0
	 model.x[i1] = nil
	 model.y[i1] = nil
	 if alph1 < C and alph1 > 0 then
	    nbound = nbound - 1
	 end
      end
      -- Update multiplier a2
      if a2 > 0 then
	 model.a[i2] = a2
	 acache[i2] = a2
	 model.x[i2] = dataset[i2][1]
	 model.y[i2] = dataset[i2][2][1]
	 -- Keep track of nbound values
	 if a2 < C and (alph2 >= C or alph2 <= 0) then
	    nbound = nbound + 1
	 elseif a2 >= C and alph2 < C and alph2 > 0 then
	    nbound = nbound - 1
	 end
	 -- Keep track of ecache
      else
	 model.a[i2] = nil
	 acache[i2] = 0
	 model.x[i2] = nil
	 model.y[i2] = nil
	 if alph2 < C and alph2 > 0 then
	    nbound = nbound - 1
	 end
      end
      -- Update the bias b
      local b = model.b
      local b1 = model.b - E1 - y1*(a1-alph1)*k11 - y2*(a2-alph2)*k12
      local b2 = model.b - E2 - y1*(a1-alph1)*k12 - y2*(a2-alph2)*k22
      if a1 > 0 and a1 < C then
	 model.b = b1
      elseif a2 > 0 and a2 < C then
	 model.b = b2
      elseif L ~= H then
	 model.b = (b1+b2)/2
      end
      -- Update ecache
      local kk1 = 0
      local kk2 = 0
      -- Update ecache for i1
      if a1 > 0 and a1 < C then
	 if alph1 <= 0 or alph1 >= C then
	    ecache[i1] = kcache_f(dataset,i1)[1] - y1
	 else
	    ecache[i1] = ecache[i1] + (model.b - b) + (a1-alph1)*y1*k11 + (a2-alph2)*y2*k12
	 end
      end
      -- Update cache for i2
      if a2 > 0 and a2 < C then
	 if alph2 <= 0 or alph2 >= C then
	    ecache[i2] = kcache_f(dataset,i2)[1] - y2
	 else
	    ecache[i2] = ecache[i2] + (model.b - b) + (a1-alph1)*y1*k12 + (a2-alph2)*y2*k22
	 end
      end
      -- Update cache for everybody we care about
      for k, v in pairs(model.a) do
	 if model.a[k] < C and k ~= i1 and k ~= i2 then
	    kk1 = kcache_query(dataset,i1, k)
	    kk2 = kcache_query(dataset,i2, k)
	    ecache[k] = ecache[k] + (model.b - b) + (a1-alph1)*y1*kk1 + (a2-alph2)*y2*kk2
	 end
      end
      return 1
   end
   -- Examing an example cached version
   local function examineExample_cache(dataset, i2, E2)
      local y2 = dataset[i2][2][1]
      local alph2 = model.a[i2] or 0
      local r2 = E2*y2
      local E1 = 0
      local i1 = 0
      if (r2 < -tol and alph2 < C) or (r2 > tol and alph2 > 0) then
	 -- Second heuristics
	 if nbound > 1 then
	    if E2 > 0 then
	       E1 = math.huge
	       for k,v in pairs(model.a) do
		  if model.a[k] < C then
		     if ecache[k] < E1 then
			E1 = ecache[k]
			i1 = k
		     end
		  end
	       end
	    else
	       E1 = -math.huge
	       for k,v in pairs(model.a) do
		  if model.a[k] < C then
		     if ecache[k] > E1 then
			E1 = ecache[k]
			i1 = k
		     end
		  end
	       end
	    end
	    if takeStep_cache(dataset, i1, i2, E1, E2) > 0 then
	       return 1
	    end
	 end
	 -- Heuristic hierarchy
	 for i1,v in pairs(model.a) do
	    if model.a[i1] < C then
	       E1 = ecache[i1]
	       if takeStep_cache(dataset, i1, i2, E1, E2) > 0 then
		  return 1
	       end
	    end
	 end
	 for i1 = 1, dataset:size() do
	    E1 = kcache_f(dataset,i1)[1] - dataset[i1][2][1]
	    if takeStep_cache(dataset, i1, i2, E1, E2) > 0 then
	       return 1
	    end
	 end
      end
      return 0
   end
   -- A cached version of the platt's SMO algorithm (private function)
   local function train_cache(dataset)
      nbound = 0
      kcache_init(dataset)
      model.a = {}
      model.b = 0
      model.x = {}
      model.y = {}
      local numChanged = 0
      local examineAll = 1
      local numIter = 0
      local maxIter = maxIterRatio*dataset:size()
      while numIter < maxIter and (numChanged > 0 or examineAll == 1) do
	 numChanged = 0
	 -- Examine all
	 if examineAll == 1 then
	    for i2 = 1,dataset:size() do
	       numChanged = numChanged + examineExample_cache(dataset, i2, kcache_f(dataset,i2)[1]-dataset[i2][2][1])
	    end
	 else
	    for i2, v in pairs(model.a) do
	       if model.a[i2] < C then
		  numChanged = numChanged + examineExample_cache(dataset, i2, ecache[i2])
	       end
	    end
	 end
	 -- Loop value
	 if examineAll == 1 then
	    examineAll = 0
	 elseif numChanged == 0 then
	    examineAll = 1
	 end
	 numIter = numIter + 1
      end
      kcache_clean()
   end
   function model:train(dataset)
      train_cache(dataset)
      return model:test(dataset)
   end
   -- Test on a dataset
   function model:test(dataset)
      -- Counter for wrong classification
      local error = 0
      for i = 1,dataset:size() do
	 -- Iterate error rate computation
	 if torch.sum(torch.ne(model:g(dataset[i][1]), dataset[i][2])) == 0 then
	    error = error*(i-1)/i
	 else
	    error = (error*i-error+1)/i
	 end
      end
      return error
   end
   -- The decision function
   function model:f(x)
      local result = torch.zeros(1)
      -- Iterate over all pairs
      for k,v in pairs(model.a) do
	 result[1] = result[1] + model.y[k]*model.a[k]*kernel(model.x[k],x)
      end
      result[1] = result[1] + model.b
      -- Return result as a tensor
      return result
   end
   -- The indicator function
   function model:g(x)
      local result = model:f(x)
      if result[1] >= 0 then
	 return torch.ones(1)
      else
	 return -torch.ones(1)
      end
   end
   -- The number of support vectors
   function model:nsv()
      local count = 0
      for k,v in pairs(model.a) do
	 count = count + 1
      end
      return count
   end
   -- Return the object
   return model
end
