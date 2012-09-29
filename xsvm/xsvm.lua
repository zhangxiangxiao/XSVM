--[[
Xiang's Kernel SVM :)
By Xiang Zhang @ New York University
09/29/2012, Version 0.1

Note: The kernel function you give must be positive
definite symmetric (PDS kernel).
]]

-- The namespace
xsvm = {}

-- Create a new xsvm trainer with kernel
-- kernel: the kernel function (callable); C: regularization parameter
-- tol: Numerical tolerance; maxi_passes: max number of times to iterate over a's without changing
-- ratio_alpha_changes: a ratio heuristic for stopping the computation.
-- cache: whether to cache the kernel
function xsvm:simple(kernel, C, tol, max_passes, ratio_alpha_changes, cache)
   model = {a = {}, x = {}, y = {}, b = 0}
   -- If kernel undefined, using linear kernel
   kernel = kernel or function (x1,x2) return torch.dot(x1,x2) end
   -- Default C is 1
   C = C or 0.05
   -- Default tolerance is 1e-3
   tol = tol or 1e-3
   -- Default max_passes is 2
   max_passes = max_passes or 3
   -- Default cache is none
   cache = cache or false
   -- Default ratio alpha changes is 2%
   ratio_alpha_changes = ratio_alpha_changes or 0.01
   -- Train on a dataset
   function model:train(dataset)
      train_ncache(dataset)
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
   -- A non-cached version of the SMO algorithm (private function)
   function train_ncache(dataset)
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
		   if a[j] > 0 and a[j] < C then
		      model.b = b1
		   elseif a[i] > 0 and a[i] < C then
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
      return table.getn(model.a)
   end
   -- Return the object
   return model
end
