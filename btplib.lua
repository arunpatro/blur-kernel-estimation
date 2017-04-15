--------------------------------------
-- Optim Library, courtesy: A.Karpathy
--------------------------------------

function sgd(x, dx, lr)
  x:add(-lr, dx)
end

function sgdm(x, dx, lr, alpha, state)
  -- sgd with momentum, standard update
  if not state.v then
    state.v = x.new(#x):zero()
  end
  state.v:mul(alpha)
  state.v:add(lr, dx)
  x:add(-1, state.v)
end

function sgdmom(x, dx, lr, alpha, state)
  -- sgd momentum, uses nesterov update (reference: http://cs231n.github.io/neural-networks-3/#sgd)
  if not state.m then
    state.m = x.new(#x):zero()
    state.tmp = x.new(#x)
  end
  state.tmp:copy(state.m)
  state.m:mul(alpha):add(-lr, dx)
  x:add(-alpha, state.tmp)
  x:add(1+alpha, state.m)
end

function adagrad(x, dx, lr, epsilon, state)
  if not state.m then
    state.m = x.new(#x):zero()
    state.tmp = x.new(#x)
  end
  -- calculate new mean squared values
  state.m:addcmul(1.0, dx, dx)
  -- perform update
  state.tmp:sqrt(state.m):add(epsilon)
  x:addcdiv(-lr, dx, state.tmp)
end

-- rmsprop implementation, simple as it should be
function rmsprop(x, dx, lr, alpha, epsilon, state)
  if not state.m then
    state.m = x.new(#x):zero()
    state.tmp = x.new(#x)
  end
  -- calculate new (leaky) mean squared values
  state.m:mul(alpha)
  state.m:addcmul(1.0-alpha, dx, dx)
  -- perform update
  state.tmp:sqrt(state.m):add(epsilon)
  x:addcdiv(-lr, dx, state.tmp)
end

function adam(x, dx, lr, beta1, beta2, epsilon, state)
  local beta1 = beta1 or 0.9
  local beta2 = beta2 or 0.999
  local epsilon = epsilon or 1e-8

  if not state.m then
    -- Initialization
    state.t = 0
    -- Exponential moving average of gradient values
    state.m = x.new(#dx):zero()
    -- Exponential moving average of squared gradient values
    state.v = x.new(#dx):zero()
    -- A tmp tensor to hold the sqrt(v) + epsilon
    state.tmp = x.new(#dx):zero()
  end

  -- Decay the first and second moment running average coefficient
  state.m:mul(beta1):add(1-beta1, dx)
  state.v:mul(beta2):addcmul(1-beta2, dx, dx)
  state.tmp:copy(state.v):sqrt():add(epsilon)

  state.t = state.t + 1
  local biasCorrection1 = 1 - beta1^state.t
  local biasCorrection2 = 1 - beta2^state.t
  local stepSize = lr * math.sqrt(biasCorrection2)/biasCorrection1
  
  -- perform update
  x:addcdiv(-stepSize, state.m, state.tmp)
end


-------------------------
-- Performance Evaluators
-------------------------


function getMean(inputs,targets,model)
    diff = torch.abs(model:forward(inputs) - targets)
    mean = diff:mean()
    return mean
end

function performanceEvaluator(trainset,model,fcnFlag,cudaFlag,fcnFlag,eList)
  items = eList or torch.Tensor({8,14,32,49,53,58,65,68,69,70})
  length = (#items)[1]
  means = torch.Tensor(length)
  iSize = fcnFlag and 32 or 1;
  for i=1,length do
    if cudaFlag then
      inputs = trainset.images[items[i]]:cuda()
      targets = torch.Tensor(1,iSize,iSize):fill(trainset.labels[items[i]]*0.3):cuda()
    else
      inputs = trainset.images[items[i]]
      targets = torch.Tensor(1,iSize,iSize):fill(trainset.labels[items[i]]*0.3)
    end
    means[i] = (getMean(inputs,targets,model)*100)/(trainset.labels[items[i]]*0.3)
    print('Image '.. string.format('%3d',items[i])..' | Label '.. string.format('%2d',trainset.labels[items[i]])  .. ' | Sigma '..string.format('%2.1f', trainset.labels[items[i]]*0.3) ..' | Pred ' ..string.format('%1.8f',model:forward(inputs)[{1,1,1}]) .. ' | MSE ' .. string.format('%1.8f',criterion:forward(model:forward(inputs),targets)) .. ' | RMSE ' .. string.format('%1.8f',torch.sqrt(criterion:forward(model:forward(inputs),targets))) .. ' | Percent Error ' .. string.format('%3.2f',means[i]) .. '%')
  end
  print('Means of means: ' .. means:mean())
end


function classPerformanceEvaluator(trainset,model,fcnFlag,eList)
  items = eList or torch.Tensor({8,14,32,49,53,58,65,68,69,70}) 
  length = (#items)[1]
  acc = torch.Tensor(10):fill(0)
  hist = torch.Tensor(10):fill(0)
  iSize = fcnFlag and 32 or 1;
  for i=1,length do
    inputs = trainset.images[items[i]]:cuda()
    targets = torch.Tensor(1,iSize,iSize):fill(trainset.labels[items[i]]*0.3):cuda()
    acc[trainset.labels[items[i]]] = acc[trainset.labels[items[i]]]+ (getMean(inputs,targets,model)*100)/(trainset.labels[items[i]]*0.3)
    hist[trainset.labels[items[i]]] = hist[trainset.labels[items[i]]] + 1
  end
  print(torch.cdiv(acc,hist))
  print(torch.cdiv(acc,hist):mean())
end

function randomEvaluator(dataSize,evalSize)
  local list = torch.randperm(dataSize)[{{1,evalSize}}]
  performanceEvaluator(trainset,model,true,true,list)
end

function testClass(dataset,model,cudaFlag,bSize,size)
  print('<trainer> on testing Set:')
  for t = 1,size,bSize do
    xlua.progress(t, dataset:size())
    local inputs = dataset.images[{{t, math.min(t+bSize-1,size)}}]
    if cudaFlag then
      inputs = inputs:cuda()
    end
    local targets = dataset.labels[{{t, math.min(t+bSize-1,size)}}]
    local preds = model:forward(inputs)
    for i = 1,bSize do
      confusion:add(preds[i], targets[i])
    end
  end
  print(confusion)
  print('\27[31mTest: ' .. confusion.totalValid * 100)
  testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
  confusion:zero()
end
-------------------------
-- Trainers
-------------------------

function trainerSingle(trainset,model,lr,item,cudaFlag)
  params,grad_params = model:getParameters();
  iSize = fcnFlag and 32 or 1;
  inputs = trainset.images[item]
  targets = torch.Tensor(1,iSize,iSize):fill(trainset.labels[item]*0.3)
  if cudaFlag then
    inputs = inputs:cuda()
    targets = targets:cuda()
  end
  grad_params:zero();
  outputs = model:forward(inputs);
  currentError = currentError + criterion:forward(outputs, targets);
  df_do = criterion:backward(outputs, targets);
  model:backward(inputs, df_do);
  -- sgd(params,grad_params,lr)
  -- rmsprop(params,grad_params,lr,0.99,1e-8,config)
  -- adam(params,grad_params,lr,0.9,0.999,1e-7,config)
  adagrad(params,grad_params,lr,1e-7,config)
end

function trainerBatch(dataset, model, lr, bSize, size, cudaFlag, classFlag, fcnFlag)
  print('Training with batch size ' .. bSize .. ' and learning rate ' .. lr .. ' and size ' .. size)
  local params,grad_params = model:getParameters();
  local iSize = fcnFlag and 32 or 1;
  local set = {
    images = torch.Tensor(size,1,32,32),
    labels = torch.Tensor(size):byte()
  }
  local p = torch.randperm(size);
  for i = 1,size do
    set.images[i] = dataset.images[p[i]]
    set.labels[i] = dataset.labels[p[i]]
  end
  for t = 1,size,bSize do
    grad_params:zero();
    local inputs = set.images[{{t, math.min(t+bSize-1,size)}}]

    if classFlag then
      targets = set.labels[{{t, math.min(t+bSize-1,size)}}]
    else
      local targets = torch.Tensor(math.min(t+bSize-1,size)-t+1,1,iSize,iSize)
      for i=t,math.min(t+bSize-1,size) do
        targets[i-t+1] = targets[i-t+1]:fill(trainset.labels[i-t+1])
      end
    end

    if cudaFlag then
      inputs = inputs:cuda();
      targets = classFlag and targets or targets:cuda();
      criterion = criterion:cuda()
    end

    local outputs = model:forward(inputs);

    if classFlag then
      for i = 1,bSize do
         confusion:add(outputs[i], targets[i])
      end
    end

    local f = criterion:forward(outputs, targets);
    local df_do = criterion:backward(outputs, targets);
    model:backward(inputs, df_do);
    adagrad(params,grad_params,lr,1e-8,config)
    -- sgd(params,grad_params,lr)
    currentError = currentError + f
    xlua.progress(t,size)
  end
  if classFlag then
    print(confusion)
    print('\27[32mTrain: ' .. confusion.totalValid * 100)
    trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
    confusion:zero()
  end
end
