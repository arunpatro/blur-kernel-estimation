require 'nn'
require 'torch'
require 'xlua'
require 'math'
require 'optim'
require 'optim_updates'

lenet = torch.load('lenet-test.t7')
trainset = torch.load('testSig.t7')

function trainset:size()
  return self.images:size(1)
end

-- --preprocess data
-- mean = {} -- store the mean, to normalize the test set in the future
-- stdv  = {} -- store the standard-deviation for the future
-- mean = trainset.images[{ {}, {1}, {}, {}  }]:mean() -- mean estimation
-- trainset.images[{ {}, {1}, {}, {}  }]:add(-mean) -- mean subtraction

-- stdv = trainset.images[{ {}, {1}, {}, {}  }]:std() -- std estimation
-- trainset.images[{ {}, {1}, {}, {}  }]:div(stdv) -- std scaling

criterion = nn.MSECriterion()

function stoch(lr,item)
  params,grad_params = lenet:getParameters();
  inputs = trainset.images[item]
  targets = torch.Tensor(1,32,32):fill(trainset.labels[item]*0.3)
  grad_params:zero();
  outputs = lenet:forward(inputs);
  f = criterion:forward(outputs, targets);
  df_do = criterion:backward(outputs, targets);
  lenet:backward(inputs, df_do);
  params:add(-lr,grad_params);
  -- sgd(params,grad_params,lr)
  -- adam(params,grad_params,lr)
  -- adagrad(params,grad_params,lr,1e-7)
  currentError = currentError + f
end

function stochBatch(lr, bSize)
  print('Training with batch size ' .. bSize .. ' and learning rate ' .. lr)
  params,grad_params = lenet:getParameters();
  for t = 1,size,bSize do
    grad_params:zero();
    inputs = trainset.images[{{t, math.min(t+bSize-1,size)}}]
    targets = torch.Tensor(math.min(t+bSize-1,size)-t+1,1,32,32)
    for i=t,math.min(t+bSize-1,size) do
      targets[i-t+1] = targets[i-t+1]:fill(trainset.labels[i-t+1]*0.3)
    end
    outputs = lenet:forward(inputs);
    f = criterion:forward(outputs, targets);
    df_do = criterion:backward(outputs, targets);
    lenet:backward(inputs, df_do);
    params:add(-lr,grad_params);
    currentError = currentError + f
    xlua.progress(t,size)
  end
end

parameters,gradParameters = lenet:getParameters()

opt = {learningRate = 0.05,
batchSize = 50,
plot = false}

config = {
    learningRate = opt.learningRate,
    learningRateDecay = 1e-7
}

function adagrad(size)
   print('Training with batch size ' .. opt.batchSize .. ' and learning rate ' .. opt.learningRate)
   for t = 1,size,opt.batchSize do
        inputs = trainset.images[{{t, math.min(t+opt.batchSize-1,size)}}]
        targets = torch.Tensor(math.min(t+opt.batchSize-1,size)-t+1,1,32,32)
        for i=t,math.min(t+opt.batchSize-1,size) do
          targets[i-t+1] = targets[i-t+1]:fill(trainset.labels[i-t+1]*0.3)
        end
      local feval = function(x)
         collectgarbage()
         if x ~= parameters then
            parameters:copy(x)
         end
         gradParameters:zero()
         local outputs = lenet:forward(inputs)
         local f = criterion:forward(outputs, targets)
         local df_do = criterion:backward(outputs, targets)
         lenet:backward(inputs, df_do)
         currentError = currentError + f
         return f,gradParameters
      end
         optim.adagrad(feval, parameters, config)
         xlua.progress(t, size)
   end
end


function meaner(inputs,targets,lenet)
    diff = torch.abs(lenet:forward(inputs) - targets)
    mean = diff:mean()
    return mean
end

-- lenet:reset()
size = 100

-- Stochastic single
-- for j=1,4 do
--   currentError = 0
--   for i=1,size do
--     stoch(0.0005,i)
--     xlua.progress(i,size)
--   end
--   print('\ntotal: ' .. currentError .. ' perunit: ' ..currentError/size)
-- end



-- Stochastic single
items = {8,14,32,49,53,58,65,68,69,70}

for j=1,100 do
  currentError = 0
  for i=1,10 do
    stoch(0.01,items[i])
    xlua.progress(i,10)
  end
  print('\ntotal: ' .. currentError .. ' perunit: ' ..currentError/size)
  torch.save('lenet-test.t7',lenet)
end

-- -- Stochastic Batch Wise
-- for i=1,5000 do
--   currentError = 0
--   -- stochBatch(0.0005,50)
--   adagrad(size)
--   print('\ntotal: ' .. currentError .. ' perunit: ' ..currentError/size)
-- end

for item = 11,20 do
  inputs = trainset.images[item]
  targets = torch.Tensor(1,32,32):fill(trainset.labels[item]*0.3)
  print(meaner(inputs,targets,lenet)*100/(trainset.labels[item]*0.3))
  print('MSE Error for Image '.. item..' with label '.. trainset.labels[item]  .. ' and sigma value '..string.format('%2.1f', trainset.labels[item]*0.3) ..': ' .. criterion:forward(lenet:forward(inputs),targets))
end

-- see if a particular image is training, yes its happening
for i=1,100 do
  stoch(0.005,18)
  outputs = lenet:forward(inputs)
  print(criterion:forward(targets,outputs))
end

