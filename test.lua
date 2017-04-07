require 'nn'
require 'torch'
require 'xlua'
require 'math'
require 'optim'
require 'optim_updates'
require 'functions'

lenet = torch.load('lenet-test.t7')
trainset = torch.load('testSig.t7')

function trainset:size()
  return self.images:size(1)
end

--no preprocessing

criterion = nn.MSECriterion()

function trainer(lr,item)
  params,grad_params = lenet:getParameters();
  inputs = trainset.images[item]
  targets = torch.Tensor(1,32,32):fill(trainset.labels[item]*0.3)
  grad_params:zero();
  outputs = lenet:forward(inputs);
  f = criterion:forward(outputs, targets);
  df_do = criterion:backward(outputs, targets);
  lenet:backward(inputs, df_do);
  -- params:add(-lr,grad_params);
  -- sgd(params,grad_params,lr)
  -- rmsprop(params,grad_params,lr,0.99,1e-8,config)
  -- adam(params,grad_params,lr,0.9,0.999,1e-7,config)
  adagrad(params,grad_params,lr,1e-7,config)
  currentError = currentError + f
end

function trainerBatch(lr, bSize, size)
  print('Training with batch size ' .. bSize .. ' and learning rate ' .. lr .. ' and size ' .. size)
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
    rmsprop(params,grad_params,lr,0.99,1e-8,config)
    currentError = currentError + f
    xlua.progress(t,size)
  end
end

config = {
    learningRate = 0.0001,
    learningRateDecay = 1e-7
}

logger = optim.Logger('./test.log')

items = {8,14,32,49,53,58,65,68,69,70}

-- Run for a lot of epochs
-- for k=1,10 do
--     for j=1,100 do
--       currentError = 0
--       for i=1,10 do
--         trainer(0.0001,items[i])
--       end
--       xlua.progress((k-1)*100+j,1000)
--       logger:add{['error'] = currentError/10}
--       print('\ntotal: ' .. currentError .. ' perunit: ' ..currentError/10)
--     end
--     torch.save('lenet-proof.t7',lenet)
-- end

-- -- Check stats for each class 
-- performanceEvaluator(trainset,lenet,items)

-- trainerBatch(0.0001,50,1000)
bSize = 50
size = 960
for i=1,100 do
  currentError = 0
  trainerBatch(0.0001,bSize,size)
  print('\nPer pixel MSE: ' ..currentError*bSize/size)
  logger:add{['error'] = currentError*bSize/size}
end
torch.save('lenet-test.t7',lenet)

logger:plot()
-- performanceEvaluator(trainset,lenet,torch.range(1,size))
classPerformance(trainset,lenet,torch.range(1,size),false)


