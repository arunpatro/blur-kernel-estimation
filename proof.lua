-------------------------
-- Proof of Concept that we can train 10 different sigma classes images and predict 
-- a dense sigma map. Here we take 10 distinct classes of the testset (small to load
-- quickly). The FCNN is 6 layers deep. Here at the end we evaluate the mean error 
-- of the classes, in absolute mean and MSE error per pixel of the image. We display 
-- in percentage points of pixel prediction as the value to output is quite less. 
-------------------------
require 'nn'
require 'torch'
require 'xlua'
require 'math'
require 'optim'
require 'optim_updates'
require 'io'
require 'functions'

lenet = torch.load('lenet-proof.t7')
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
    rmsprop(params,grad_params,lr,0.99,1e-8,config)
    currentError = currentError + f
    xlua.progress(t,size)
  end
end

config = {
    learningRate = 0.0001,
    learningRateDecay = 1e-7
}

logger = optim.Logger('./proof.log')

items = {8,14,32,49,53,58,65,68,69,70}

-- Run for a lot of epochs
for k=1,10 do
    for j=1,100 do
      currentError = 0
      for i=1,10 do
        trainer(0.0001,items[i])
      end
      xlua.progress((k-1)*100+j,1000)
      logger:add{['error'] = currentError/10}
      print('\ntotal: ' .. currentError .. ' perunit: ' ..currentError/10)
    end
    torch.save('lenet-proof.t7',lenet)
end

-- Check stats for each class 
performanceEvaluator(trainset,lenet,items)

-- trainerBatch(0.0001,50,1000)
logger:plot()
