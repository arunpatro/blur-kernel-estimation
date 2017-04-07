-------------------------
-- Instead of outputting a class, we are outputting an actual value that
-- we attempt to determine using regression. MSE criterion.
-------------------------
require 'nn'
require 'torch'
require 'xlua'
require 'math'
require 'optim'
require 'optim_updates'
require 'functions'

lenet = torch.load('lenet-one.t7')
trainset = torch.load('testSig.t7')

function trainset:size()
  return self.images:size(1)
end

criterion = nn.MSECriterion()

function trainerBatch(lr, bSize, size)
  print('Training with batch size ' .. bSize .. ' and learning rate ' .. lr .. ' and size ' .. size)
  params,grad_params = lenet:getParameters();
  for t = 1,size,bSize do
    grad_params:zero();
    inputs = trainset.images[{{t, math.min(t+bSize-1,size)}}]
    targets = torch.Tensor(math.min(t+bSize-1,size)-t+1,1,1,1)
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
-- size = 1000
-- for k=1,5 do
--     for j=1,2 do
--       currentError = 0
--       for i=1,size do
--         trainSingle(trainset,lenet,0.0001,i)
--         xlua.progress(i,size)
--       end
--       logger:add{['error'] = currentError/size}
--       print('\ntotal: ' .. currentError .. ' perunit: ' ..currentError/size)
--     end
--     torch.save('lenet-one.t7',lenet)
-- end

-- -- Check stats for each class 
-- performanceEvaluator(trainset,lenet,items)

bSize = 50
size = 96000
for i=1,3 do
  currentError = 0
  trainerBatch(0.001,bSize,size)
  print('\nPer pixel MSE: ' ..currentError*bSize/size)
  logger:add{['error'] = currentError*bSize/size}
end
torch.save('lenet-one.t7',lenet)

logger:plot()
-- performanceEvaluator(trainset,lenet,torch.range(1,size))
classPerformance(trainset,lenet,torch.range(1,size),true)


