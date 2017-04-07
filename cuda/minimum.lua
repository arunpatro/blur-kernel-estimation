-------------------------
-- Instead of outputting a class, we are outputting an actual value that
-- we attempt to determine using regression. MSE criterion.
-------------------------
require 'nn'
require 'torch'
require 'xlua'
require 'math'
require 'optim'
require 'btplib'
require 'cudnn'
require 'cunn'
require 'cutorch'

lenet = torch.load('lenet-one.t7'):cuda()
trainset = torch.load('testSig.t7')

function trainset:size()
  return self.images:size(1)
end

criterion = nn.MSECriterion():cuda()

logger = optim.Logger('./test.log')

-- Run for a lot of epochs
-- size = 1000
-- for k=1,5 do
--     for j=1,10 do
--       currentError = 0
--       for i=1,size do
--         trainerSingle(trainset,lenet,0.001,i)
--         xlua.progress(i,size)
--       end
--       logger:add{['error'] = currentError/size}
--       print('\nPer pixel MSE: ' ..currentError/size)
--     end
--     torch.save('lenet-one.t7',lenet)
-- end
-- -- Check stats for each class 
-- performanceEvaluator(trainset,lenet,items)

bSize = 100
size = 1000
jMax = 2 -- Large Epochs
iMax = 300 -- Small Epochs after which to save
for j=1,jMax do
  for i=1,iMax do
    currentError = 0
    trainerBatch(trainset,lenet,0.001,bSize,size,true)
    print('\nEpoch: ' .. (j-1)*iMax+i..' Per pixel MSE: ' ..currentError*bSize/size)
    logger:add{['error'] = currentError*bSize/size}
  end
  torch.save('lenet-one.t7',lenet)
end

-- logger:plot()
-- performanceEvaluator(trainset,lenet,true)
classPerformanceEvaluator(trainset,lenet,torch.range(1,size),true)
randomEvaluator(size,10)
