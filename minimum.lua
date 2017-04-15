-------------------------
-- Instead of outputting a class, we are outputting an actual value that
-- we attempt to determine using regression. MSE criterion.
-------------------------
require 'nn'
require 'torch'
require 'xlua'
require 'optim'
require 'btplib'
-- require 'cudnn'
-- require 'cunn'
-- require 'cutorch'

model = torch.load('model_FCN_cpu.t7')
trainset = torch.load('train.t7')

function trainset:size()
  return self.images:size(1)
end

criterion = nn.MSECriterion()

logger = optim.Logger('./testmse.log')
logger2 = optim.Logger('./testrmse.log')

bSize = 50
size = 1000
jMax = 5 -- Large Epochs
iMax = 5 -- Small Epochs after which to save
for j=1,jMax do
  for i=1,iMax do
    currentError = 0
    trainerBatch(trainset,model,0.001,bSize,size,false,false,true)
    print('\nEpoch: ' .. (j-1)*iMax+i..'/'..iMax*jMax..' Per pixel MSE: ' ..currentError*bSize/size..' RMSE: ' ..torch.sqrt(currentError*bSize/size))
    logger:add{['error'] = currentError*bSize/size}
    logger2:add{['error'] = torch.sqrt(currentError*bSize/size)}
  end
  torch.save('model_FCN_cpu.t7',model)
end

classPerformanceEvaluator(trainset,model,true,torch.range(1,size))
randomEvaluator(size,10)
