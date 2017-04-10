require 'nn'
require 'cudnn'
require 'cunn'
require 'cutorch'

-------------------------------------------------------
-- This is a straight CNN without any reduction in size, can be used for proof
-------------------------------------------------------
lenet = nn.Sequential()
lenet:add(cudnn.SpatialConvolution(1,4,5,5))
lenet:add(cudnn.ReLU())
lenet:add(cudnn.SpatialConvolution(4,8,5,5))
lenet:add(cudnn.ReLU())
lenet:add(cudnn.SpatialConvolution(8,16,5,5))
lenet:add(cudnn.ReLU())
lenet:add(cudnn.SpatialMaxPooling(2,2,2,2))
lenet:add(cudnn.SpatialConvolution(16,64,5,5))
lenet:add(cudnn.ReLU())
lenet:add(nn.View(64*6*6))
lenet:add(nn.Linear(64*6*6,800))
lenet:add(nn.ReLU())
lenet:add(nn.Linear(800,100))
lenet:add(nn.ReLU())
lenet:add(nn.Linear(100,10))
lenet:add(nn.LogSoftMax()) 

params, gg = lenet:getParameters()
print(#params)

lenet = lenet:cuda()
print(lenet:forward(torch.rand(1,32,32):cuda()))

torch.save('lenet.t7',lenet)

