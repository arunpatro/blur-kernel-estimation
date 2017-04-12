require 'nn'
require 'cudnn'
require 'cunn'
require 'cutorch'

-------------------------------------------------------
-- This is a straight CNN without any reduction in size, can be used for proof
-------------------------------------------------------
lenet = nn.Sequential()
lenet:add(cudnn.SpatialConvolution(1,4,3,3))
lenet:add(cudnn.ReLU())
lenet:add(cudnn.SpatialConvolution(4,8,3,3))
lenet:add(cudnn.ReLU())
lenet:add(cudnn.SpatialConvolution(8,16,3,3))
lenet:add(cudnn.ReLU())
lenet:add(cudnn.SpatialConvolution(16,32,3,3))
lenet:add(cudnn.ReLU())
lenet:add(cudnn.SpatialConvolution(32,64,3,3))
lenet:add(cudnn.ReLU())
lenet:add(cudnn.SpatialConvolution(64,64,3,3))
lenet:add(cudnn.ReLU())
lenet:add(cudnn.SpatialConvolution(64,64,5,5))
lenet:add(cudnn.ReLU())
lenet:add(cudnn.SpatialMaxPooling(2,2,2,2))
lenet:add(cudnn.SpatialConvolution(64,64,5,5))
lenet:add(cudnn.ReLU())
lenet:add(nn.View(64*4*4))
lenet:add(nn.Linear(64*4*4,1600))
lenet:add(nn.ReLU())
lenet:add(nn.Linear(1600,800))
lenet:add(nn.ReLU())
lenet:add(nn.Linear(800,100))
lenet:add(nn.ReLU())
lenet:add(nn.Linear(100,30))
lenet:add(nn.LogSoftMax()) 

params, gg = lenet:getParameters()
print(#params)

lenet = lenet:cuda()
print(lenet:forward(torch.rand(1,32,32):cuda()))

torch.save('lenet30.t7',lenet)

