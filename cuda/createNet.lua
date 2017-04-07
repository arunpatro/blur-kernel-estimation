require 'nn'
require 'cudnn'
require 'cunn'
require 'cutorch'

-------------------------------------------------------
-- This is a straight CNN without any reduction in size
-------------------------------------------------------
-- lenet = nn.Sequential()
-- lenet:add(nn.SpatialConvolution(1,4,5,5,1,1,2,2))
-- lenet:add(nn.ReLU())
-- lenet:add(nn.SpatialConvolution(4,8,3,3,1,1,1,1))
-- lenet:add(nn.ReLU())
-- lenet:add(nn.SpatialConvolution(8,16,3,3,1,1,1,1))
-- lenet:add(nn.ReLU())
-- -- lenet:add(nn.SpatialConvolution(16,64,5,5,1,1,2,2))
-- -- lenet:add(nn.ReLU())
-- -- lenet:add(nn.SpatialFullConvolution(64,16,5,5,1,1,2,2))
-- -- lenet:add(nn.ReLU())
-- lenet:add(nn.SpatialConvolution(16,8,3,3,1,1,1,1))
-- lenet:add(nn.ReLU())
-- lenet:add(nn.SpatialConvolution(8,4,3,3,1,1,1,1))
-- lenet:add(nn.ReLU())
-- lenet:add(nn.SpatialConvolution(4,1,5,5,1,1,2,2))
-- lenet:add(nn.ReLU())

-- params, gg = lenet:getParameters()
-- print(#params)

-- torch.save('lenet-test.t7',lenet)

---------------------------------------------------
-- This is for producing one value for MSE
---------------------------------------------------
lenet = nn.Sequential()
lenet:add(cudnn.SpatialConvolution(1,4,5,5))
lenet:add(cudnn.ReLU())
lenet:add(cudnn.SpatialConvolution(4,8,3,3))
lenet:add(cudnn.ReLU())
lenet:add(cudnn.SpatialConvolution(8,16,5,5))
lenet:add(cudnn.ReLU())
lenet:add(cudnn.SpatialConvolution(16,32,5,5))
lenet:add(cudnn.ReLU())
lenet:add(cudnn.SpatialConvolution(32,64,5,5))
lenet:add(cudnn.ReLU())
lenet:add(cudnn.SpatialMaxPooling(2,2,2,2))
lenet:add(cudnn.SpatialConvolution(64,64,3,3))
lenet:add(cudnn.ReLU())
lenet:add(cudnn.SpatialConvolution(64,8,3,3))
lenet:add(cudnn.ReLU())
lenet:add(nn.SpatialConvolution(8,1,3,3))
lenet:add(nn.ReLU())

params, gg = lenet:getParameters()
print(#params)

lenet = lenet:cuda()


torch.save('lenet-one.t7',lenet)