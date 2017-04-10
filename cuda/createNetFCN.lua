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
lenet:add(cudnn.SpatialConvolution(1,8,5,5,1,1,2,2))
lenet:add(cudnn.ReLU())
-- lenet:add(cudnn.SpatialConvolution(8,16,3,3))
-- lenet:add(cudnn.ReLU())
lenet:add(cudnn.SpatialConvolution(8,64,5,5,1,1,2,2))
lenet:add(cudnn.ReLU())
-- lenet:add(cudnn.SpatialConvolution(32,64,3,3))
-- lenet:add(cudnn.ReLU())
lenet:add(cudnn.SpatialConvolution(64,128,5,5,1,1,2,2))
lenet:add(cudnn.ReLU())
-- lenet:add(cudnn.SpatialConvolution(128,128,5,5))
-- lenet:add(cudnn.ReLU())
lenet:add(cudnn.SpatialConvolution(128,128,5,5,1,1,2,2))
lenet:add(cudnn.ReLU())
-- lenet:add(cudnn.SpatialMaxPooling(2,2,2,2))
-- lenet:add(cudnn.SpatialConvolution(128,128,3,3))
-- lenet:add(cudnn.ReLU())
lenet:add(cudnn.SpatialConvolution(128,256,5,5,1,1,2,2))
lenet:add(cudnn.ReLU())
lenet:add(cudnn.SpatialConvolution(256,64,5,5,1,1,2,2))
lenet:add(cudnn.ReLU())
lenet:add(cudnn.SpatialConvolution(64,1,5,5,1,1,2,2))
lenet:add(cudnn.ReLU())

params, gg = lenet:getParameters()
print(#params)

lenet = lenet:cuda()
a = torch.rand(1,32,32):cuda()
print(lenet:forward(a))

torch.save('lenet-one.t7',lenet)

