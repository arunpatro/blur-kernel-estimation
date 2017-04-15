require 'nn'
require 'cudnn'
require 'cunn'

model = nn.Sequential()
model:add(cudnn.SpatialConvolution(1,4,5,5))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(4,8,5,5))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialMaxPooling(2,2,2,2))
model:add(cudnn.SpatialConvolution(8,16,5,5))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(16,32,5,5))
model:add(cudnn.ReLU())
model:add(nn.View(32*4*4))
model:add(nn.Linear(32*4*4,200))
model:add(cudnn.ReLU())
model:add(nn.Linear(200,30))
model:add(nn.LogSoftMax()) 

params, gg = model:getParameters()
print(#params)
model = model:cuda()

torch.save('model_30_gpu.t7',model)

