require 'nn'

model = nn.Sequential()
model:add(nn.SpatialConvolution(1,4,5,5))
model:add(nn.ReLU())
model:add(nn.SpatialConvolution(4,8,5,5))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.SpatialConvolution(8,16,5,5))
model:add(nn.ReLU())
model:add(nn.SpatialConvolution(16,32,5,5))
model:add(nn.ReLU())
model:add(nn.View(32*4*4))
model:add(nn.Linear(32*4*4,200))
model:add(nn.ReLU())
model:add(nn.Linear(200,30))
model:add(nn.LogSoftMax()) 

params, gg = model:getParameters()
print(#params)

torch.save('model_30_cpu.t7',model)

