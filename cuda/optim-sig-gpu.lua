
require 'nn'
require 'torch'
require 'xlua'
require 'math'
require 'optim'
require 'cudnn'
require 'cunn'
require 'cutorch'

lenet = torch.load('lenet.t7')

print('Loading the training set')
trainset = torch.load('trainSig3.t7');
print('Loading the testing set')
testset = torch.load('testSig3.t7')

--create indexable dataset so that we can call each sample individually
setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.images[i]:cuda(), t.labels[i]} 
                end}
);
trainset.images = trainset.images:double(); -- convert the data from a ByteTensor to a DoubleTensor.

function trainset:size() 
    return self.images:size(1) 
end

setmetatable(testset, 
    {__index = function(t, i) 
                    return {t.images[i]:cuda(), t.labels[i]} 
                end}
);
testset.images = testset.images:double(); -- convert the data from a ByteTensor to a DoubleTensor.
function testset:size() 
    return self.images:size(1) 
end

--preprocess data
mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
mean = trainset.images[{ {}, {1}, {}, {}  }]:mean() -- mean estimation
trainset.images[{ {}, {1}, {}, {}  }]:add(-mean) -- mean subtraction

stdv = trainset.images[{ {}, {1}, {}, {}  }]:std() -- std estimation
trainset.images[{ {}, {1}, {}, {}  }]:div(stdv) -- std scaling

testset.images[{ {}, {1}, {}, {}  }]:add(-mean) -- mean subtraction
testset.images[{ {}, {1}, {}, {}  }]:div(stdv) -- std scaling

print(mean)
print(stdv)

-- criterion for multiclass classification for loss calculation
criterion = nn.ClassNLLCriterion():cuda()

--all of this for implementing the AdaGrad algorithm using the optim library of torch 
parameters,gradParameters = lenet:getParameters()
confusion = optim.ConfusionMatrix(10)
trainLogger = optim.Logger('./train.log')
testLogger = optim.Logger('./test.log')

opt = {learningRate = 0.05,
batchSize = 50,
plot = false}

config = {
    learningRate = opt.learningRate,
    learningRateDecay = 1e-7
}

epoch = 1
function train(dataset)
   local time = sys.clock()
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,dataset:size(),opt.batchSize do
      local inputs = torch.Tensor(opt.batchSize,1,32,32):cuda()
      local targets = torch.Tensor(opt.batchSize)
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
        sample = trainset[i]
        inputs[k] = sample[1]:cuda()
        targets[k] = sample[2]
        k = k + 1
      end
      local feval = function(x)
         collectgarbage()
         if x ~= parameters then
            parameters:copy(x)
         end
         gradParameters:zero()
         local outputs = lenet:forward(inputs)
         local f = criterion:forward(outputs, targets)
         local df_do = criterion:backward(outputs, targets)
         lenet:backward(inputs, df_do)
         for i = 1,opt.batchSize do
            confusion:add(outputs[i], targets[i])
         end
         return f,gradParameters
      end
         optim.adagrad(feval, parameters, config)
         xlua.progress(t, dataset:size())
   end
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')
   print(confusion)
   print('\27[32mTrain: ' .. confusion.totalValid * 100)
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   confusion:zero()
   epoch = epoch + 1
end

-- function to test the accuracy in batches for speed 
function test(dataset)
   local time = sys.clock()
   print('<trainer> on testing Set:')
   for t = 1,dataset:size(),opt.batchSize do
      xlua.progress(t, dataset:size())
      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,1,32,32):cuda()
      local targets = torch.Tensor(opt.batchSize)
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
        sample = testset[i]
        inputs[k] = sample[1]:cuda()
        targets[k] = sample[2]
        k = k + 1
      end
      -- test samples
      local preds = lenet:forward(inputs)
      -- confusion:
      for i = 1,opt.batchSize do
         confusion:add(preds[i], targets[i])
      end
   end
   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')
   -- print confusion matrix
   print(confusion)
   print('\27[31mTest: ' .. confusion.totalValid * 100)
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   confusion:zero()
end


-- test(testset)
--run the code for 100 epochs
for ctr1 = 1,100 do
  train(trainset)
  if ctr1%2==0 then
    -- test(testset)
  end
  if opt.plot then
    trainLogger:style{['% mean class accuracy (train set)'] = '-'}
    testLogger:style{['% mean class accuracy (test set)'] = '-'}
    trainLogger:plot()
    testLogger:plot()
  end
  torch.save('lenet.t7',lenet)
end
