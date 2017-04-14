require 'nn'
require 'torch'
require 'xlua'
require 'math'
require 'optim'
require 'btplib'

model = torch.load('model_10_cpu.t7')

print('Loading the training set')
trainset = torch.load('train.t7');
print('Loading the testing set')
testset = torch.load('test.t7')

--create indexable dataset so that we can call each sample individually
setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.images[i], t.labels[i]} 
                end}
);
trainset.images = trainset.images:double(); -- convert the data from a ByteTensor to a DoubleTensor.

function trainset:size() 
    return self.images:size(1) 
end

setmetatable(testset, 
    {__index = function(t, i) 
                    return {t.images[i], t.labels[i]} 
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

-- criterion for multiclass classification for loss calculation
criterion = nn.ClassNLLCriterion()

--all of this for implementing the AdaGrad algorithm using the optim library of torch 
parameters,gradParameters = model:getParameters()
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

-- function to test the accuracy in batches for speed 
function test(dataset)
   local time = sys.clock()
   print('<trainer> on testing Set:')
   for t = 1,dataset:size(),opt.batchSize do
      xlua.progress(t, dataset:size())
      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,1,32,32)
      local targets = torch.Tensor(opt.batchSize)
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
        sample = testset[i]
        inputs[k] = sample[1]
        targets[k] = sample[2]
        k = k + 1
      end
      -- test samples
      local preds = model:forward(inputs)
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
  currentError = 0
for ctr1 = 1,2 do
  trainerBatch(trainset,model,0.001,50,3000,false,true,false)
  -- print(currentError)
    -- test(testset)
  if ctr1%2==0 then
  end
  if opt.plot then
    trainLogger:style{['% mean class accuracy (train set)'] = '-'}
    testLogger:style{['% mean class accuracy (test set)'] = '-'}
    trainLogger:plot()
    testLogger:plot()
  end
end
test(testset)
  torch.save('model_10_cpu.t7',model)
