require 'nn'
require 'torch'
require 'xlua'
require 'math'
require 'optim'
require 'btplib'

model = torch.load('model_10_cpu.t7')

print('Loading the training set')
trainset = torch.load('train.t7');
trainset.images = trainset.images:double();
function trainset:size() 
    return self.images:size(1) 
end

print('Loading the testing set')
testset = torch.load('test.t7')
testset.images = testset.images:double();
function testset:size() 
    return self.images:size(1) 
end


print('Preprocessing data')
mean = {}
stdv  = {}
mean = trainset.images[{ {}, {1}, {}, {}  }]:mean()
trainset.images[{ {}, {1}, {}, {}  }]:add(-mean)

stdv = trainset.images[{ {}, {1}, {}, {}  }]:std()
trainset.images[{ {}, {1}, {}, {}  }]:div(stdv)

testset.images[{ {}, {1}, {}, {}  }]:add(-mean)
testset.images[{ {}, {1}, {}, {}  }]:div(stdv)

criterion = nn.ClassNLLCriterion()

parameters,gradParameters = model:getParameters()
confusion = optim.ConfusionMatrix(10)
trainLogger = optim.Logger('./train.log')
testLogger = optim.Logger('./test.log')

function test(dataset,bSize,size)
   print('<trainer> on testing Set:')
   for t = 1,size,bSize do
      xlua.progress(t, dataset:size())
      -- create mini batch
      local inputs = dataset.images[{{t, math.min(t+bSize-1,size)}}]
      local targets = dataset.labels[{{t, math.min(t+bSize-1,size)}}]
      -- test samples
      local preds = model:forward(inputs)
      -- confusion:
      for i = 1,bSize do
         confusion:add(preds[i], targets[i])
      end
   end
   print(confusion)
   print('\27[31mTest: ' .. confusion.totalValid * 100)
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   confusion:zero()
end

-- test(testset)
currentError = 0
for epoch = 1,5 do
  trainerBatch(trainset,model,0.001,10,500,false,true,false)
  if epoch%3==0 then
    test(testset,50,1000)
    torch.save('model_10_cpu.t7',model)
  end
end
