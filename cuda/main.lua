require 'cudnn'
require 'cunn'
require 'nn'
require 'torch'
require 'xlua'
require 'math'
require 'optim'
require 'btplib'

model = torch.load('model_30_gpu.t7')

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
mean = trainset.images[{ {}, {1}, {}, {}  }]:mean();
trainset.images[{ {}, {1}, {}, {}  }]:add(-mean);

stdv = trainset.images[{ {}, {1}, {}, {}  }]:std();
trainset.images[{ {}, {1}, {}, {}  }]:div(stdv);

testset.images[{ {}, {1}, {}, {}  }]:add(-mean);
testset.images[{ {}, {1}, {}, {}  }]:div(stdv);

criterion = nn.ClassNLLCriterion()

parameters,gradParameters = model:getParameters()
confusion = optim.ConfusionMatrix(30)
trainLogger = optim.Logger('./train.log')
testLogger = optim.Logger('./test.log')

currentError = 0
for epoch = 1,500 do
  trainerBatch(trainset,model,0.001,50,trainset:size(),true,true,false)
  if epoch%3==0 then
    testClass(testset,model,true,50,testset:size())
    torch.save('model_30_gpu.t7',model)
  end
end