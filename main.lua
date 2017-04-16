require 'nn'
require 'torch'
require 'xlua'
require 'math'
require 'btplib'

model = torch.load('model_30_cpu.t7')

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
for i=1,trainset.images:size(2) do
	mean[i] = trainset.images[{ {}, {i}, {}, {}  }]:mean()
	trainset.images[{ {}, {i}, {}, {}  }]:add(-mean[i])

	stdv[i] = trainset.images[{ {}, {i}, {}, {}  }]:std()
	trainset.images[{ {}, {i}, {}, {}  }]:div(stdv[i])

	testset.images[{ {}, {i}, {}, {}  }]:add(-mean[i])
	testset.images[{ {}, {i}, {}, {}  }]:div(stdv[i])
end

criterion = nn.ClassNLLCriterion()

currentError = 0
for epoch = 1,5 do
  trainerBatch(trainset,model,0.001,50,1000,false,true,false)
  if epoch%3==0 then
    testClass(testset,model,false,50,1000)
    -- torch.save('model_30_cpu.t7',model)
  end
end