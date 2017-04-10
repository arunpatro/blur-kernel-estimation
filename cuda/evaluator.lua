------------------------------
-- This code generates the sigma map for a ramped blurred image by evaluating 
-- each 32x32 patch with stride 1. Since this CNN Classifies, we get a quantized
-- plot.
-------------------------------

require 'nn'
require 'torch'
require 'image'
require 'cudnn'
require 'cunn'
require 'cutorch'

-- local matio = require 'matio'

trainset = torch.load('trainSig3.t7');
mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
mean = trainset.images[{ {}, {1}, {}, {}  }]:mean(); -- mean estimation
stdv = trainset.images[{ {}, {1}, {}, {}  }]:std(); -- std estimation

lenet = torch.load('lenet.t7')

imageNo = 40
ramp = image.load('sin'.. imageNo.. '.jpg',1,'byte'):double():cuda();
ramp[{ {}, {}, {}  }]:add(-mean); -- mean subtraction
ramp[{ {}, {}, {}  }]:div(stdv);

function getSigmaClass(patch)
	prediction = lenet:forward(patch)
	confidences, indices = torch.sort(prediction, true)
	return indices[1]
end


map = torch.Tensor(609,609):byte();
for row = 1,609 do
for col = 1,609 do
print(row..'|'..col)
map[row][col] = getSigmaClass(ramp[{{1},{row,row+31},{col,col+31}}]);
end
end

torch.save('sin40.t7',map)
image.save('sin40.jpg',map)