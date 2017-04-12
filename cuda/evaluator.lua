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

local matio = require 'matio'

trainset = torch.load('trainSig3small.t7');
mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
mean = trainset.images[{ {}, {1}, {}, {}  }]:mean(); -- mean estimation
stdv = trainset.images[{ {}, {1}, {}, {}  }]:std(); -- std estimation

lenet = torch.load('lenetSmallClassPerfect.t7')
imageNo = 15
test = image.load('ramp'.. imageNo.. '_10.jpg',1,'byte'):double():cuda();
test[{ {}, {}, {}  }]:add(-mean); -- mean subtraction
test[{ {}, {}, {}  }]:div(stdv);

function getSigmaClassification(patch)
	prediction = lenet:forward(patch)
	confidences, indices = torch.sort(prediction, true)
	return indices[1]
end

function getSigmaRegression(patch)
	return lenet:forward(patch)[{1,1,1}]
end

map = torch.Tensor(609,609):byte();
for row = 1,609 do
	-- print(row)
	xlua.progress(row,609)
	for col = 1,609 do
		-- print(row..'|'..col)
		temp = getSigmaClassification(test[{{1},{row,row+31},{col,col+31}}]);
		map[row][col] = temp
	end
end

-- torch.save('ramp.t7',map)
matio.save('ramp15.mat',map)