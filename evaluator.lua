------------------------------
-- This code generates the sigma map for a ramped blurred image by evaluating 
-- each 32x32 patch with stride 1. Since this CNN Classifies, we get a quantized
-- plot.
-------------------------------

require 'nn'
require 'torch'
require 'image'
local matio = require 'matio'

trainset = torch.load('trainSigActual.t7');
mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
mean = trainset.images[{ {}, {1}, {}, {}  }]:mean(); -- mean estimation
stdv = trainset.images[{ {}, {1}, {}, {}  }]:std(); -- std estimation

lenet = torch.load('lenet.t7')

imageNo = 40
ramp = image.load('ramps/ramp'.. imageNo.. '.jpg',1,'byte'):double();
ramp[{ {}, {}, {}  }]:add(-mean); -- mean subtraction
ramp[{ {}, {}, {}  }]:div(stdv);

function getSigma(patch)
	prediction = lenet:forward(patch)
	confidences, indices = torch.sort(prediction, true)
	return indices[1]
end


map = torch.Tensor(609,609):byte();
for row = 1,609 do
for col = 1,609 do
print(row..'|'..col)
map[row][col] = getSigma(ramp[{{1},{row,row+31},{col,col+31}}]);
end
end

matio.save('map'..imageNo..'.mat',map)