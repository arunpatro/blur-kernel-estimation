------------------------------
-- This code generates the sigma map for a ramped blurred image by evaluating 
-- each 32x32 patch with stride 1. Since this CNN Classifies, we get a quantized
-- plot. Run it as `th evaluator.lua image.jpg`
-------------------------------

require 'nn';
require 'torch';
require 'image';
require 'cudnn';
require 'cunn';
require 'xlua';
require 'string'

local matio = require 'matio';

trainset = torch.load('trainSig30small.t7');
mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
mean = trainset.images[{ {}, {1}, {}, {}  }]:mean(); -- mean estimation
stdv = trainset.images[{ {}, {1}, {}, {}  }]:std(); -- std estimation

lenet = torch.load('lenet30.t7')

imgName = arg[1]
print(imgName)
img = image.load(imgName,1,'byte'):double():cuda();
img[{ {}, {}, {}  }]:add(-mean); -- mean subtraction
img[{ {}, {}, {}  }]:div(stdv);

function getSigmaClassification(patch)
	prediction = lenet:forward(patch)
	confidences, indices = torch.sort(prediction, true)
	return indices[1]
end

function getSigmaRegression(patch)
	return lenet:forward(patch)[{1,1,1}]
end

rows = img:size()[2] - 31
cols = img:size()[3] - 31
map = torch.Tensor(rows,cols):byte();
for row = 1,rows do
	xlua.progress(row,rows)
	for col = 1,cols do
		map[row][col] = getSigmaClassification(img[{{1},{row,row+31},{col,col+31}}]);
	end
end

matio.save(string.sub(imgName,1,-4)..'mat',map)