------------------------------
-- This code generates the sigma map for a ramped blurred image by evaluating 
-- each 32x32 patch with stride 1. Since this CNN Classifies, we get a quantized
-- plot. Run it as `th evaluator.lua image.jpg`
-------------------------------

require 'nn';
require 'torch';
require 'image';
require 'xlua';
require 'string'
-- require 'cudnn';
-- require 'cunn';

local matio = require 'matio';

trainset = torch.load('trainSig30small.t7');
trainset.images = trainset.images:double();
mean = {}
stdv  = {}
mean = trainset.images[{ {}, {1}, {}, {}  }]:mean();
stdv = trainset.images[{ {}, {1}, {}, {}  }]:std();

model = torch.load('lenet30.t7')

function getSigmaClassification(patch)
	prediction = model:forward(patch)
	confidences, indices = torch.sort(prediction, true)
	return indices[1]
end

function getSigmaRegression(patch)
	return model:forward(patch)[{1,1,1}]
end

for i, imgName in ipairs(arg) do
	img = image.load(imgName,1,'byte'):double();
	img:add(-mean);
	img:div(stdv);
	rows = img:size(2) - 31
	cols = img:size(3) - 31
	map = torch.Tensor(rows,cols);
	print('Generating sigma map for ' .. imgName);
	for row = 1,rows do
		xlua.progress(row,rows)
		for col = 1,cols do
			map[row][col] = 0.1*getSigmaClassification(img[{{},{row,row+31},{col,col+31}}]);
		end
	end
	matio.save(string.sub(imgName,1,-4)..'mat',map)
end