------------------------------
-- This code generates the sigma map for a ramped blurred image by evaluating 
-- each 32x32 patch with stride 1. Since this CNN Classifies, we get a quantized
-- plot.
-------------------------------

require 'nn'
require 'torch'
require 'image'
local matio = require 'matio';

trainset = torch.load('train.t7');
trainset.images = trainset.images:double();
mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
mean = trainset.images[{ {}, {1}, {}, {}  }]:mean(); -- mean estimation
stdv = trainset.images[{ {}, {1}, {}, {}  }]:std(); -- std estimation

lenet = torch.load('model_10_cpu.t7')

imgName = arg[1]
print(imgName)
img = image.load(imgName,1,'byte'):double();
-- img = torch.reshape(img,1,img:size()[1],img:size()[2])
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
map = torch.Tensor(rows,cols);
for row = 1,rows do
	xlua.progress(row,rows)
	for col = 1,cols do
		map[row][col] = 0.1*getSigmaClassification(img[{{1},{row,row+31},{col,col+31}}]);
	end
end

matio.save(string.sub(imgName,1,-4)..'mat',map)

