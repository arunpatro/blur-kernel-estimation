matio = require 'matio'

trname = 'train.t7'
tename = 'test.t7'

print('Loading .mat file')
torchDB = matio.load('torchDB.mat')
imgsTrain = torchDB.trainTorch
imgsTest = torchDB.testTorch

-- create the trainset and testset in order
lblsTrain = torch.ceil( torch.range(1,imgsTrain:size(1)) / (imgsTrain:size(1)/30) ) 
lblsTest = torch.ceil(torch.range(1,imgsTest:size(1)) / (imgsTest:size(1)/30) ) 

dataset = {
	images = torch.Tensor(imgsTrain:size(1),1,32,32):byte(),
	labels = torch.Tensor(imgsTrain:size(1)):byte()
}

dataset2 = {
	images = torch.Tensor(imgsTest:size(1),1,32,32):byte(),
	labels = torch.Tensor(imgsTest:size(1)):byte()
}

print('Creating shuffled trainset')
p = torch.randperm(imgsTrain:size(1))
for i = 1,imgsTrain:size(1) do
	dataset.images[i] = imgsTrain[p[i]]
	dataset.labels[i] = lblsTrain[p[i]]
end

print('Creating shuffled testset')
p2 = torch.randperm(imgsTest:size(1))
for i = 1,imgsTest:size(1) do
	dataset2.images[i] = imgsTest[p2[i]]
	dataset2.labels[i] = lblsTest[p2[i]]
end

print('Saving datasets')
torch.save(trname,dataset)
torch.save(tename,dataset2)