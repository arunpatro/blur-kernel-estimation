require 'image'
require 'paths'
matio = require 'matio'

trname = 'trainSig.t7'
trsize = 224000
tename = 'testSig.t7'
tesize = 96000

print('1')
-- create the trainset in order
imgs = matio.load('train.mat').A
lbls = torch.ceil( torch.range(1,trsize) / (trsize/10) ) 

dataset = {
	images = torch.Tensor(trsize,1,32,32),
	labels = torch.Tensor(trsize)
}

-- create the testset in order
imgs2 = matio.load('test.mat').A
lbls2 = torch.ceil(torch.range(1,tesize) / (tesize/10) ) 

dataset2 = {
	images = torch.Tensor(tesize,1,32,32),
	labels = torch.Tensor(tesize)
}
print('2')
-- shuffling the trainset
p = torch.randperm(trsize)
for i = 1,trsize do
	dataset.images[i] = imgs[p[i]]
	dataset.labels[i] = lbls[p[i]]
end

--shuffling the testset
p2 = torch.randperm(tesize)
for i = 1,tesize do
	dataset2.images[i] = imgs2[p2[i]]
	dataset2.labels[i] = lbls2[p2[i]]
end

--saving
print('3')
torch.save(trname,dataset)
torch.save(tename,dataset2)