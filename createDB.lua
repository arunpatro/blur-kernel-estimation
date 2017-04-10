require 'image'
require 'paths'
require 'xlua'

trname = 'trainSig3.t7'
trsize = 224000
tename = 'testSig3.t7'
tesize = 96000

-- create the trainset in order
imgs = torch.Tensor(trsize,1,32,32)
lbls = torch.ceil( torch.range(1,trsize) / (trsize/10) ) 
print('Training loading')
for j = 1,trsize do
	imgs[j] = image.load('./train/img_'..j..'.jpg',1,'byte')
	xlua.progress(j,trsize)
end
dataset = {
	images = torch.Tensor(trsize,1,32,32),
	labels = torch.Tensor(trsize)
}


-- create the testset in order
imgs2 = torch.Tensor(tesize,1,32,32)
lbls2 = torch.ceil(torch.range(1,tesize) / (tesize/10) ) 
print('Testing loading')
for j = 1,tesize do
	imgs2[j] = image.load('./test/img_'..j..'.jpg',1,'byte')
	xlua.progress(j,tesize)
end
dataset2 = {
	images = torch.Tensor(tesize,1,32,32),
	labels = torch.Tensor(tesize)
}

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
print('Saving training dataset')
torch.save(trname,dataset)
torch.save(tename,dataset2)
