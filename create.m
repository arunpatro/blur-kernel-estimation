% load unblurred images
a = dir('images');
a = a(3:end);
images = {};
for i = 1:size(a,1)
    images{i} = imread(fullfile('images',a(i).name));
    [r c channels] = size(images{i});
    images{i} = images{i}(1:32*floor(r/32),1:32*floor(c/32),:);
end

% create blur kernels
sigma = 0.1:0.1:3;
kernelSizes = 2*ceil(2*sigma)+1;
for i = 1:size(sigma,2)
    sigmaFilters{i} = fspecial('gaussian',kernelSizes(i),sigma(i));
end

% create blurred images
for i = 1:size(sigma,2)
    for j = 1:size(images,2)
        blurredImages{(i-1)*size(images,2)+j} = imfilter(images{j},sigmaFilters{i});
    end
end

%create individual images by sampling with ratio2 of each blurred image
train = {};
test = {};

n = size(blurredImages,2);
ratio1 = 0.25;
ratio2 = 0.3;
for ctrn = 1:n
    disp(ctrn)
    img = blurredImages{ctrn};
    [r c ~] = size(img);
    nRowPatch = floor(r/32);
    nColPatch = floor(c/32);
    temp = reshape(mat2cell(img,32*ones(1,nRowPatch),32*ones(1,nColPatch)),1,nRowPatch*nColPatch);
    perm = randperm(nRowPatch*nColPatch);
    temp = temp(perm(1:floor(ratio1*nRowPatch*nColPatch)));
    test = [test temp(1:floor(ratio2*size(temp,2)))];
    train = [train temp(floor(ratio2*size(temp,2))+1:size(temp,2))];
end

trainTorch = zeros(size(train,2),channels,32,32,'uint8');
testTorch = zeros(size(test,2),channels,32,32,'uint8');

%reshaping for torch
for patch = 1:size(train,2)
    for channel = 1:channels
        trainTorch(patch,channel,:,:) = train{patch}(:,:,channel);
    end
end
for patch = 1:size(test,2)
    for channel = 1:channels
        testTorch(patch,channel,:,:) = test{patch}(:,:,channel);
    end
end

save('torchDB.mat','trainTorch','testTorch')
