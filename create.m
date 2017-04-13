% load unblurred images
a = dir('images');
a = a(3:end);
images = {};
for i = 1:80
    images{i} = imread(fullfile('images',a(i).name));
end
images{1} = images{1}(1:640,1:640);
% create blur kernels
sigma = 0.3:0.3:3;
kernelSizes = 2*ceil(2*sigma)+1;
for i = 1:size(sigma,2)
    sigmaFilters{i} = fspecial('gaussian',kernelSizes(i),sigma(i));
end

% create blurred images
for i = 1:size(sigma,2)
    for j = 1:80
        blurredImages{(i-1)*80+j} = imfilter(images{j},sigmaFilters{i});
    end
end

%create individual images by sampling 70/30 of each blurred image
train = {};
test = {};
div = 32*ones(1,20);
n = size(blurredImages,2);
for ctrn = 1:n
    sampI = blurredImages{ctrn};
    disp(ctrn)
    % for 20x20 = 400 patches per image
    temp = reshape(mat2cell(sampI,div,div),1,400);
    perm = randperm(400);
    temp = temp(perm(1:400));
    train = [train temp(1:280)];
    test = [test temp(281:400)];
end

trainTorch = zeros(size(train,2),1,32,32,'uint8');
testTorch = zeros(size(test,2),1,32,32,'uint8');

%reshaping for torch
for ctr = 1:size(train,2)
	trainTorch(ctr,1,:,:) = train{ctr};
end
for ctr = 1:size(test,2)
	testTorch(ctr,1,:,:) = test{ctr};
end

save('torchDB.mat','trainTorch','testTorch')
