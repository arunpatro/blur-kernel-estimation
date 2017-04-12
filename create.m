% load unblurred images
a = dir('images');
a = a(3:end);
images = {};
for i = 1:80
    images{i} = imread(fullfile('images',a(i).name));
end
images{1} = images{1}(1:640,1:640);
% create blur kernels
sigma = 0.05:0.05:3;
kernelSizes = 2*ceil(2*sigma)+1;
for i = 1:60
    sigmaFilters{i} = fspecial('gaussian',kernelSizes(i),sigma(i));
end

% create blurred images
for i = 1:60
    for j = 1:80
        blurredImages{(i-1)*80+j} = imfilter(images{j},sigmaFilters{i});
    end
end

%need to resize 1st image of 640x640 first
%create blurred patches
% samples_non_over_32 = samper(blurredImages);

%create individual images by sampling 70/30 of each blurred image
%(standard)
standard

%save images to path
seqsaver(test,'test/');
seqsaver(train,'train/');