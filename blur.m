function [out] = ramper(A,sigmaMin,sigmaMax)
out = [];
cols = size(A,2);
sigmaCols = 1.35*sin(0.02*(1:640))+1.65;
kernelSizes = 2*ceil(2*sigmaCols)+1;
pF = (kernelSizes(end)-1)/2; %padding factor == maximum of the padding required for largest filter kernel. This is neccesary as we are doing column wise convolution.
A = padarray(A,[pF,pF]);
for i = 1:cols
    filter = fspecial('gaussian',kernelSizes(i),sigmaCols(i));
    fF = (kernelSizes(i)-1)/2; %filter factor
    temp = imfilter(A(:,i+pF-fF:i+pF+fF),filter,'conv');
    out = [out temp(pF+1:end-pF,fF+1)];
end