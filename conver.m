% Column wise convolution
function [out] = conver(A,B)
out = [];
cols = size(A,2);
A = padarray(A,[1,1]);
for i = 1:cols
    temp = conv2(A(:,i:i+2),B,'same');
    out = [out temp(2:end-1,2)];
end
end