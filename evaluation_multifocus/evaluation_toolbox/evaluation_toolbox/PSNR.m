function  [rmse psnr]=PSNR(A,B)

if size(A)~=size(B)
    error('two images are not the same size')
end

if A==B
    error('two images are the same,their PSNR has infinite value')
end

A=double(A);
B=double(B);%必须先转换成double类型，否则数据会截断到255
diff=A-B;

rmse=sqrt(mean(mean(diff.*diff)));
psnr=20*log10(255/rmse);
