function [ y ] = Qe_n( I,F,a)
[r,c,N]=size(I);
I=double(I)/255;
F=double(F)/255;
for i=1:N
Sobel=[-1 -2 -1;
        0  0  0;
        1  1  1];
    
ghI(:,:,i)=imfilter(I(:,:,i),Sobel);
glI(:,:,i)=imfilter(I(:,:,i),Sobel');
gI(:,:,i)=sqrt(ghI(:,:,i).^2+glI(:,:,i).^2);

end

ghF=imfilter(F,Sobel);
glF=imfilter(F,Sobel');
gF=sqrt(ghF.^2+glF.^2);

y1=Qw_n(I,F);
y2=Qw_n(gI,gF);
y=y1*y2^a;

