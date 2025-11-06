function [ y ] = MI_n( I,F )
%MI_N Summary of this function goes here
%   Detailed explanation goes here
[r,c,N]=size(I);
% I=double(I)/255;
% F=double(F)/255;
estimated=zeros(1,N);
for i=1:N
mono=I(:,:,i);
% estimated(i)=Tsallis_MI(mono,F);
estimated(1,i)=MI2(mono,F);
end
y=sum(estimated);

