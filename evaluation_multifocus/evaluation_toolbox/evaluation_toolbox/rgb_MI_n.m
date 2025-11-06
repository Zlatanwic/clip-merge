function [ y ] = rgb_MI_n( I,F )
%RGB_MI_N Summary of this function goes here
%   Detailed explanation goes here
r=I(:,:,1,:);
g=I(:,:,2,:);
b=I(:,:,3,:);
[w,h,l,N]=size(r);
r=reshape(r,[w,h,N]);
g=reshape(g,[w,h,N]);
b=reshape(b,[w,h,N]);
y_r=MI_n(r,F(:,:,1));
y_g=MI_n(g,F(:,:,2));
y_b=MI_n(b,F(:,:,3));
y=(y_r+y_g+y_b)/3;
end

