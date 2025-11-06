function [ y] = rgb_Q0_n( I,F )
%RGB_Q0_N Summary of this function goes here
%   Detailed explanation goes here
r=I(:,:,1,:);
g=I(:,:,2,:);
b=I(:,:,3,:);
[w,h,l,N]=size(r);
r=reshape(r,[w,h,N]);
g=reshape(g,[w,h,N]);
b=reshape(b,[w,h,N]);
y_r=Q0_n(r,F(:,:,1));
y_g=Q0_n(g,F(:,:,2));
y_b=Q0_n(b,F(:,:,3));
y=(y_r+y_g+y_b)/3;
end

