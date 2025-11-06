function [ y ] = rgb_Qw_n( I,F )
r=I(:,:,1,:);
g=I(:,:,2,:);
b=I(:,:,3,:);
[w,h,l,N]=size(r);
r=reshape(r,[w,h,N]);
g=reshape(g,[w,h,N]);
b=reshape(b,[w,h,N]);
y_r=Qw_n(r,F(:,:,1));
y_g=Qw_n(g,F(:,:,2));
y_b=Qw_n(b,F(:,:,3));
y=(y_r+y_g+y_b)/3;
end


