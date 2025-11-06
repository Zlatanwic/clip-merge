function [ y ] = rgb_Qe_n( I,F )
r=I(:,:,1,:);
g=I(:,:,2,:);
b=I(:,:,3,:);
[w,h,l,N]=size(r);
r=reshape(r,[w,h,N]);
g=reshape(g,[w,h,N]);
b=reshape(b,[w,h,N]);
y_r=Qe_n(r,F(:,:,1),1);
y_g=Qe_n(g,F(:,:,2),1);
y_b=Qe_n(b,F(:,:,3),1);
y=(y_r+y_g+y_b)/3;
end
