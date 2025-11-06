function R=EOG(A)

A=double(A);

dx=[-1 1]';
dy=[-1 1];
gx=imfilter(A,dx,'replicate');
gy=imfilter(A,dy,'replicate');
R=sum(sum(gx.^2+gy.^2));