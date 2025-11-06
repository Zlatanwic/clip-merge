function [SF,EOG,SML]=spatial_clarity(A)
%-------------------------------------------------------------------------%
%函数功能：测试函数，用于对融合结果进行一些准则函数下的定量评价
% 参数说明：
% A：输入图像
% 输出：
% SF，EOG，SML：一些准则函数值
%-------------------------------------------------------------------------%
A=double(A);

% %SF
[m,n]=size(A);
df_c=A(2:m,:)-A(1:m-1,:);
df_r=A(:,2:n)-A(:,1:n-1);
mean_c=sum(sum(df_c.^2))/(m*n);
mean_r=sum(sum(df_r.^2))/(m*n);
SF=sqrt(mean_c+mean_r);

%EOG
[dx,dy]=gradient(A);
EOG=sum(sum(dx.^2+dy.^2));

% %SML
hx=[-1 2 -1];
hy=[-1 2 -1]';
dx2=imfilter(A,hx);
dy2=imfilter(A,hy);
d2=abs(dx2)+abs(dy2);
SML=sum(sum(d2));

