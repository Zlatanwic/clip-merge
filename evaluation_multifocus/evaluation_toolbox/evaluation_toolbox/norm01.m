function normA=norm01(A)
%--------------------------------------------------------------------------
%函数功能：讲一个矩阵A化归到0到1的double型
%--------------------------------------------------------------------------
N=size(A);
min_value=min(min(A)); %最小
max_value=max(max(A)); %最大

Min=ones(N(1),N(2)).*min_value;
normA=1.0.*(A-Min)./(max_value-min_value);%归一化