clear all;
close all;
%clc;
addpath(genpath('evaluation_toolbox'));
addpath(genpath('vifvec_release'));

num_alg=1;
num_metric=4;
num_img=148;
Q=zeros(num_img,num_alg,num_metric);

for kk=1:num_img
disp(kk)
h1=floor(kk/100);
h2=floor((kk-h1*100)/10);
h3=mod(kk,10);

name1=['../input_image/GFP/GFP_' num2str(h1) num2str(h2) num2str(h3) '.bmp'];
name2=['../input_image/PCI/PCI_' num2str(h1) num2str(h2) num2str(h3) '.bmp'];
namef=cell(1,num_alg);

% namef{1}=['../comparision/lp_cnn/PCI_' num2str(h1) num2str(h2) num2str(h3) '.bmp'];
% namef{2}=['../comparision/CST/' num2str(h1) num2str(h2) num2str(h3) '.bmp'];
% namef{3}=['../comparision/F_PCI_6_6_6/epoch9/imsave/' num2str(h1) num2str(h2) num2str(h3) '.bmp'];
% namef{4}=['../comparision/SFL-CT/' num2str(h1) num2str(h2) num2str(h3) '.bmp'];
% namef{5}=['../comparision/SOMP_SR/PCI_' num2str(h1) num2str(h2) num2str(h3) '.bmp'];
% namef{6}=['../comparision/nsst_papcnn/PCI_' num2str(h1) num2str(h2) num2str(h3) '.bmp'];
% namef{7}=['../comparision/DTCWT/' num2str(h1) num2str(h2) num2str(h3) '.bmp'];
% namef{8}=['../comparision/CVT/' num2str(h1) num2str(h2) num2str(h3) '.bmp'];
% namef{9}=['../comparision/NSCT/' num2str(h1) num2str(h2) num2str(h3) '.bmp'];
% namef{10}=['../result/' num2str(h1) num2str(h2) num2str(h3) '.bmp'];
namef{11}=['../result_low_average/' num2str(h1) num2str(h2) num2str(h3) '.bmp'];
for i=1:num_alg

A=imread(name1);B=imread(name2);
F=imread(namef{11});

A=rgb2gray(A);F=rgb2gray(F); 

img1=double(A);img2=double(B);imgf=double(F);
%[H W]=size(img1);I=zeros(H,W,2);I(:,:,1)=img1;I(:,:,2)=img2;


  
% %Information Theory-Based Metrics
Q(kk,i,1)=metricMI(img1,img2,imgf,1);%% normalized mutual informtion $Q_{MI}$
% Q(kk,i,2)=metricMI(img1,img2,imgf,3);% Tsallis entropy $Q_{TE}$
% Q(kk,i,3)=metricWang(img1,img2,imgf); % Wang - NCIE $Q_{NCIE}$
%Image Feature-Based Metrics
Q(kk,i,4)=edge_association(img1,img2,imgf);% Xydeas $Q_G$
% Q(kk,i,5)=metricPWW(img1,img2,imgf);% PWW $Q_M$
% Q(kk,i,6)=metricZheng(img1,img2,imgf);   %越接近0越好 %Yufeng Zheng (spatial frequency) $Q_{SF}$
% Q(kk,i,7)=metricZhao(img1,img2,imgf);% Zhao (phase congrency) $Q_P$
% Image Structural Similarity-Based Metrics
% Q(kk,i,8)=metricPeilla(img1,img2,imgf,1);% Piella  (need to select only one) $Q_S$
% Q(kk,i,9)=metricCvejic(img1,img2,imgf,2);% Cvejie $Q_C$
% Q(kk,i,10)=metricYang(img1,img2,imgf);% Yang $Q_Y$
% %Human Perception Inspired Fusion Metrics
% Q(kk,i,11)=metricChen(img1,img2,imgf)*0.1;   % 越小越好Qcv % Chen-Varshney $Q_{CV}$
Q(kk,i,12)=metricChenBlum(img1,img2,imgf);  % Chen-Blum $Q_{CB}$
% %%%%%%%%%%%%%%
% Q(kk,i,13)=entropy_fusion(imgf,256);  
% Q(kk,i,14)=LMI(img1,img2,imgf,0);
% Q(kk,i,15)=fmi(img1,img2,imgf); 
% Q(kk,i,16)=metricPeilla(img1,img2,imgf,2);  %Q_W
Q(kk,i,17)=metricPeilla(img1,img2,imgf,3);  %Q_E
% Q(kk,i,18)=VIFF_Public(img1,img2,imgf);  %
% Q(kk,i,19)=vifvec(img1, imgf);  %
% Q(kk,i,20)=vifvec(img2, imgf);  %



end
end

Q_ave=sum(Q,1)/num_img
% Q_std=std(Q,1)

% save Q Q_proposed 

%xlswrite('gray.xlsx',Q_ave);


