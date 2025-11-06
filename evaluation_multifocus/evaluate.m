clear all;
close all;
%clc;
addpath(genpath('evaluation_toolbox'));
addpath(genpath('vifvec_release'));

num_alg=9;
num_metric=4;
num_img=40;
Q=zeros(num_img,num_alg,num_metric);

for kk=1:num_img
    disp(kk)
    h1=floor(kk/100);
    h2=floor((kk-h1*100)/10);
    h3=mod(kk,10);
  
    N2=floor((kk)/10);
    N3=mod((kk),10);
    name1=['G:\dataset\multifocus\haveGT\test\renum\A\A_' num2str(h1) num2str(h2) num2str(h3) '.bmp']; 
    name2=['G:\dataset\multifocus\haveGT\test\renum\B\B_' num2str(h1) num2str(h2) num2str(h3) '.bmp'];
    namef=cell(1,num_alg);
				
 
  		
		

						
		
			
%     namef{2}=['G:\COMPARISON\multi_focus\result\test\BGSC\rgb\BGSC_'  num2str(h1) num2str(h2) num2str(h3) '.bmp'];
%         namef{3}=['G:\COMPARISON\multi_focus\result\test\CBF\rgb\CBF_'  num2str(h1) num2str(h2) num2str(h3) '.bmp'];
%     namef{5}=['G:\COMPARISON\multi_focus\result\test\CSR\rgb\CSR_'  num2str(h1) num2str(h2) num2str(h3) '.bmp'];
%     namef{6}=['G:\COMPARISON\multi_focus\result\test\DTC_Coor\rgb\'  num2str(h1) num2str(h2) num2str(h3) '.bmp'];
%     namef{7}=['G:\COMPARISON\multi_focus\result\test\DCT_EOL\rgb\DCT_EOL_'  num2str(h1) num2str(h2) num2str(h3) '.bmp'];
%     namef{8}=['G:\COMPARISON\multi_focus\result\test\DSIFT\rgb\DSIFT_'  num2str(h1) num2str(h2) num2str(h3) '.bmp'];
%     namef{9}=['G:\COMPARISON\multi_focus\result\test\DWTDE\rgb\DWTDE_'  num2str(h1) num2str(h2) num2str(h3) '.bmp'];
%     namef{10}=['G:\COMPARISON\multi_focus\result\test\GD\rgb\GD_'  num2str(h1) num2str(h2) num2str(h3) '.bmp'];
    namef{1}=['G:\COMPARISON\multi_focus\result\test\GFF\rgb\GFF_'  num2str(h1) num2str(h2) num2str(h3) '.bmp'];
        namef{2}=['G:\COMPARISON\multi_focus\result\test\BFMF_rgb\BFMF_'   num2str(h1) num2str(h2) num2str(h3) '.bmp'];
    namef{3}=['G:\COMPARISON\multi_focus\result\test\GFDF\rgb\GFDF_'  num2str(h1) num2str(h2) num2str(h3) '.bmp'];
    namef{4}=['G:\COMPARISON\multi_focus\result\test\TF\rgb\'  num2str(h1) num2str(h2) num2str(h3) '.bmp'];
    namef{5}=['G:\COMPARISON\multi_focus\result\test\CNN\rgb\CNN_'  num2str(h1) num2str(h2) num2str(h3) '.bmp'];
    namef{6}=['G:\COMPARISON\multi_focus\result\test\SESF\rgb\'  num2str(h1) num2str(h2) num2str(h3) '.bmp'];
    namef{7}=['G:\COMPARISON\multi_focus\result\test\GACN\rgb\'  num2str(h1) num2str(h2) num2str(h3) '.bmp'];
    namef{8}=['G:\COMPARISON\multi_focus\result\test\SwinFusion\rgb\A_'  num2str(h1) num2str(h2) num2str(h3) '.bmp'];
    namef{9}=['G:\thinkingsTest\waveFormer\code_2023_02_14_pm\models\1e-3lr_64batch_size_20epochs_1ssim_5L1_1depth_64_channel_recheckConv_order\epoch10_rgb\'  num2str(h1) num2str(h2) num2str(h3) '.bmp'];

%     namef{13}=['G:\COMPARISON\multi_focus\result\test\IFCNN\'  num2str(h1) num2str(h2) num2str(h3) '.bmp'];
%     namef{14}=['G:\COMPARISON\multi_focus\result\test\IFM\y\IFM_'  num2str(h1) num2str(h2) num2str(h3) '.bmp'];
%     namef{15}=['G:\COMPARISON\multi_focus\result\test\MFM\MFM_'  num2str(h1) num2str(h2) num2str(h3) '.bmp'];
%     namef{16}=['G:\COMPARISON\multi_focus\result\test\MGFF\'  num2str(h1) num2str(h2) num2str(h3) '.bmp'];
%     namef{17}=['G:\COMPARISON\multi_focus\result\test\PCANet\rgb\PCANet_'  num2str(h1) num2str(h2) num2str(h3) '.bmp'];
%     namef{18}=['G:\COMPARISON\multi_focus\result\test\RP-SR\rgb\'  num2str(h1) num2str(h2) num2str(h3) '.bmp'];
%     namef{20}=['G:\COMPARISON\multi_focus\result\test\SFMD\rgb\SFMD_'  num2str(h1) num2str(h2) num2str(h3) '.bmp'];
%     namef{21}=['G:\COMPARISON\multi_focus\result\test\SVDDCT\rgb\'  num2str(h1) num2str(h2) num2str(h3) '.bmp'];
%     namef{23}=[''  num2str(h1) num2str(h2) num2str(h3) '.bmp'];
%     namef{24}=[''  num2str(h1) num2str(h2) num2str(h3) '.bmp'];

    
    for i=1:num_alg
        
        A=imread(name1);B=imread(name2);
        F=imread(namef{i});
        %         F=imread(namef);
        A=rgb2gray(A);
        B=rgb2gray(B);
        F=rgb2gray(F);
        
        img1=double(A);img2=double(B);imgf=double(F);
        %         [H W]=size(imgf);I=zeros(H,W,1);I(:,:,1)=imgf;
        %         I(:,:,2)=img2;
        
        
        
        % %Information Theory-Based Metrics
%         Q(kk,i,1)=metricMI(img1,img2,imgf,1);%% normalized mutual informtion $Q_{MI}$
        Q(kk,i,2)=metricMI(img1,img2,imgf,3);% Tsallis entropy $Q_{TE}$
%         Q(kk,i,3)=metricWang(img1,img2,imgf); % Wang - NCIE $Q_{NCIE}$
%         % %Image Feature-Based Metrics
%         
%         Q(kk,i,4)=metricXydeas(img1,img2,imgf);% Xydeas $Q_G$
%         % Q(kk,i,4)=edge_association(img1,img2,imgf);% Xydeas $Q_G$ by Qu Xiaobo
%         Q(kk,i,5)=metricPWW(img1,img2,imgf);% PWW $Q_M$
%         Q(kk,i,6)=metricZheng(img1,img2,imgf);   %越接近0越好 %Yufeng Zheng (spatial frequency) $Q_{SF}$
%         Q(kk,i,7)=metricZhao(img1,img2,imgf);% Zhao (phase congrency) $Q_P$
%         % Image Structural Similarity-Based Metrics
%         Q(kk,i,8)=metricPeilla(img1,img2,imgf,1);% Piella  (need to select only one) $Q_s$
%         Q(kk,i,9)=metricCvejic(img1,img2,imgf,2);% Cvejie $Q_C$
%         Q(kk,i,10)=metricYang(img1,img2,imgf);% Yang $Q_Y$
%         % % %Human Perception Inspired Fusion Metrics
%         Q(kk,i,11)=metricChen(img1,img2,imgf)*0.1;   % 越小越好Qcv % Chen-Varshney $Q_{CV}$
%         Q(kk,i,12)=metricChenBlum(img1,img2,imgf);  % Chen-Blum $Q_{CB}$
%         %         % % %%%%%%%%%%%%%%
        Q(kk,i,13)=entropy_fusion(imgf,256);
%         Q(kk,i,14)=LMI(img1,img2,imgf,0);
%         Q(kk,i,15)=fmi(img1,img2,imgf);
%         Q(kk,i,16)=metricPeilla(img1,img2,imgf,2);  %Q_W
%         Q(kk,i,17)=metricPeilla(img1,img2,imgf,3);  %Q_E
        Q(kk,i,18)=VIFF_Public(img1,img2,imgf);  %
%         %         Q(kk,i,19)=vifvec(img1, imgf);  %
%         %         Q(kk,i,20)=vifvec(img2, imgf);  %
%         
%                 Q(kk,i,21) = analysis_SCD(img1,img2,imgf);%   SCD   RFN-Nest
        Q(kk,i,22) = SD_evaluation(imgf);% SD  SeAFusion
%         Q(kk,i,23) =AG_evaluation(imgf);% AG  SeAFusion
%         Q(kk,i,24) = CC_evaluation(img1,img2,imgf);% CC  SeAFusion
%        
%         Q(kk,i,25)  = PSNR_evaluation(img1,img2,imgf);% PSNR  SeAFusion
%                 Q(kk,i,26)  = MSSSIM2image(img1,img2,imgf);% MS-SSIM
%                 Q(kk,i,27)  = VIF(img1,img2,imgf);% VIF_average
%                 Q(kk,i,28)  = MS_SSIM(img1,img2,imgf);% multi-exposure
%  Q(kk,i,29)  =  Edge_Intensity(imgf); % 边缘强度 EI  yuFu
%  Q(kk,i,30)  =OverallCrossEntropy(img1,img2,imgf); % 总体平均交叉熵     yuFu
%  Q(kk,i,31)  = analysis_fmi(img1,img2,imgf,'wavelet');%FMI_w      yuFu
%  Q(kk,i,32)  = analysis_fmi(img1,img2,imgf,'dct');%FMI_dct       yuFu
% Q(kk,i,33) = analysis_nabf(img1,img2,imgf); %Nabf    yuFu
% Q(kk,i,34)  = Definition(imgf); % 清晰度 DF   yuFu
% Q(kk,i,35) = MIabf(img1,img2, uint8(imgf)); % MI2   yuFu
%         Q(kk,i,36) = metricsCross_entropy(img1,img2,imgf); %  CE   Xingchen Zhang
%         Q(kk,i,37) = metricsNMI(img1,img2,imgf); %  NMI   Xingchen Zhang
%                  Q(kk,i,38) = MSE_evaluation(img1,img2,imgf);  % https://github.com/Linfeng-Tang/Image-Fusion


        
    end
end

Q_ave=sum(Q,1)/num_img;
Q_std=std(Q,1);

save Q Q
save Q_ave
save Q_std

%xlswrite('gray.xlsx',Q_ave);


