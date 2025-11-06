clear all;
close all;
%clc;
addpath(genpath('evaluation_toolbox'));
addpath(genpath('vifvec_release'));

num_alg=5;
num_metric=4;
num_img=25;
Q=zeros(num_img,num_alg,num_metric);

for kk=1:num_img
    disp(kk)
    h1=floor(kk/100);
    h2=floor((kk-h1*100)/10);
    h3=mod(kk,10);
    
    N2=floor((kk)/10);
    N3=mod((kk),10);
    name1=['G:\dataset\multifocus\haveGT\ver\renum\A\A_' num2str(h1) num2str(h2) num2str(h3) '.bmp'];
    name2=['G:\dataset\multifocus\haveGT\ver\renum\B\B_' num2str(h1) num2str(h2) num2str(h3) '.bmp'];
    namef=cell(1,num_alg);
    
%     namef{1}=['F:\thinkingsFile\dataset\multi-focus\MFIF-f0f0e0725e7d82fe91aa30eb6296d8241d26e4c9\fused_images\lytro_'   num2str(N2) num2str(N3) '_ASR.jpg'];
%     
%     namef{3}=['F:\thinkingsFile\dataset\multi-focus\MFIF-f0f0e0725e7d82fe91aa30eb6296d8241d26e4c9\fused_images\lytro_'   num2str(N2) num2str(N3) '_BGSC.jpg'];
%     namef{4}=['F:\thinkingsFile\dataset\multi-focus\MFIF-f0f0e0725e7d82fe91aa30eb6296d8241d26e4c9\fused_images\lytro_'   num2str(N2) num2str(N3) '_CBF.jpg'];
%     namef{6}=['F:\thinkingsFile\dataset\multi-focus\MFIF-f0f0e0725e7d82fe91aa30eb6296d8241d26e4c9\fused_images\lytro_'   num2str(N2) num2str(N3) '_CSR.jpg'];
%     namef{7}=['F:\thinkingsFile\dataset\multi-focus\MFIF-f0f0e0725e7d82fe91aa30eb6296d8241d26e4c9\fused_images\lytro_'   num2str(N2) num2str(N3) '_DCT_Corr.jpg'];
%     namef{8}=['F:\thinkingsFile\dataset\multi-focus\MFIF-f0f0e0725e7d82fe91aa30eb6296d8241d26e4c9\fused_images\lytro_'   num2str(N2) num2str(N3) '_DCT_EOL.jpg'];
%     namef{9}=['F:\thinkingsFile\dataset\multi-focus\MFIF-f0f0e0725e7d82fe91aa30eb6296d8241d26e4c9\fused_images\lytro_'   num2str(N2) num2str(N3) '_DRPL.jpg'];
%     namef{10}=['F:\thinkingsFile\dataset\multi-focus\MFIF-f0f0e0725e7d82fe91aa30eb6296d8241d26e4c9\fused_images\lytro_'   num2str(N2) num2str(N3) '_DSIFT.jpg'];
%     namef{11}=['F:\thinkingsFile\dataset\multi-focus\MFIF-f0f0e0725e7d82fe91aa30eb6296d8241d26e4c9\fused_images\lytro_'   num2str(N2) num2str(N3) '_DWTDE.jpg'];
%     namef{12}=['F:\thinkingsFile\dataset\multi-focus\MFIF-f0f0e0725e7d82fe91aa30eb6296d8241d26e4c9\fused_images\lytro_'   num2str(N2) num2str(N3) '_ECNN.jpg'];
%     namef{13}=['F:\thinkingsFile\dataset\multi-focus\MFIF-f0f0e0725e7d82fe91aa30eb6296d8241d26e4c9\fused_images\lytro_'   num2str(N2) num2str(N3) '_FusionDN.jpg'];
%     namef{14}=['F:\thinkingsFile\dataset\multi-focus\MFIF-f0f0e0725e7d82fe91aa30eb6296d8241d26e4c9\fused_images\lytro_'   num2str(N2) num2str(N3) '_GCF.jpg'];
%     namef{15}=['F:\thinkingsFile\dataset\multi-focus\MFIF-f0f0e0725e7d82fe91aa30eb6296d8241d26e4c9\fused_images\lytro_'   num2str(N2) num2str(N3) '_GD.jpg'];
%    
    namef{1}=['G:\thinkingsTest\waveFormer\waveFormer\ablation\withoutWaveKeepHHFormer\models\1e-3lr_64batch_size_20epochs_1ssim_5L1_1depth_64_channel_recheckConv_orderWithoutWaveKeepHHFormer\epoch10_ver_rgb\'  num2str(h1) num2str(h2) num2str(h3) '.bmp'];
    namef{2}=['G:\thinkingsTest\waveFormer\waveFormer\ablation\withoutWaveKeepLowFormer\models\1e-3lr_64batch_size_20epochs_1ssim_5L1_1depth_64_channel_recheckConv_order_wiouthWaveKeepLowFormer\epoch10_ver_rgb\'  num2str(h1) num2str(h2) num2str(h3) '.bmp'];
     namef{3}=['G:\thinkingsTest\waveFormer\waveFormer\ablation\replaceHighFormerWithConv\models\1e-3lr_64batch_size_20epochs_1ssim_5L1_1depth_64_channel_recheckConv_order_replaceHighFormerWithConv\epoch10_ver_rgb\'  num2str(h1) num2str(h2) num2str(h3) '.bmp'];
         namef{4}=['G:\thinkingsTest\waveFormer\waveFormer\ablation\replaceLowFormerWithConv\models\1e-3lr_64batch_size_20epochs_1ssim_5L1_1depth_64_channel_recheckConv_order_replaceLowFormerWithConv\epoch10_ver_rgb\'  num2str(h1) num2str(h2) num2str(h3) '.bmp'];

         namef{5}=['G:\thinkingsTest\waveFormer\waveFormer\models\1e-3lr_64batch_size_20epochs_1ssim_5L1_1depth_64_channel_recheckConv_order\epoch10_ver_rgb\'  num2str(h1) num2str(h2) num2str(h3) '.bmp'];
%     namef{6}=['F:\thinkingsFile\dataset\multi-focus\MFIF-f0f0e0725e7d82fe91aa30eb6296d8241d26e4c9\fused_images\lytro_'   num2str(N2) num2str(N3) '_SESF.jpg'];

%     namef{18}=['F:\thinkingsFile\dataset\multi-focus\MFIF-f0f0e0725e7d82fe91aa30eb6296d8241d26e4c9\fused_images\lytro_'   num2str(N2) num2str(N3) '_IFCNN.jpg'];
%     namef{19}=['F:\thinkingsFile\dataset\multi-focus\MFIF-f0f0e0725e7d82fe91aa30eb6296d8241d26e4c9\fused_images\lytro_'   num2str(N2) num2str(N3) '_IFM.jpg'];
%     namef{20}=['F:\thinkingsFile\dataset\multi-focus\MFIF-f0f0e0725e7d82fe91aa30eb6296d8241d26e4c9\fused_images\lytro_'   num2str(N2) num2str(N3) '_MADCNN.jpg'];
%     namef{21}=['F:\thinkingsFile\dataset\multi-focus\MFIF-f0f0e0725e7d82fe91aa30eb6296d8241d26e4c9\fused_images\lytro_'   num2str(N2) num2str(N3) '_MFF-GAN.jpg'];
%     namef{22}=['F:\thinkingsFile\dataset\multi-focus\MFIF-f0f0e0725e7d82fe91aa30eb6296d8241d26e4c9\fused_images\lytro_'   num2str(N2) num2str(N3) '_MFM.jpg'];
%     namef{23}=['F:\thinkingsFile\dataset\multi-focus\MFIF-f0f0e0725e7d82fe91aa30eb6296d8241d26e4c9\fused_images\lytro_'   num2str(N2) num2str(N3) '_MGFF.jpg'];
%     namef{24}=['F:\thinkingsFile\dataset\multi-focus\MFIF-f0f0e0725e7d82fe91aa30eb6296d8241d26e4c9\fused_images\lytro_'   num2str(N2) num2str(N3) '_MST_SR.jpg'];
%     namef{25}=['F:\thinkingsFile\dataset\multi-focus\MFIF-f0f0e0725e7d82fe91aa30eb6296d8241d26e4c9\fused_images\lytro_'   num2str(N2) num2str(N3) '_MWGF.jpg'];
%     namef{26}=['F:\thinkingsFile\dataset\multi-focus\MFIF-f0f0e0725e7d82fe91aa30eb6296d8241d26e4c9\fused_images\lytro_'   num2str(N2) num2str(N3) '_NSCT_SR.jpg'];
%     namef{27}=['F:\thinkingsFile\dataset\multi-focus\MFIF-f0f0e0725e7d82fe91aa30eb6296d8241d26e4c9\fused_images\lytro_'   num2str(N2) num2str(N3) '_PCANet.jpg'];
%     namef{28}=['F:\thinkingsFile\dataset\multi-focus\MFIF-f0f0e0725e7d82fe91aa30eb6296d8241d26e4c9\fused_images\lytro_'   num2str(N2) num2str(N3) '_PMGI.jpg'];
%     namef{29}=['F:\thinkingsFile\dataset\multi-focus\MFIF-f0f0e0725e7d82fe91aa30eb6296d8241d26e4c9\fused_images\lytro_'   num2str(N2) num2str(N3) '_QB.jpg'];
%     namef{30}=['F:\thinkingsFile\dataset\multi-focus\MFIF-f0f0e0725e7d82fe91aa30eb6296d8241d26e4c9\fused_images\lytro_'   num2str(N2) num2str(N3) '_RP_SR.jpg'];
%     namef{32}=['F:\thinkingsFile\dataset\multi-focus\MFIF-f0f0e0725e7d82fe91aa30eb6296d8241d26e4c9\fused_images\lytro_'   num2str(N2) num2str(N3) '_SFMD.jpg'];
%     namef{33}=['F:\thinkingsFile\dataset\multi-focus\MFIF-f0f0e0725e7d82fe91aa30eb6296d8241d26e4c9\fused_images\lytro_'   num2str(N2) num2str(N3) '_SVDDCT.jpg'];
%     namef{35}=['F:\thinkingsFile\dataset\multi-focus\MFIF-f0f0e0725e7d82fe91aa30eb6296d8241d26e4c9\fused_images\lytro_'   num2str(N2) num2str(N3) '_U2Fusion.jpg'];
    
    
    
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
%         Q(kk,i,19)=vifvec(img1, imgf);  %
%         Q(kk,i,20)=vifvec(img2, imgf);  %
%         Q(kk,i,21) = analysis_SCD(img1,img2,imgf);%   SCD   RFN-Nest
        Q(kk,i,22) = SD_evaluation(imgf);% SD  SeAFusion
%         Q(kk,i,23) =AG_evaluation(imgf);% AG  SeAFusion
%         Q(kk,i,24) = CC_evaluation(img1,img2,imgf);% CC  SeAFusion
%         Q(kk,i,25)  = PSNR_evaluation(img1,img2,imgf);% PSNR  SeAFusion
%         %                 Q(kk,i,26)  = MSSSIM2image(img1,img2,imgf);% MS-SSIM
%         Q(kk,i,27)  = VIF(img1,img2,imgf);% VIF_average
        %          Q(kk,i,28)  = MS_SSIM(img1,img2,imgf);% multi-exposure
        
        
        
        
    end
end

Q_ave=sum(Q,1)/num_img;
Q_std=std(Q,1);

save Q Q
save Q_ave
save Q_std

%xlswrite('gray.xlsx',Q_ave);


