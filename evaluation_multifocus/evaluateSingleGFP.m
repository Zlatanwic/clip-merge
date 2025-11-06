clear all;
close all;
%clc;
addpath(genpath('evaluation_toolbox'));
addpath(genpath('vifvec_release'));

num_alg=1;
num_metric=20;
num_img=18;
Q=zeros(num_img,num_alg,num_metric);

for kk=1:num_img
    disp(kk)
    h1=floor(kk/100);
    h2=floor((kk-h1*100)/10);
    h3=mod(kk,10);
    L1=floor((kk-1)/100);
    L2=floor(((kk-1)-L1*100)/10);
    L3=mod((kk-1),10);
    N1=floor((kk+130)/100);
    N2=floor(((kk+130)-N1*100)/10);
    N3=mod((kk+130),10);
    name1=['F:\thinkingsFile\dataset\GFP_PC\test_GFPrenum\GFP_' num2str(h1) num2str(h2) num2str(h3) '.bmp']; 
    name2=['F:\thinkingsFile\dataset\GFP_PC\test_PCIremun\PCI_' num2str(h1) num2str(h2) num2str(h3) '.bmp'];
%     name1=['F:\thinkingsFile\dataset\infrared and visible image fusion\ALL\test\ir\IR_' num2str(h1) num2str(h2) num2str(h3) '.bmp']; 
%     name2=['F:\thinkingsFile\dataset\infrared and visible image fusion\ALL\test\vi\VIS_' num2str(h1) num2str(h2) num2str(h3) '.bmp'];
%     name1=['F:\thinkingsFile\dataset\infrared and visible image fusion\verification\ir\IR_' num2str(h1) num2str(h2) num2str(h3) '.bmp']; 
%     name2=['F:\thinkingsFile\dataset\infrared and visible image fusion\verification\vis\VIS_' num2str(h1) num2str(h2) num2str(h3) '.bmp'];

%     namef=cell(1,num_alg);
    
%     namef=['F:\thinkingsFile\thinkingsTest\fusionTransformer\code\20211026\models\ReNet_1e-4withoutWeightDecay_128_60_1ir_1vis_100ssimVIS_100ssimIR_3depthDARM\epoch60\'  num2str(h1) num2str(h2) num2str(h3) '.bmp'];

%     namef=['F:\thinkingsFile\thinkingsTest\comparison\infrared and visible\test\STDFusion\TNO\'   num2str(h2) num2str(h3) '.bmp'];
    
% %     
%     namef=['F:\thinkingsFile\thinkingsTest\comparison\infrared and visible\test\SR\RoadScene\SR_' num2str(L2) num2str(L3) '.bmp'];
% % %     
    namef=['F:\thinkingsFile\COMPARISON\GFP_PC\RESULT\MSTR\'  num2str(h1) num2str(h2) num2str(h3)  '.bmp'];
% 
%     namef{1}=['F:\thinkingsFile\thinkingsTest\fusionTransformer\code\20211026\models\120_1_ReNet_1e-4withoutWeightDecay_128_60_1ir_1vis_ssimVIS_ssimIR_3depthDARM\epoch10\'   num2str(h1) num2str(h2) num2str(h3) '.bmp'];
%   
%     namef{1}=['F:\thinkingsFile\thinkingsTest\comparison\infrared and visible\verification\CNN\CNN_' num2str(kk) '.bmp'];
%     namef{2}=['F:\thinkingsFile\thinkingsTest\comparison\infrared and visible\verification\CVT\CVT_'  num2str(h1) num2str(h2) num2str(h3) '.bmp'];
%     namef{4}=['F:\thinkingsFile\thinkingsTest\comparison\infrared and visible\verification\SR\SR_'  num2str(kk) '.bmp'];
%     namef=['F:\thinkingsFile\thinkingsTest\fusionTransformer\code\20211008\models\ReNet_1e-5withoutWeightDecay_128_120_5ir_5vis_1ssimVIS_1ssimIR_2depth\epoch70\' num2str(h1) num2str(h2) num2str(h3) '.bmp'];
%     namef{2}=['F:\thinkingsFile\thinkingsTest\comparison\infrared and visible\verification\metaLearning\' num2str(kk) '.bmp'];
%     namef{1}=['F:\thinkingsFile\thinkingsTest\fusionTransformer\code\20211008\models\ReNet_1e-5withoutWeightDecay_128_120_ir_vis_1ssimVIS_1ssimIR_2depth\epoch30\' num2str(h1) num2str(h2) num2str(h3) '.bmp'];
%     namef{2}=['F:\thinkingsFile\thinkingsTest\fusionTransformer\code\20211008\models\ReNet_1e-5withoutWeightDecay_128_120_ir_vis_1ssimVIS_1ssimIR_2depth\epoch60\' num2str(h1) num2str(h2) num2str(h3) '.bmp'];
%     namef{3}=['F:\thinkingsFile\thinkingsTest\fusionTransformer\code\20211008\models\ReNet_1e-5withoutWeightDecay_128_120_ir_vis_1ssimVIS_1ssimIR_2depth\epoch100\' num2str(h1) num2str(h2) num2str(h3) '.bmp'];
%   
    for i=1:num_alg
        
        A=imread(name1);B=imread(name2);
%         F=imread(namef{i});
        F=imread(namef);
        A=rgb2gray(A);
%         B=rgb2gray(B);
        F=rgb2gray(F);
       
        img1=double(A);img2=double(B);imgf=double(F);
        [H W]=size(imgf);I=zeros(H,W,1);I(:,:,1)=imgf;
%         I(:,:,2)=img2;
        
        
        
        % %Information Theory-Based Metrics
        Q(kk,i,1)=metricMI(img1,img2,imgf,1);%% normalized mutual informtion $Q_{MI}$
        Q(kk,i,2)=metricMI(img1,img2,imgf,3);% Tsallis entropy $Q_{TE}$
        Q(kk,i,3)=metricWang(img1,img2,imgf); % Wang - NCIE $Q_{NCIE}$
        % %Image Feature-Based Metrics
        
        Q(kk,i,4)=metricXydeas(img1,img2,imgf);% Xydeas $Q_G$
        % Q(kk,i,4)=edge_association(img1,img2,imgf);% Xydeas $Q_G$ by Qu Xiaobo
        Q(kk,i,5)=metricPWW(img1,img2,imgf);% PWW $Q_M$
        Q(kk,i,6)=metricZheng(img1,img2,imgf);   %越接近0越好 %Yufeng Zheng (spatial frequency) $Q_{SF}$
        Q(kk,i,7)=metricZhao(img1,img2,imgf);% Zhao (phase congrency) $Q_P$
        % Image Structural Similarity-Based Metrics
        Q(kk,i,8)=metricPeilla(img1,img2,imgf,1);% Piella  (need to select only one) $Q_s$
        Q(kk,i,9)=metricCvejic(img1,img2,imgf,2);% Cvejie $Q_C$
        Q(kk,i,10)=metricYang(img1,img2,imgf);% Yang $Q_Y$
        % % %Human Perception Inspired Fusion Metrics
        Q(kk,i,11)=metricChen(img1,img2,imgf)*0.1;   % 越小越好Qcv % Chen-Varshney $Q_{CV}$
        Q(kk,i,12)=metricChenBlum(img1,img2,imgf);  % Chen-Blum $Q_{CB}$
%         % % %%%%%%%%%%%%%%
        Q(kk,i,13)=entropy_fusion(imgf,256);
        Q(kk,i,14)=LMI(img1,img2,imgf,0);
        Q(kk,i,15)=fmi(img1,img2,imgf);
        Q(kk,i,16)=metricPeilla(img1,img2,imgf,2);  %Q_W
        Q(kk,i,17)=metricPeilla(img1,img2,imgf,3);  %Q_E
        Q(kk,i,18)=VIFF_Public(img1,img2,imgf);  %
        Q(kk,i,19)=vifvec(img1, imgf);  %
        Q(kk,i,20)=vifvec(img2, imgf);  %
        
        
        
    end
end

Q_ave=sum(Q,1)/num_img;
Q_std=std(Q,1);

save Q Q
save Q_ave
save Q_std

%xlswrite('gray.xlsx',Q_ave);


