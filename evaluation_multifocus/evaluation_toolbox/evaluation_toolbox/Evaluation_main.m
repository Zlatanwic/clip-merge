
close all
%clc
clear all


name1='../disk1.gif';
name2='../disk2.gif';
namer='../disk_groundtruth.gif';
%namef='../disk_DWT1.gif';
%namef='../disk_DWT2.gif';
%namef='../disk_NSCT.gif';
%namef='../disk_nsctpcnn.gif';
namef='../disk_nsctsfpcnn.gif';
%namef='../disk_blockDE.gif';
%namef='../disk_128.gif';
%namef='../disk_256.gif';
%namef='../disk_128_adaptive.gif';
% namef='../disk_blockDE_map1.gif';
%namef='../disk_blockDE_map2.gif';
img1=imread(name1);
img2=imread(name2);
imgr=imread(namer);
imgf=imread(namef);

[rmse psnr]=PSNR(imgf,imgr);
disp(sprintf('RMSE=%f ',rmse))
disp(sprintf('PSNR=%f ',psnr))
 
[mssim ssim_map] = ssim(imgf,imgr);
disp(sprintf('SSIM=%f ',mssim))
 
x=[8 8];
a=0.5;
[q_abf,qw_abf]=Qabf(img1,img2,imgf,x);
qe_abf=Qeabf(img1,img2,imgf,x,a);
disp(sprintf('Q=%f ',q_abf))
disp(sprintf('Qw=%f ',qw_abf))
disp(sprintf('Qe=%f ',qe_abf))

grey_level=256;
mutural_information=mutural_information(img1,img2,imgf,grey_level);
disp(sprintf('MI=%f ',mutural_information))

result=edge_association(img1,img2,imgf);
disp(sprintf('Qab/f=%f ',result))

% Qabf_X=Qabf_xy(name1,name2,namef);
% disp(sprintf('Qab/f=%f ',Qabf_X))

% [SF,EOG,SML]=spatial_clarity(imgf);
% disp(sprintf('SF=%f ',SF))
% disp(sprintf('EOG=%f ',EOG))
% disp(sprintf('SML=%f ',SML))
% figure,imshow(imgf);
% figure,imshow(double(imgf)-double(img1),[])
% figure,imshow(double(imgf)-double(img2),[])