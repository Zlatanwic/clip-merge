function evaluation(A,B,img1,img2,imgf)

[h w]=size(img1);I=zeros(h,w,2);I(:,:,1)=img1;I(:,:,2)=img2;
%spatial_clarity
[SF,EOG,SML]=spatial_clarity(imgf);
disp(sprintf('SF=%f ',SF))
disp(sprintf('EOG=%f ',EOG))
disp(sprintf('SML=%f ',SML))

%mutural information  input images should be uint8,not double
grey_level=256; %Qu 
mutural_information=mutural_information(A,B,uint8(imgf),grey_level);
disp(sprintf('MI=%f ',mutural_information))

MI=MI_n(uint8(I),uint8(imgf));  %Kang 
disp(sprintf('MI=%f ',MI))


%Qabf
Qabf=edge_association(img1,img2,imgf);  %Qu
disp(sprintf('Qabf=%f ',Qabf))

Qabf=Qabf_n(I,imgf);  %Kang
disp(sprintf('Qabf=%f ',Qabf))

Qabf=Qabf_xy(img1,img2,imgf);  %Hong
disp(sprintf('Qabf=%f ',Qabf))


%Q0,Qw.Qe                 
Q0=Q0_n(I,imgf);               %Kang
disp(sprintf('Q0=%f ',Q0))
Qw_1=Qw_n(I,imgf);
disp(sprintf('Qw=%f ',Qw))
Qe=Qe_n(I,imgf,0.5);
disp(sprintf('Qe=%f ',Qe))

