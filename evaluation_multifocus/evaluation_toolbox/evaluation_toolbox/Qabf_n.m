function [ y ] = Qabf_n( I,F )
%QABF_N I represents the input image sequence which is assumed to be grayscale image. F is the fused image.

[w,h,N]=size(I);
I=double(I)/255;
F=double(F)/255;

%Sobel operators
Yoperator=-fspecial('sobel');
Xoperator=Yoperator';

%Initinalization
GX=zeros(w,h,N);
GY=zeros(w,h,N);
G_strength=zeros(w,h,N);
G_orientation=zeros(w,h,N);
Gf_strength=zeros(w,h);
Gf_orientation=zeros(w,h);
%gradient strength & orientation
GfX = imfilter(F,Xoperator,'same');%换成same，则最终结果变大
GfY = imfilter(F,Yoperator,'same');
GfX(GfX==0)=0.00000001;
GfY(GfY==0)=0.00000001;
Gf_strength=sqrt(GfX.^2+GfY.^2);
Gf_orientation=atan(GfY./GfX);
for n=1:N
GX(:,:,n) = imfilter(I(:,:,n),Xoperator,'same');
GY(:,:,n) = imfilter(I(:,:,n),Yoperator,'same');
GX(GX==0)=0.00000001;
GY(GY==0)=0.00000001;
G_strength(:,:,n)= sqrt(GX(:,:,n).^2+GY(:,:,n).^2);
G_orientation(:,:,n)=atan(GY(:,:,n)./GX(:,:,n));
G_f_str(:,:,n)=(Gf_strength./G_strength(:,:,n)).^(double(Gf_strength<=G_strength(:,:,n))-double(Gf_strength>G_strength(:,:,n)));
G_f_ori(:,:,n)=1-2*abs(G_orientation(:,:,n)-Gf_orientation)./pi;
end

%new Initinization
F_g=0.9994;k_g=-15;o_g=0.5;
F_a=0.9879;k_a=-22;o_a=0.8;
L=1;
for n=1:N
Qf(:,:,n)=F_g*F_a*((1+exp(k_g*(G_f_str(:,:,n)-o_g))).^(-1)).*((1+exp(k_a*(G_f_ori(:,:,n)-o_a))).^(-1));

W(:,:,n)=G_strength(:,:,n).^L;
end

y=sum(Qf(:).*W(:))/sum(W(:));

end

