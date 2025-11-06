function [ y ] = Qw_n( I,F )
[r,c,N]=size(I);
A=double(I)/255;
F=double(F)/255;


bloF = im2col(F,[16,16],'distinct');
meanF=mean(bloF);meanF(meanF==0)=0.1^25;
varF=var(bloF);varF(varF==0)=0.1^25;
for i=1:N
bloA(:,:,i) = im2col(A(:,:,i),[16,16],'distinct');   % 'sliding'
meanA(i,:)=mean(bloA(:,:,i));


varA(i,:)=var(bloA(:,:,i));

end
meanA(meanA==0)=0.1^25;
varA(varA==0)=0.1^25;
S_a=varA;
C=max(S_a,[],1);
c=C./sum(C);
for i=1:N
lanmuda(i,:)=S_a(i,:)./(sum(S_a,1));
end

for i=1:N
covarianceAF(i,:)=1/(size(bloA(:,:,i),1)-1)*sum((bloA(:,:,i)-ones(size(bloA(:,:,i),1),1)*meanA(i,:)).*(bloF-ones(size(bloF,1),1)*meanF));
end
for i=1:N
Q_0_AF_matrix(i,:)=(covarianceAF(i,:)./sqrt(varA(i,:).*varF)).*((2.*meanA(i,:).*meanF)./(meanA(i,:).^2+meanF.^2)).*((2.*varA(i,:).*varF)./(varA(i,:).^2+varF.^2));
end

y=sum(c.*(sum(Q_0_AF_matrix.*lanmuda)));
end
