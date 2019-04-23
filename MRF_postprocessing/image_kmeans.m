%%  kmeans algorithm for an image
%   Y: 2D image
%   k: number of clusters
%   X: 2D label image
%   mu: vector of means of clusters
%   sigma: vector of standard deviations of clusters

function [y1,y2,y3,IX,maxX,maxY,IM,IMMM]=image_kmeans(Y)
k = 3;
IM=double(Y);
y=Y(:);
x=kmeans(y,k);
X=reshape(x,size(Y));
[maxX,maxY]=size(X);

num1=0;
num2=0;
num3=0;
for i=1:maxX
    for j=1:maxY
        if X(i,j)==1
            num1=num1+1;
            y1(num1)=IM(i,j);
            IX(i,j)=1;
        elseif X(i,j)==2
            num2=num2+1;
            y2(num2)=IM(i,j);
            IX(i,j)=2;
        else
            num3=num3+1;
            y3(num3)=IM(i,j);
            IX(i,j)=3;
        end
    end
end

IMMM=zeros(maxX,maxY);
for i=1:maxX
    for j=1:maxY
        if IX(i,j)==3
            IMMM(i,j)=255;
        elseif IX(i,j)==2
            IMMM(i,j)=120;                                  
        else
            IMMM(i,j)=0;
        end
    end
end
IMMM=uint8(IMMM);
figure(2);imshow(IMMM);title('Initial segmentation');