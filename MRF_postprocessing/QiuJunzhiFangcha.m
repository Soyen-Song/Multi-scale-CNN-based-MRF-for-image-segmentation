%求高斯分布后验概率
function [gauss]=QiuJunzhiFangcha(y1,y2,y3,IM,maxX,maxY)
 %输入：三种类别的真实像素值向量，图像IM，图像size
 %输出：cell矩阵gauss，包含三种类别的高斯分布后验概率
type(1:3)=0;
miyu(1:3)=0;
sita(1:3)=0;
gauss=zeros(maxX,maxY,3);
sq2Pi=sqrt(2*3.1415926);
 
miyu(1)=mean(y1);
miyu(2)=mean(y2);
miyu(3)=mean(y3);
 
sita(1)=std(y1);
sita(2)=std(y2);
sita(3)=std(y3);

sita2=2*(sita.*sita);
 
for i=1:maxX
    for j=1:maxY
        for k=1:3
            gauss(i,j,k)=exp(-(IM(i,j)-miyu(k))*(IM(i,j)-miyu(k))/(sita2(k)))/(sq2Pi*sita(k));    %高斯概率
        end
    end
end
 