%逐点对像素进行分类
function [IX,times,IMMM]=BianXiangsuLeibie2(maxX,maxY,StruInfo,gauss,IX,~)    
 %输入：图像size，先验概率StruInfo，高斯分布概率gauss，图像类别矩阵IX
 %输出：图像新类别矩阵IX，图像分类结果：IMMM
times=0;
IX1=zeros(maxX,maxY);
temp1=StruInfo.*gauss;
for i=1:maxX
    for j=1:maxY
        tmp=max(temp1(i,j,:));%取最大概率
        for k=1:3
            if tmp==temp1(i,j,k)
                IX1(i,j)=k;
            end
        end
        if IX(i,j)==IX1(i,j)
            IX(i,j)=IX1(i,j);
        else
            times=times+1;
            IX(i,j)=IX1(i,j);                       %取xij概率最大者
        end
    end
end
 
IMMM=zeros(maxX,maxY);
for i=1:maxX
    for j=1:maxY
        if IX(i,j)==3
            IMMM(i,j)=255;
        elseif IX(i,j)==2
            IMMM(i,j)=120;                           %由类别分象素
        else
            IMMM(i,j)=0;
        end
    end
end
IMMM=uint8(IMMM);
figure(3);
imshow(IMMM);