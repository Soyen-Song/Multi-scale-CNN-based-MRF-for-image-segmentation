function [y1,y2,y3]=FenxiGeleiZhifang(IX,IM,maxX,maxY)
%输入：图像IM，图像类别矩阵IX，图像大小maxX、maxY
%输出：三种类别分别对应的真实像素值，顺序为IX中类别的行排列顺序
num1=0;
num2=0;
num3=0;
for i=1:maxX
    for j=1:maxY
        if IX(i,j)==1
            num1=num1+1;
            y1(num1)=IM(i,j);
        elseif IX(i,j)==2
            num2=num2+1;
            y2(num2)=IM(i,j);
        else
            num3=num3+1;
            y3(num3)=IM(i,j);
        end
    end
end
