%通过硬c-均值聚类进行图像初始分割
function [y1,y2,y3,IX,maxX,maxY,IM]=initial(elm,IM)

IM=uint8(IM);
% IM=IM(2:end-1,2:end-1,1);
figure(1);
% imshow(IM);
IM=double(IM);
[maxX,maxY]=size(elm); 
 y1 = 0;
 y2 = 0;
 y3 = 0;

    num1=0;
    num2=0;
    num3=0;
    for i=1:maxX
        for j=1:maxY
            if elm(i,j) == 1
                num1=num1+1;
                y1(num1)=IM(i,j);
                IX(i,j)=1;
            elseif elm(i,j) == 2
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
figure(2);imshow(IMMM);
% title('初始分割');
 
l=0;