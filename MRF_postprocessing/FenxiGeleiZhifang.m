function [y1,y2,y3]=FenxiGeleiZhifang(IX,IM,maxX,maxY)
%���룺ͼ��IM��ͼ��������IX��ͼ���СmaxX��maxY
%������������ֱ��Ӧ����ʵ����ֵ��˳��ΪIX������������˳��
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
