%�������ؽ��з���
function [IX,times,IMMM]=BianXiangsuLeibie2(maxX,maxY,StruInfo,gauss,IX,~)    
 %���룺ͼ��size���������StruInfo����˹�ֲ�����gauss��ͼ��������IX
 %�����ͼ����������IX��ͼ���������IMMM
times=0;
IX1=zeros(maxX,maxY);
temp1=StruInfo.*gauss;
for i=1:maxX
    for j=1:maxY
        tmp=max(temp1(i,j,:));%ȡ������
        for k=1:3
            if tmp==temp1(i,j,k)
                IX1(i,j)=k;
            end
        end
        if IX(i,j)==IX1(i,j)
            IX(i,j)=IX1(i,j);
        else
            times=times+1;
            IX(i,j)=IX1(i,j);                       %ȡxij���������
        end
    end
end
 
IMMM=zeros(maxX,maxY);
for i=1:maxX
    for j=1:maxY
        if IX(i,j)==3
            IMMM(i,j)=255;
        elseif IX(i,j)==2
            IMMM(i,j)=120;                           %����������
        else
            IMMM(i,j)=0;
        end
    end
end
IMMM=uint8(IMMM);
figure(3);
imshow(IMMM);