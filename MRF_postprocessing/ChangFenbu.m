%由邻域信息求先验概率
function [StruInfo]=ChangFenbu(maxX,maxY,IX,IM)
%输入：图像长宽值，图像类别矩阵IX，图像像素矩阵IM
%输出：StruInfo=cat(3,temp1,temp2,temp3)，结构体，包含三种类别的先验概率值
temp=zeros(maxX,maxY);
temp1=zeros(maxX,maxY);
temp2=zeros(maxX,maxY);
temp3=zeros(maxX,maxY);

beita=1;
myu=zeros(maxX,maxY);
xigmaMyu=zeros(maxX,maxY);

for i=1:maxX
    for j=1:maxY
        if and(i==1,j==1)
            for m=1:3
                temp(i,j)=temp(i,j)+deta(m,IX(i,j+1))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i+1,j))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i+1,j+1))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i,j))-1;
                xigmaMyu(i,j)=exp(beita*temp(i,j))+xigmaMyu(i,j);
                if m==1
                    temp1(i,j)=temp(i,j);
                end
                if m==2
                    temp2(i,j)=temp(i,j);
                end
                if m==3
                    temp3(i,j)=temp(i,j);
                end
                temp(i,j)=0;
            end
        elseif and(i==1,j==maxY)
            for m=1:3
                temp(i,j)=temp(i,j)+deta(m,IX(i,j-1))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i+1,j-1))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i+1,j))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i,j))-1;
                xigmaMyu(i,j)=exp(beita*temp(i,j))+xigmaMyu(i,j);
                if m==1
                    temp1(i,j)=temp(i,j);
                end
                if m==2
                    temp2(i,j)=temp(i,j);
                end
                if m==3
                    temp3(i,j)=temp(i,j);
                end
                temp(i,j)=0;
            end
        elseif and(i==maxX,j==1)
            for m=1:3
                temp(i,j)=temp(i,j)+deta(m,IX(i,j+1))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i-1,j+1))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i-1,j))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i,j))-1;
                xigmaMyu(i,j)=exp(beita*temp(i,j))+xigmaMyu(i,j);
                if m==1
                    temp1(i,j)=temp(i,j);
                end
                if m==2
                    temp2(i,j)=temp(i,j);
                end
                if m==3
                    temp3(i,j)=temp(i,j);
                end
                temp(i,j)=0;
            end
        elseif and(i==maxX,j==maxY)
            for m=1:3
                temp(i,j)=temp(i,j)+deta(m,IX(i,j-1))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i-1,j-1))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i-1,j))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i,j))-1;
                xigmaMyu(i,j)=exp(beita*temp(i,j))+xigmaMyu(i,j);
                if m==1
                    temp1(i,j)=temp(i,j);
                end
                if m==2
                    temp2(i,j)=temp(i,j);
                end
                if m==3
                    temp3(i,j)=temp(i,j);
                end
                temp(i,j)=0;
            end
        elseif i==1
            for m=1:3
                temp(i,j)=temp(i,j)+deta(m,IX(i,j-1))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i+1,j-1))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i+1,j))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i+1,j+1))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i,j+1))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i,j))-1;
                xigmaMyu(i,j)=exp(beita*temp(i,j))+xigmaMyu(i,j);
                if m==1
                    temp1(i,j)=temp(i,j);
                end
                if m==2
                    temp2(i,j)=temp(i,j);
                end
                if m==3
                    temp3(i,j)=temp(i,j);
                end
                temp(i,j)=0;
            end
        elseif j==1
            for m=1:3
                temp(i,j)=temp(i,j)+deta(m,IX(i-1,j))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i-1,j+1))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i,j+1))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i+1,j+1))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i+1,j))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i,j))-1;
                xigmaMyu(i,j)=exp(beita*temp(i,j))+xigmaMyu(i,j);
                if m==1
                    temp1(i,j)=temp(i,j);
                end
                if m==2
                    temp2(i,j)=temp(i,j);
                end
                if m==3
                    temp3(i,j)=temp(i,j);
                end
                temp(i,j)=0;
            end
        elseif i==maxX
            for m=1:3
                temp(i,j)=temp(i,j)+deta(m,IX(i,j-1))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i-1,j-1))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i-1,j))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i-1,j+1))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i,j+1))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i,j))-1;
                xigmaMyu(i,j)=exp(beita*temp(i,j))+xigmaMyu(i,j);
                if m==1
                    temp1(i,j)=temp(i,j);
                end
                if m==2
                    temp2(i,j)=temp(i,j);
                end
                if m==3
                    temp3(i,j)=temp(i,j);
                end
                temp(i,j)=0;
            end
        elseif j==maxY
            for m=1:3
                temp(i,j)=temp(i,j)+deta(m,IX(i-1,j))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i-1,j-1))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i,j-1))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i+1,j-1))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i+1,j))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i,j))-1;
                xigmaMyu(i,j)=exp(beita*temp(i,j))+xigmaMyu(i,j);
                if m==1
                    temp1(i,j)=temp(i,j);
                end
                if m==2
                    temp2(i,j)=temp(i,j);
                end
                if m==3
                    temp3(i,j)=temp(i,j);
                end
                temp(i,j)=0;
            end
        else
            for m=1:3
                temp(i,j)=temp(i,j)+deta(m,IX(i-1,j-1))-1;    %2*u(x);u(x)=deta(xi-xj)-1,xi取三类
                temp(i,j)=temp(i,j)+deta(m,IX(i-1,j))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i-1,j+1))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i,j-1))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i,j+1))-1;      %2*u(x);u(x)=deta(xi-xj)-1,xi取三类
                temp(i,j)=temp(i,j)+deta(m,IX(i+1,j-1))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i+1,j))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i+1,j+1))-1;
                temp(i,j)=temp(i,j)+deta(m,IX(i,j))-1;
                xigmaMyu(i,j)=exp(beita*temp(i,j))+xigmaMyu(i,j);

                if m==1
                    temp1(i,j)=temp(i,j);
                end
                if m==2
                    temp2(i,j)=temp(i,j);
                end
                if m==3
                    temp3(i,j)=temp(i,j);
                end
                temp(i,j)=0;
            end
        end
    end
end

temp1=exp(beita*temp1);
temp1=temp1./xigmaMyu;
temp2=exp(beita*temp2);
temp2=temp2./xigmaMyu;
temp3=exp(beita*temp3);
temp3=temp3./xigmaMyu;
StruInfo=cat(3,temp1,temp2,temp3);
