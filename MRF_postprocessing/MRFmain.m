clear 
close all
clc
 
tmp=imread('test.jpg'); % read raw sonar images
tmp = medfilt2(tmp); % median filtering
figure,imshow(tmp)
IM = tmp;
 
yyy=0;
times=20;
tic
% initial segmentation using k-means clustering
% [y1,y2,y3,IX,maxX,maxY,IM,IMMM]=image_kmeans(IM);

% segmentation result given by CNN / MSCNN
load AugMSCNN_test
IX = AugMSCNN_test;
[y1,y2,y3,IX,maxX,maxY,IM]=initial(IX,IM);
while(1)
    timesTmp=times;
% yyy is iteration times
    yyy=yyy+1; 
    yyy
[y1,y2,y3]=FenxiGeleiZhifang(IX,IM,maxX,maxY); 
[StruInfo]=ChangFenbu(maxX,maxY,IX,IM);
[gauss]=QiuJunzhiFangcha(y1,y2,y3,IM,maxX,maxY);
[IX,times,IMMM]=BianXiangsuLeibie2(maxX,maxY,StruInfo,gauss,IX,times);
    if yyy==20
        break;
    end
end
time = toc
demo_label = IMMM;
imshow(demo_label);
