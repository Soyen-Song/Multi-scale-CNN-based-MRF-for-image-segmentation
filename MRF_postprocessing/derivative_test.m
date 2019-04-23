% derivative_test
I = imread('demo3.png');
I = I(:,:,1);
tic
[gx, gy,gxx, gyy,gxy] = DERIVATIVE7(I, 'x', 'y', 'xx','yy','xy');
toc
figure,subplot(2,3,1),imshow(I);
subplot(2,3,2),imshow(gx);
subplot(2,3,3),imshow(gy);
subplot(2,3,4),imshow(gxx);
subplot(2,3,5),imshow(gyy);
subplot(2,3,6),imshow(gxy);

