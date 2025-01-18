function [warpI1, warpI2 ] = realignmentviaSIFTflow(im1, im2, mask_p)

%% pre-process
cellsize=3;
gridspacing=1;
SIFTflowpara.alpha=2*255;
SIFTflowpara.d=40*255;
SIFTflowpara.gamma=0.005*255;
SIFTflowpara.nlevels=4;
SIFTflowpara.wsize=2;
SIFTflowpara.topwsize=10;
SIFTflowpara.nTopIterations = 60;
SIFTflowpara.nIterations= 30;
para_beta = 8;

%% sift flow
sift1 = mexDenseSIFT(im1,cellsize,gridspacing);
sift2 = mexDenseSIFT(im2,cellsize,gridspacing);
[vx,vy,~]=SIFTflowc2f(sift2,sift1,SIFTflowpara);
[h_im1,w_im1,nchannels]=size(im1);
% [h_im2,w_im2,nchannels]=size(im2);
[h_vx, w_vx]=size(vx);
% [py, px] = ind2sub([h_im1,w_im1],find(mask_p));
% seam_pts = [px, py];

%% smoothly realignment 
[xx1,yy1]=meshgrid(1:w_im1,1:h_im1);
% [xx2,yy2]=meshgrid(1:w_im2,1:h_im2);
[XX,YY]=meshgrid(1:w_vx,1:h_vx);
vec_XY = [XX(:), YY(:)];


%% generate sigmoid smooth function % version 1: smooth from "left" to "right"
% m_vx = mean(vx(:));
% m_vy = mean(vy(:)); orth_v = orth_v./norm(orth_v,2);
%if sum(As(:,1))==h_im1 % if seam is left->right
    orth_v = [1,0];%[m_vy, -m_vx];
%end
if sum(mask_p(:,end))==h_im1 % if seam is right->left
    orth_v = [-1,0];%[m_vy, -m_vx];
end
if sum(mask_p(1,:))==w_im1 % if seam is up->down
    orth_v = [0,1];%[m_vy, -m_vx];
end
if sum(mask_p(end,:))==w_im1 % if seam is down->up
    orth_v = [0,-1];%[m_vy, -m_vx];
end

    corner_x = orth_v*[0, 0, w_im1-1, w_im1-1; 0, h_im1-1, 0, h_im1-1];
    max_x = max(corner_x);
    min_x = min(corner_x);
    proj_x = (sum(repmat(orth_v,length(vec_XY),1).*(vec_XY-1),2)-min_x)/(max_x-min_x);
    proj_y = 1./(1+exp(-para_beta.*(proj_x-0.5)));

smooth_v = reshape(proj_y, [h_vx, w_vx]);
smooth_vx = vx.*smooth_v;
smooth_vy = vy.*smooth_v;

%% show vector flow
% ord_x = XX(1:5:end, 1:5:end);
% ord_y = YY(1:5:end, 1:5:end);
% vec_x = -vx(1:5:end, 1:5:end);
% vec_y = -vy(1:5:end, 1:5:end);
% figure,subplot(121),imshow(im1);
% hold on
% quiver(ord_x,ord_y,vec_x,vec_y,'LineWidth',1);
% hold off
% subplot(122),imshow(im1);hold on
% quiver(ord_x,ord_y,-smooth_vx(1:5:end, 1:5:end),-smooth_vy(1:5:end, 1:5:end),'LineWidth',1);
% hold off

%% vector flow calculation
    
XX1=XX+smooth_vx;
YY1=YY+smooth_vy;
XX1=min(max(XX1,1),w_im1); YY1=min(max(YY1,1),h_im1);

%% patch re-alignment
warpI1 = zeros(h_vx,w_vx,nchannels);
warpI2 = im2; %zeros(h_vx,w_vx,nchannels);
for i=1:nchannels
    foo1=interp2(xx1,yy1,im1(:,:,i),XX1,YY1,'bicubic');
    warpI1(:,:,i)=foo1;
end


end