function [seam_As, seam_Bs, imgw1, imgw2] = seamImproving(imgw1, imgw2, As, Bs)
% imgw1: aligned target image
% imgw2: aligned reference image
%  As:   mask of target seam
%  Bs:   mask of reference seam
%  C:    mask of overlapping region
%% pre-process and settings
patch_size = 21;       
SE_seam = strel('diamond', 1);
A = imfill(imbinarize(rgb2gray(imgw1), 0),'holes');
B = imfill(imbinarize(rgb2gray(imgw2), 0),'holes');
C = A & B;  % mask of overlapping region
As_seam = imdilate(As, SE_seam) & A;
Cs_seam = As_seam & Bs;  % mask of stitching seam
[sz1, sz2]=size(C);

%%  find potential artifacts along the seam for patch mark
% extract pixels on the seam and evaluate the patch error
[seam_ptsy,seam_ptsx] = ind2sub([size(C,1),size(C,2)], find(Cs_seam));
seam_pts = [seam_ptsy,seam_ptsx];
[ssim_error, patch_coor] = evalQualityofSeam(imgw1, imgw2, C, seam_pts, patch_size);
% mark misaligned local regions
if max(ssim_error)<=1.5*mean(ssim_error)
    seam_As=As;
    seam_Bs=Bs;
    return;
end
T = graythresh(ssim_error);
artifacts_pixels = seam_pts(ssim_error>=T,:);
artifacts_patchs = patch_coor(ssim_error>=T,:);
artifacts_masks = false(sz1,sz2);
mask_pixels = false(sz1,sz2);
for i=1:size(artifacts_patchs,1)
    artifacts_masks(artifacts_patchs(i,1):artifacts_patchs(i,2),artifacts_patchs(i,3):artifacts_patchs(i,4))=1;
    mask_pixels(artifacts_pixels(i,1),artifacts_pixels(i,2))=1;
end
% add modification to artifacts_masks: connect neighboring patches if they are close enough
artifacts_masks = imclose(artifacts_masks, strel("square",10));

% colored_seam = imgseam;
% colored_seam(artifacts_masks)=0;
% % figure,imshow(patch_seam);
% imwrite(colored_seam,[outpath, 'patch_seam.jpg']);

%% delete photometric misaligned patches, preserve geometric misaligned patches for correspondences insertion
[L,n] = bwlabel(artifacts_masks);
As2 = As;
Bs2 = Bs;
tmpimgw1=imgw1;
tmpimgw2=imgw2;
for i=1:n
    tmp_L = L==i;
    [tmpm, tmpn]=ind2sub([sz1,sz2],find(tmp_L));
    s_y = min(tmpm); e_y = max(tmpm);
    s_x = min(tmpn); e_x = max(tmpn);
    tmpimgw1(s_y,s_x,:)=[1,0,0];
    tmpimgw2(s_y,s_x,:)=[1,0,0];
    tmpimgw1(e_y,e_x,:)=[1,0,0];
    tmpimgw2(e_y,e_x,:)=[1,0,0];
    crop_img1 = imgw1(s_y:e_y,s_x:e_x,:);
    crop_img2 = imgw2(s_y:e_y,s_x:e_x,:);
    s_c_img1 = As(s_y:e_y,s_x:e_x);
    s_c_img2 = Bs(s_y:e_y,s_x:e_x);
    [w_c_img1, w_c_img2]=realignmentviaSIFTflow(crop_img1, crop_img2, s_c_img1);
    [tmp_As, tmp_Bs] = patchSeamEstimation(w_c_img1, w_c_img2, s_c_img1, s_c_img2);
    As2(s_y:e_y,s_x:e_x)=tmp_As;
    Bs2(s_y:e_y,s_x:e_x)=tmp_Bs;
    imgw1(s_y:e_y,s_x:e_x,:)=w_c_img1;
    imgw2(s_y:e_y,s_x:e_x,:)=w_c_img2;
%     imgout2 = imgw1.*cat(3,As2,As2,As2) + imgw2.*cat(3,Bs2,Bs2,Bs2); 
    
% %     As_seam2 = imdilate(As2, SE_seam) & A;
% %     Cs_seam2 = As_seam2 & Bs2;
%     A2 = imfill(imbinarize(rgb2gray(imgw1), 0),'holes');
%     B2 = imfill(imbinarize(rgb2gray(imgw2), 0),'holes');
%     C2 = A2 & B2;  % mask of overlapping region
%     Cs_seam2 = imdilate(As2, strel('diamond', 5)) & imdilate(Bs2, strel('diamond', 5)) & C2;
%     imgseam2 = imoverlay(imgout2,Cs_seam2,'red');
        
%     imwrite(imgseam2,[outpath, 'imgseam' num2str(i) '.jpg']);
%     imwrite(imgout2,[outpath, 'imgout' num2str(i) '.jpg']);
    
%     As_seam2 = imdilate(As2, SE_seam) & A2;
%     Cs_seam_ = As_seam2 & Bs2;
%     seam_pts2 = contourTracingofSeam(Cs_seam_);
%     [~, ssim_error2, ~] = evalSSIMofSeam(imgw1, imgw2, C2, seam_pts2, patch_size);
%     [ extend_error2, extend_seam2 ] = signalExtend( ssim_error2, seam_pts2, As2, Bs2, C2);
%     colored_seam2  = generatePatchseam( imgout2, extend_error2, extend_seam2 );
%     imwrite(colored_seam2,[outpath, 'imgseam_ssim' num2str(i) '.jpg']);
end

% As_seam2 = imdilate(As2, SE_seam) & A2;
% Cs_seam_final = As_seam2 & Bs2;
% seam_pts_final = contourTracingofSeam(Cs_seam_final);
% [~, eval_ssim_error, ~] = evalSSIMofSeam(imgw1, imgw2, C2, seam_pts_final, patch_size);
% seam_quality = mean(eval_ssim_error);

seam_As = As2;
seam_Bs = Bs2;

% imgout = imgw1.*cat(3,seam_As,seam_As,seam_As) + imgw2.*cat(3,seam_Bs,seam_Bs,seam_Bs); % without gradient domain blending
% As_seam = imdilate(seam_As, SE_seam) & A;
% Cs_seam = As_seam & seam_Bs;  % mask of stitching seam
% imgseam=imoverlay(imgout,Cs_seam,'red');

end