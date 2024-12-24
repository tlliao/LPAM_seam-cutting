function [seam_As, seam_Bs, imgw1, imgw2] = seamImproving(imgw1, imgw2, As, Bs, C)
% imgw1: aligned target image
% imgw2: aligned reference image
%  As:   mask of target seam
%  Bs:   mask of reference seam
%  C:    mask of overlapping region
%% pre-process and settings
patch_size = 21;       
SE_seam = strel('diamond', 1);  
A = As | C;
As_seam = imdilate(As, SE_seam) & A;
Cs_seam = As_seam & Bs;  % mask of stitching seam

%%  find potential artifacts along the seam for patch mark
% extract pixels on the seam and evaluate the patch error
seam_pts = contourTracingofSeam(Cs_seam);
[ssim_error, ~, patch_coor] = evalQualityofSeam(imgw1, imgw2, C, seam_pts, patch_size);
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

%% delete photometric misaligned patches, preserve geometric misaligned patches for correspondences insertion
[L,n] = bwlabel(artifacts_masks);
As2 = As;
Bs2 = Bs;
for i=1:n
    tmp_L = L==i;
    [tmpm, tmpn]=ind2sub([sz1,sz2],find(tmp_L));
    s_y = min(tmpm); e_y = max(tmpm);
    s_x = min(tmpn); e_x = max(tmpn);
    crop_img1 = imgw1(s_y:e_y,s_x:e_x,:);
    crop_img2 = imgw2(s_y:e_y,s_x:e_x,:);
    s_c_img1 = As(s_y:e_y,s_x:e_x);
    s_c_img2 = Bs(s_y:e_y,s_x:e_x);
    [w_c_img1, w_c_img2]=realignmentviaSIFTflow(crop_img1, crop_img2, s_c_img1, patch_size);
    [tmp_As, tmp_Bs] = patchSeamEstimation(w_c_img1, w_c_img2, s_c_img1, s_c_img2);
    As2(s_y:e_y,s_x:e_x)=tmp_As;
    Bs2(s_y:e_y,s_x:e_x)=tmp_Bs;
    imgw1(s_y:e_y,s_x:e_x,:)=w_c_img1;
    imgw2(s_y:e_y,s_x:e_x,:)=w_c_img2;
end

seam_As = As2;
seam_Bs = Bs2;

end