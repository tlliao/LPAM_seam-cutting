function [ eval_signal, patch_coor ] = evalQualityofSeam(img1, img2, C_lap, seam_pts, patchsize)
% evaluate the seam according to patch difference between input images (img1,img2)
bound_num = size(seam_pts,1);
eval_signal = zeros(bound_num,1);
patch_coor = zeros(bound_num, 4);

for i=1:bound_num
    i_bound = seam_pts(i,1);
    j_bound = seam_pts(i,2);

    y1 = max(i_bound-(patchsize-1)/2, 1);
    y2 = min(i_bound+(patchsize-1)/2, size(img1,1));
    x1 = max(j_bound-(patchsize-1)/2, 1);
    x2 = min(j_bound+(patchsize-1)/2, size(img1,2));
    patch_coor(i,:) = [y1, y2, x1, x2];
    patch_mask = C_lap(y1:y2,x1:x2);
    
    img1_crop = img1(y1:y2,x1:x2,:).*cat(3,patch_mask,patch_mask,patch_mask);
    img2_crop = img2(y1:y2,x1:x2,:).*cat(3,patch_mask,patch_mask,patch_mask);

    ssim_error1 = ssim(img1_crop(:,:,1), img2_crop(:,:,1));
    ssim_error2 = ssim(img1_crop(:,:,2), img2_crop(:,:,2));
    ssim_error3 = ssim(img1_crop(:,:,3), img2_crop(:,:,3));
    ssim_error = (ssim_error1 + ssim_error2 +ssim_error3)/3;

    eval_signal(i) = 1-ssim_error;
end

end