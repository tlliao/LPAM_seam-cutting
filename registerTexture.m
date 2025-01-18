function [warped_img1, warped_img2] = registerTexture(img1,img2, parameters)
% given two images img1 target  img2 reference
% detect and match sift features, estimate homography transformation
% and calculate alignment result

[pts1, pts2] = siftMatch(img1, img2, parameters);

%% image alignment via homography method
[matches_1, matches_2] = homoRANSAC(pts1, pts2, parameters); % delete wrong match features
init_H = calcHomo(matches_1, matches_2);  % fundamental homography
[warped_img1, warped_img2] = homographyAlign(img1, img2,init_H);

end