clear; clc; close all; 
%% Setup VLFeat toolbox.
%----------------------
addNeedingPaths;
run ../vlfeat-0.9.21/toolbox/vl_setup;

% setup parameters
% Parameters of SIFT detection
parameters.peakthresh = 0;
parameters.edgethresh = 500;

% % Parameters of RANSAC via fundamental matrix
parameters.minPtNum = 4;    % minimal number for model fitting
parameters.iterNum = 2000;  % maximum number of trials
parameters.thDist = 0.01;   % distance threshold for inliers

path1 =  (''); %
path2 =  (''); %
img1 = im2double(imread(path1));  % target image
img2 = im2double(imread(path2));  % reference image
    
%% image alignment
[warped_img1,  warped_img2] = registerTexture(img1, img2, parameters);

%% seam-cutting
[seam_As, seam_Bs]=seamEstimationInDiff(warped_img1, warped_img2); % intial seam
[seam_As_, seam_Bs_, warped_img1_, warped_img2_] = seamImproving(warped_img1, warped_img2, seam_As, seam_Bs); % improved seam
imgout = warped_img1.*cat(3,seam_As_,seam_As_,seam_As_) + warped_img2.*cat(3,seam_Bs_,seam_Bs_,seam_Bs_); % % seam-based composition