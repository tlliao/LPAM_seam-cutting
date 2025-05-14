function [seam_As, seam_Bs] = patchSeamEstimation(warped_img1, warped_img2, seam_img1, seam_img2)


[ sz1, sz2 ]=size(seam_img1);
C = true(sz1,sz2);
revindC = (1:sz1*sz2)'; 
nNodes = sz1*sz2;

%%  terminalWeights, choose source and sink nodes
% border of overlapping region
border_A = findBorder(seam_img1);
border_B = findBorder(seam_img2);
border_C = findBorder(C);

imgseedA = border_A & border_C;
imgseedB = border_B & border_C; %~imgseedA & border_C; 

% data term
tw=zeros(nNodes,2);
tw(revindC(imgseedA),1)=inf;
tw(revindC(imgseedB),2)=inf;

terminalWeights=tw;       % data term

%% calculate edgeWeights
CL1=C&[C(:,2:end) false(sz1,1)];
CL2=[false(sz1,1) CL1(:,1:end-1)];
CU1=C&[C(2:end,:);false(1,sz2)];
CU2=[false(1,sz2);CU1(1:end-1,:)];
    
%  edgeWeights:  Euclidean-weighted norm
ang_1=warped_img1(:,:,1); sat_1=warped_img1(:,:,2); val_1=warped_img1(:,:,3);
ang_2=warped_img2(:,:,1); sat_2=warped_img2(:,:,2); val_2=warped_img2(:,:,3);
% baseline difference map
imgdif = sqrt( ( (ang_1.*C-ang_2.*C).^2 + (sat_1.*C-sat_2.*C).^2 + (val_1.*C-val_2.*C).^2 )./3 );   

% sigmoid-metric difference map
imgdif_sig = calcSigmoidDiff(imgdif, C);
% sigmoid method
DL = (imgdif_sig(CL1)+imgdif_sig(CL2))./2;
DU = (imgdif_sig(CU1)+imgdif_sig(CU2))./2;        

% smoothness term
edgeWeights=[
    revindC(CL1) revindC(CL2) DL+1e-8 DL+1e-8;
    revindC(CU1) revindC(CU2) DU+1e-8 DU+1e-8];

%%  graph-cut labeling
[~, labels] = graphCutMex(terminalWeights, edgeWeights); 

As=C;
Bs=C;
As(revindC(labels==1))=false;   % mask of target seam
Bs(revindC(labels==0))=false;   % mask of reference seam

seam_As = As;
seam_Bs = Bs;

end