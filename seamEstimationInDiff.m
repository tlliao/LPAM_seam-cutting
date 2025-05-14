function [seam_As, seam_Bs]=seamEstimationInDiff(warped_img1, warped_img2)

%%  pre-process of seam-cutting
w1 = imfill(imbinarize(rgb2gray(warped_img1), 0),'holes');
w2 = imfill(imbinarize(rgb2gray(warped_img2), 0),'holes');
A = w1;  B = w2;
C = A & B;  % mask of overlapping region
[ sz1, sz2 ]=size(C);
ind = find(C);  % index of overlapping region
nNodes = size(ind,1);
revindC = zeros(sz1*sz2,1);
revindC(C) = 1:length(ind);

%%  terminalWeights, choose source and sink nodes
% border of overlapping region
% border_A = findBorder(A);
border_B = findBorder(B);
border_C = findBorder(C);

imgseedA = border_B & border_C;
imgseedB = ~imgseedA & border_C; 

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

% sigmoid method
DL = (imgdif(CL1)+imgdif(CL2))./2;
DU = (imgdif(CU1)+imgdif(CU2))./2;        

% smoothness term
edgeWeights=[
    revindC(CL1) revindC(CL2) DL+1e-8 DL+1e-8;
    revindC(CU1) revindC(CU2) DU+1e-8 DU+1e-8];

%%  graph-cut labeling
[~, labels] = graphCutMex(terminalWeights, edgeWeights); 

As=A;
Bs=B;
As(ind(labels==1))=false;   % mask of target seam
Bs(ind(labels==0))=false;   % mask of reference seam
seam_As=As;
seam_Bs=Bs;

end