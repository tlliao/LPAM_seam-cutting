function [ imgdif_sig ] = calcSigmoidDiff(imgdif, C)

% sigmoid-metric difference map
a_rgb = 0.06; % bin of histogram
beta=4/a_rgb; % beta
gamma=exp(1); % base number
para_alpha = histOstu(imgdif(C), a_rgb);  % parameter:tau
imgdif_sig = 1./(1+power(gamma,beta*(-imgdif+para_alpha))); % difference map with logistic function
imgdif_sig = imgdif_sig.*C;   % difference to compute the smoothness term 

end