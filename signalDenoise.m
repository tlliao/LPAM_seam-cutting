function sigDEN = signalDenoise(SIG)
% FUNC_DENOISE_DW1D Saved Denoising Process.
%   SIG: vector of data
%   -------------------
%   sigDEN: vector of denoised data

%  Auto-generated by Wavelet Toolbox on 13-Mar-2018 17:09:34

% Analysis parameters.
%---------------------
wname = 'haar';
level = 3;

[c,l] = wavedec(SIG, 3, wname);
[cd1, cd2, cd3] = detcoef(c, l, 1:3);
thr1 = max(abs(cd1));
thr2 = max(abs(cd2));
thr3 = max(abs(cd3));

thrSettings =  [thr1; thr2; thr3];
% Denoising parameters.
%----------------------
% meth = 'minimaxi';
% scal_or_alfa = one;
sorh = 's';    % Specified soft or hard thresholding

% Denoise using CMDDENOISE.
%--------------------------
sigDEN = cmddenoise(SIG,wname,level,sorh,NaN,thrSettings);
sigDEN = sigDEN';

end
