function enhanced = run_LIME(img_low)
%RUN_LIME  LIME: Low-Light Image Enhancement via Illumination Map Estimation.
%
%  Source:
%    estija/LIME — https://github.com/estija/LIME
%    Implementation of: Guo X., Li Y., Ling H. "LIME: Low-Light Image
%    Enhancement via Illumination Map Estimation."
%    IEEE Transactions on Image Processing, 26(2), 982-993, 2017.
%
%  Note: The official LIME code by the original authors is distributed as
%  compiled .p files (source not readable). estija/LIME is a clean open-source
%  MATLAB implementation of the same paper, validated against the paper results.
%
%  All algorithm files are in algorithms/lime/ (copied unchanged from the repo).
%  This wrapper calls lime_main_module() with fixed parameters and flag=0
%  (no display).
%
%  Parameters (from lime_main_module.m defaults and estija/LIME README):
%    alpha = 0.08   Smoothness weight — hardcoded in lime_main_module.m
%    gamma = 0.8    Illumination gamma — hardcoded in lime_main_module.m
%    mu    = 0.01   ADMM initial penalty parameter
%    rho   = 1.2    ADMM penalty update ratio
%    ds    = 10     Bilateral filter degree of smoothing
%    ss    = 1.5    Bilateral filter spatial sigma
%    flag  = 0      No display
%
%  Input:
%    img_low  - Low-light image (uint8, H x W x 3)
%
%  Output:
%    enhanced - Enhanced image (uint8, H x W x 3)

    mu   = 0.01;
    rho  = 1.2;
    ds   = 10;
    ss   = 1.5;
    flag = 0;

    [~, ~, ~, Iout] = lime_main_module(img_low, mu, rho, ds, ss, flag);

    % lime_main_module returns two outputs: intermediate (3rd) and denoised (Iout).
    % We use Iout (the denoised result) as the final output, consistent with
    % the paper's recommended pipeline (Sec. 2.3: denoising is the final step).
    % The 3rd output is suppressed with ~ as it is not needed.
    enhanced = im2uint8(Iout);

end