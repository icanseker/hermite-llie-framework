function enhanced = run_BIMEF(img_low)
%RUN_BIMEF  Comparison baseline: BIMEF (Bio-Inspired Multi-Exposure Fusion).
%
%  Source:
%    Ying Z., Li G., Gao W. "A Bio-Inspired Multi-Exposure Fusion Framework
%    for Low-light Image Enhancement."
%    IEEE Transactions on Cybernetics, 50(6), 2400-2414, 2020.
%    arXiv: 1711.00591
%    Official code: https://github.com/baidut/BIMEF
%
%  Files used from the repo (algorithms/bimef/):
%    BIMEF.m  — readable source code (full implementation, all sub-functions)
%    BIMEF.p  — compiled version of the same code; MATLAB uses .p when present
%
%  This wrapper calls BIMEF() directly with default parameters
%  (mu=0.5, k=auto, a=-0.3293, b=1.1258) as defined in BIMEF.m.
%
%  Input:
%    img_low  - Low-light image (uint8, H x W x 3)
%
%  Output:
%    enhanced - Enhanced image (uint8, H x W x 3)

    % BIMEF() returns double in [0,1]
    enh_d    = BIMEF(img_low);
    enhanced = im2uint8(max(0, min(1, enh_d)));

end