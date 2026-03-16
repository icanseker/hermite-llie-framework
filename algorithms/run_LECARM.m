function enhanced = run_LECARM(img_low)
%RUN_LECARM  Comparison baseline: Low-light Enhancement with Camera Response Model.
%
%  Ying Z., Li G., Ren Y., Wang R., Wang W.
%  "A New Image Contrast Enhancement Algorithm Using Exposure Fusion Framework"
%  Computer Analysis of Images and Patterns (CAIP), 2017.
%  Official code: https://github.com/baidut/LECARM
%
%  Official source code used unchanged, located at algorithms/lecarm/.
%  This wrapper adds the lecarm/ directory to the MATLAB path (handled by
%  main.m) and calls LECARM() with the default Sigmoid camera model.
%
%  Pipeline:
%    1. T = max(R,G,B) per pixel
%    2. T smoothed via LIME-style solver (lambda=0.15, sigma=2) at half res
%    3. K = min(1/T, 7) — per-pixel exposure ratio
%    4. enhanced = CameraModels.Sigmoid().btf(I, K)
%
%  Input:
%    img_low  - Low-light input image (H x W x 3, uint8)
%
%  Output:
%    enhanced - Enhanced image (H x W x 3, uint8)
%               Metrics computed by main.m (calc_PSNR, calc_SSIM, etc.)

    % CameraModels.Sigmoid: default model in original LECARM.m (n=0.90, sigma=0.60)
    % Path to algorithms/lecarm/ is added by main.m
    model   = CameraModels.Sigmoid();
    enh_d   = LECARM(img_low, model);     % Returns double in [0,1]
    enhanced = uint8(max(0, min(1, enh_d)) * 255);

end