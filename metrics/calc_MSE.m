function val = calc_MSE(enhanced, reference)
%CALC_MSE  Compute Mean Squared Error (MSE) between two images.
%
%  ── What it measures ─────────────────────────────────────────────────────────
%  MSE is the most fundamental pixel-level reconstruction error metric.
%  It measures the average squared intensity difference between the enhanced
%  image and the ground truth, summed over all pixels and all color channels.
%
%  ── Formula ──────────────────────────────────────────────────────────────────
%  MSE = (1 / H·W·C) · Σᵢ Σⱼ Σ꜀ (I_enhanced(i,j,c) - I_reference(i,j,c))²
%
%  where H × W is the spatial resolution and C is the number of channels.
%  The squaring penalizes large individual errors more heavily than small ones.
%
%  ── Range and interpretation ─────────────────────────────────────────────────
%  Range: [0, ∞) — lower is better; 0 = perfect reconstruction
%
%  MSE operates in squared pixel units ([0,255]²), so absolute values depend on
%  image bit-depth. Typical low-light enhancement MSE values: 100–2000.
%  PSNR is derived directly from MSE and is more interpretable in practice.
%
%  ── Limitations ──────────────────────────────────────────────────────────────
%  MSE (and PSNR) are purely numeric measures. Two images with the same MSE
%  can look very different perceptually. For example, a globally brightened
%  image with mild pixel errors across the whole image may have the same MSE
%  as a locally distorted image with large errors in one region.
%  SSIM complements MSE by capturing structural and perceptual quality.
%
%  ── Requires ground truth ────────────────────────────────────────────────────
%  Full-reference metric: requires a reference image. If no ground truth is
%  available (no file in source/high/), MSE is reported as N/A.
%
%  ── Inputs ───────────────────────────────────────────────────────────────────
%  enhanced  - Enhanced output image (uint8, H × W × C)
%  reference - Ground truth reference image (uint8, same dimensions)
%
%  ── Output ───────────────────────────────────────────────────────────────────
%  val  - MSE value (scalar, in squared pixel units)

    % Convert to double before subtraction to allow negative differences
    % (uint8 subtraction would clamp at 0 due to unsigned arithmetic)
    diff = double(enhanced) - double(reference);

    % Mean of squared differences over all pixels and all channels
    val = mean(diff(:).^2);

end