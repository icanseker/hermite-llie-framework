function val = calc_CII(original, enhanced)
%CALC_CII  Compute Contrast Improvement Index (CII).
%
%  ── What it measures ─────────────────────────────────────────────────────────
%  CII measures how much the mean brightness of the image has been increased
%  by the enhancement. It is defined as the ratio of the enhanced image's
%  mean intensity to the original image's mean intensity:
%
%  ── Formula ──────────────────────────────────────────────────────────────────
%  CII = mean(I_enhanced) / mean(I_original)
%
%  where mean() is computed over all pixels and all color channels.
%
%  ── Range and interpretation ─────────────────────────────────────────────────
%  CII > 1  — enhancement increased mean brightness (expected for low-light)
%  CII = 1  — no change in mean brightness
%  CII < 1  — enhancement decreased brightness (unusual; may indicate darkening)
%
%  For very dark input images (I_mean < 64) that are enhanced to near-normal
%  brightness, CII values of 3–8× are typical and expected.
%
%  ── Characteristics ──────────────────────────────────────────────────────────
%  Strengths:   No-reference metric — does not require ground truth.
%               Simple and interpretable as a "brightness gain factor."
%
%  Limitations: Only captures mean brightness, not distribution or structure.
%               A high CII merely means the image got brighter on average —
%               it does not indicate whether detail was preserved or whether
%               the enhancement is perceptually correct. An extremely
%               over-brightened (blown-out) image would have a very high CII
%               while being visually degraded.
%               CII is complementary to SSIM/PSNR, not a substitute.
%
%  ── Inputs ───────────────────────────────────────────────────────────────────
%  original - Original low-light image (uint8, H × W × C)
%  enhanced - Enhanced image (uint8, same dimensions)
%
%  ── Output ───────────────────────────────────────────────────────────────────
%  val  - CII value (scalar). Returns Inf if original mean = 0 (all-black image).

    % Compute mean intensities over all pixels and channels
    mean_orig = mean(double(original(:)));
    mean_enh  = mean(double(enhanced(:)));

    % Guard against all-black input (mean = 0 → division by zero)
    if mean_orig == 0
        val = Inf;
        return;
    end

    % Ratio of enhanced to original mean intensity
    val = mean_enh / mean_orig;

end