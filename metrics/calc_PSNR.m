function val = calc_PSNR(enhanced, reference)
%CALC_PSNR  Compute Peak Signal-to-Noise Ratio (PSNR) between two images.
%
%  ── What it measures ─────────────────────────────────────────────────────────
%  PSNR measures reconstruction fidelity: how closely the enhanced image
%  matches the ground truth in terms of pixel-level signal power vs. error power.
%  It is expressed in decibels (dB) — a logarithmic scale that better reflects
%  the range of differences encountered in practice.
%
%  ── Formula ──────────────────────────────────────────────────────────────────
%  PSNR = 10 · log₁₀(MAX² / MSE)
%
%  where:
%    MAX = 255  — maximum possible pixel intensity for 8-bit images
%    MSE        — Mean Squared Error (see calc_MSE.m)
%
%  Equivalently: PSNR = 10·log₁₀(255²) - 10·log₁₀(MSE)
%                      ≈ 48.13 - 10·log₁₀(MSE)
%
%  ── Range and interpretation ─────────────────────────────────────────────────
%  Inf dB  — perfect reconstruction (MSE = 0, identical images)
%  > 30 dB — generally considered good quality
%  20–30 dB — moderate quality (visible differences)
%  < 20 dB — poor quality (significant reconstruction error)
%
%  A difference of 3 dB corresponds to approximately halving/doubling the MSE.
%  A difference of 10 dB corresponds to a 10× change in MSE.
%
%  ── Relationship to MSE ───────────────────────────────────────────────────────
%  PSNR is a monotonic function of MSE: higher PSNR ↔ lower MSE.
%  Both capture pixel-level numeric accuracy, not perceptual quality.
%
%  ── Requires ground truth ────────────────────────────────────────────────────
%  Full-reference metric: requires a reference image. If no ground truth is
%  available (no file in source/high/), PSNR is reported as N/A.
%
%  ── Inputs ───────────────────────────────────────────────────────────────────
%  enhanced  - Enhanced output image (uint8, H × W × C)
%  reference - Ground truth reference image (uint8, same dimensions)
%
%  ── Output ───────────────────────────────────────────────────────────────────
%  val  - PSNR in decibels (dB). Returns Inf if MSE = 0.

    % Compute MSE first (reuse calc_MSE for consistency)
    mse_val = calc_MSE(enhanced, reference);

    % Handle perfect reconstruction case (avoid log(0))
    if mse_val == 0
        val = Inf;
        return;
    end

    % PSNR formula: 10 · log₁₀(MAX² / MSE), with MAX = 255 for uint8
    val = 10 * log10(255^2 / mse_val);

end