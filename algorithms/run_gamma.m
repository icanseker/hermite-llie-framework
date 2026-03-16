function enhanced = run_gamma(img_low)
%RUN_GAMMA  Gamma correction with fixed gamma = 0.4.
%
%  ── Method Overview ─────────────────────────────────────────────────────────
%  Gamma correction is a point-wise (per-pixel) nonlinear intensity transformation
%  defined by the power-law function:
%
%    s = r^γ
%
%  where r ∈ [0, 1] is the normalized input pixel intensity and s ∈ [0, 1] is
%  the corrected output intensity. In practice:
%    - Normalize to [0,1]: r = pixel_value / 255
%    - Apply: s = r^γ
%    - Scale back: output = s × 255
%
%  ── Effect of γ < 1 (brightening) ───────────────────────────────────────────
%  For γ < 1, the power-law curve is concave (bends upward), which means:
%    - Very dark pixels (r ≈ 0) are lifted disproportionately more than bright ones
%    - Brighter pixels (r ≈ 1) are affected less
%  This selectively brightens the dark regions while limiting over-exposure of
%  already-bright regions — making it suitable for low-light enhancement.
%
%  For example, with γ = 0.4:
%    r = 0.1 (dark pixel, ~25/255) → s = 0.1^0.4 ≈ 0.398 (~101/255) — bright
%    r = 0.8 (bright pixel, ~204/255) → s = 0.8^0.4 ≈ 0.893 (~228/255) — modest lift
%
%  ── Parameter choice: γ = 0.4 ────────────────────────────────────────────────
%  γ = 0.4 is a standard fixed value for low-light enhancement benchmarks.
%  It provides consistent, moderate brightening without extreme over-exposure.
%
%  Alternative tested: adaptive γ = max(0.2, I_mean/255), which for images with
%  I_mean < 64 produces γ ≈ 0.06–0.30. These small gamma values cause
%  severe over-brightening (almost all pixels mapped to near-white). Fixed γ = 0.4
%  was chosen for fair, consistent baseline comparison.
%
%  ── Characteristics ─────────────────────────────────────────────────────────
%  Strengths:   Extremely fast; no parameters to tune; consistent behavior.
%  Limitations: Global transformation — no spatial adaptivity. Same correction
%               applied to every pixel regardless of local context. May wash out
%               bright regions if dark areas need very strong brightening.
%
%  ── Reference ───────────────────────────────────────────────────────────────
%  Gonzalez & Woods. "Digital Image Processing" (4th ed.), Chapter 3, 2018.
%  (Power-law / gamma transformations section)
%
%  ── Input ───────────────────────────────────────────────────────────────────
%  img_low  - Low-light image (uint8, H × W × 3 or H × W)
%
%  ── Output ──────────────────────────────────────────────────────────────────
%  enhanced - Gamma-corrected image (uint8, same size as input)

    gamma = 0.4;  % Fixed gamma value for standard low-light baseline

    % Apply s = r^γ to all pixels simultaneously (vectorized):
    %   1. double(img_low)/255  → normalize to [0, 1]
    %   2. .^ gamma             → apply power-law transformation
    %   3. × 255                → scale back to [0, 255]
    %   4. uint8(...)           → cast to 8-bit integer (implicit floor + clamp)
    enhanced = uint8(255 * (double(img_low) / 255) .^ gamma);

end