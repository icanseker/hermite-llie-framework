function val = calc_NIQE(img)
%CALC_NIQE  Compute Natural Image Quality Evaluator (NIQE) score.
%
%  ── What it measures ─────────────────────────────────────────────────────────
%  NIQE is a blind (no-reference) image quality assessment metric. It measures
%  the deviation of an image's local patch statistics from a multivariate
%  Gaussian model fitted to a corpus of pristine natural images. No ground
%  truth reference image is required.
%
%  ── Interpretation ───────────────────────────────────────────────────────────
%  A LOWER score indicates better perceptual quality and higher naturalness.
%  Enhancement artifacts — noise amplification, halo ringing, over-saturation,
%  or unnatural tonal shifts — all increase the NIQE score. Methods that
%  preserve the statistical regularity of natural images score lower.
%
%  ── Relationship to other metrics in this framework ──────────────────────────
%  NIQE complements SSIM and Entropy:
%    - SSIM measures structural fidelity against a reference (full-reference)
%    - Entropy measures information richness (no-reference, higher = more detail)
%    - NIQE measures perceptual naturalness (no-reference, lower = more natural)
%  An enhancement can maximize Entropy by introducing noise — NIQE penalizes
%  this, providing a counterbalancing naturalness signal.
%
%  ── Requirements ─────────────────────────────────────────────────────────────
%  Requires MATLAB Image Processing Toolbox (niqe built-in function).
%  The niqe() function operates on grayscale uint8 images.
%
%  ── Reference ────────────────────────────────────────────────────────────────
%  Mittal A., Soundararajan R., Bovik A.C. "Making a 'Completely Blind' Image
%  Quality Analyzer." IEEE Signal Processing Letters, 20(3), 209–212, 2013.
%
%  ── Input ────────────────────────────────────────────────────────────────────
%  img - Enhanced image (H × W × 3 or H × W, uint8)
%
%  ── Output ───────────────────────────────────────────────────────────────────
%  val - NIQE score (scalar, lower is better, typically in range [0, 15])

    % Convert to grayscale if RGB — NIQE operates on luminance statistics
    if size(img, 3) == 3
        img_gray = rgb2gray(img);
    else
        img_gray = img;
    end

    % Ensure uint8 — niqe() expects integer-valued pixel data
    if ~isa(img_gray, 'uint8')
        img_gray = uint8(img_gray);
    end

    % Standard NIQE computation via MATLAB Image Processing Toolbox
    val = niqe(img_gray);

end