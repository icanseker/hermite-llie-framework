function val = calc_SSIM(enhanced, reference)
%CALC_SSIM  Compute Structural Similarity Index (SSIM) between two images.
%
%  ── What it measures ─────────────────────────────────────────────────────────
%  SSIM measures perceptual and structural similarity between an enhanced image
%  and a ground truth reference. Unlike PSNR/MSE which measure pixel-level
%  numeric error, SSIM captures whether the enhanced image looks correct to a
%  human observer — does it preserve the textures, edges, and spatial structure
%  of the reference? Does it match in local brightness and contrast?
%
%  SSIM decomposes similarity into three components evaluated locally:
%
%    l(x,y) = (2μₓμᵧ + C₁) / (μₓ² + μᵧ² + C₁)         Luminance comparison
%    c(x,y) = (2σₓσᵧ + C₂) / (σₓ² + σᵧ² + C₂)           Contrast comparison
%    s(x,y) = (σₓᵧ + C₃)   / (σₓσᵧ + C₃)                Structure comparison
%
%  The full SSIM index is:
%    SSIM(x,y) = l(x,y) · c(x,y) · s(x,y)
%
%  where:
%    μₓ, μᵧ    — local mean intensities
%    σₓ², σᵧ²  — local variances
%    σₓᵧ       — local cross-covariance (measures structural correlation)
%    C₁, C₂, C₃ — small stabilizing constants (avoid division by zero)
%
%  The overall score is the mean SSIM over all local windows in the image.
%
%  ── Range and interpretation ─────────────────────────────────────────────────
%  Range: [-1, 1] — higher is better; 1 = perfect structural match
%
%  Typical values in low-light enhancement:
%    > 0.80  — excellent structural preservation
%    0.60–0.80 — good
%    0.40–0.60 — moderate (visible structural differences)
%    < 0.40  — poor (significant distortion or artifacts)
%
%  ── Why SSIM is the primary metric in this paper ─────────────────────────────
%  The proposed Hermite method's key strength is structure preservation:
%  the 8-directional convolution kernels follow spatial gradients isotropically,
%  lifting dark regions while maintaining the structural content of the scene.
%  SSIM directly captures this property — a high SSIM score confirms that the
%  enhancement is perceptually faithful, not just numerically close.
%
%  MSE/PSNR can rate a heavily over-brightened image similarly to a
%  structure-preserving one if the mean squared error happens to be similar.
%  SSIM penalizes structural distortions that PSNR would miss.
%
%  ── Requires ground truth ────────────────────────────────────────────────────
%  SSIM is a full-reference metric: it requires a reference image to compare
%  against. If no ground truth is available (no file in source/high/), this
%  function is not called and SSIM is reported as N/A in the metrics file.
%
%  ── Reference ───────────────────────────────────────────────────────────────
%  Wang Z., Bovik A.C., Sheikh H.R., Simoncelli E.P.
%  "Image Quality Assessment: From Error Visibility to Structural Similarity."
%  IEEE Transactions on Image Processing, 13(4), 600–612, 2004.
%
%  ── Inputs ───────────────────────────────────────────────────────────────────
%  enhanced  - Enhanced output image (uint8, H × W × 3 or H × W)
%  reference - Ground truth reference image (uint8, same dimensions)
%
%  ── Output ───────────────────────────────────────────────────────────────────
%  val  - SSIM value (scalar, range [-1, 1])

    % ── Convert to grayscale ─────────────────────────────────────────────────
    % SSIM is computed on the luminance channel to measure structural and tonal
    % similarity without being influenced by color channel-specific differences.
    if size(enhanced, 3) == 3
        enh_gray = rgb2gray(enhanced);
        ref_gray = rgb2gray(reference);
    else
        enh_gray = enhanced;
        ref_gray = reference;
    end

    % ── Compute SSIM using MATLAB's built-in function ─────────────────────────
    % ssim() applies the standard 11×11 Gaussian-weighted local window with
    % the default constants C1=(0.01·L)², C2=(0.03·L)², where L=255 for uint8.
    val = ssim(enh_gray, ref_gray);

end