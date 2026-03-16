function val = calc_entropy(img)
%CALC_ENTROPY  Compute Shannon entropy of an image.
%
%  ── What it measures ─────────────────────────────────────────────────────────
%  Shannon entropy quantifies the information content (randomness) of the
%  pixel intensity distribution. It measures how uniformly pixel values are
%  spread across the available range [0, 255].
%
%  A dark image with most pixels clustered near intensity 0 has very low
%  entropy — it contains little tonal variation and little information.
%  An enhanced image with pixel values spread broadly across [0, 255] has
%  high entropy — richer content, more detail, more tonal variation.
%
%  ── Formula ──────────────────────────────────────────────────────────────────
%  H(X) = -Σₓ p(x) · log₂(p(x))
%
%  where p(x) is the empirical probability of intensity x, estimated from
%  the normalized 256-bin grayscale histogram of the image.
%  The sum runs over x ∈ {0, 1, ..., 255}, skipping bins where p(x) = 0
%  (since 0 · log₂(0) is defined as 0 by convention, but log₂(0) = -∞
%  numerically — removing zero bins avoids NaN/Inf in the sum).
%
%  ── Units and range ──────────────────────────────────────────────────────────
%  Result is in bits. Maximum entropy = log₂(256) = 8.0 bits, achieved when
%  all 256 intensity levels are equally likely (perfectly flat histogram).
%  Typical enhanced images: 6.5–7.8 bits. Very dark images: 2.0–5.0 bits.
%
%  ── Role in this paper ───────────────────────────────────────────────────────
%  Entropy serves two roles:
%    1. OPTIMIZATION CRITERION: during parameter search, the (ν, t) pair that
%       produces the highest-entropy enhancement is selected as optimal. This
%       drives the optimization toward parameter values that produce the most
%       informative, detail-rich result.
%    2. EVALUATION METRIC: reported in the results table alongside PSNR/SSIM,
%       providing a no-reference measure of output quality.
%
%  Note: entropy maximization alone does not guarantee visually correct results
%  (an over-exposed or noisy image can also have high entropy). This is why
%  SSIM and PSNR remain the primary evaluation metrics.
%
%  ── Input ────────────────────────────────────────────────────────────────────
%  img  - Enhanced image (uint8, H × W × C or H × W)
%         Converted to grayscale internally if RGB
%
%  ── Output ───────────────────────────────────────────────────────────────────
%  val  - Shannon entropy value in bits (scalar)

    % ── Convert to grayscale ─────────────────────────────────────────────────
    % Entropy is computed on the luminance (grayscale) channel, which captures
    % overall tonal distribution without counting color channel redundancy.
    if size(img, 3) == 3
        gray = rgb2gray(img);
    else
        gray = img;
    end

    % ── Compute normalized histogram ─────────────────────────────────────────
    % imhist returns 256-bin intensity counts; normalizing by total pixel count
    % converts counts to probabilities p(x) summing to 1.
    counts = imhist(gray);           % 256 × 1 vector of pixel counts per intensity
    prob   = counts / sum(counts);   % Normalize → probability distribution

    % ── Remove zero-probability bins ─────────────────────────────────────────
    % Intensities with zero count contribute 0 to entropy (0 · log₂(0) = 0),
    % but computing log₂(0) numerically yields -Inf. Removing these bins
    % keeps the computation finite and correct.
    prob = prob(prob > 0);

    % ── Shannon entropy ───────────────────────────────────────────────────────
    % H = -Σ p(x) · log₂(p(x))
    % log2 is used so the result is in bits (as is standard for discrete entropy).
    val = -sum(prob .* log2(prob));

end