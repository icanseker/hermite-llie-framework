function enhanced = run_dong(img_low)
%RUN_DONG  Inverted dehazing-based low-light enhancement (Dong et al. 2010).
%
%  ── Method Overview ─────────────────────────────────────────────────────────
%  This method exploits a structural similarity between low-light images and
%  hazy images: both exhibit a bright, low-contrast "veil" that obscures image
%  content. In a hazy image, the veil is additive (atmospheric scattering); in
%  a low-light image, the veil appears after image inversion (pixel-wise complement).
%
%  The pipeline:
%    1. INVERT: I_inv = 255 - I   (low-light → resembles hazy image)
%    2. DEHAZE: I_dehazed = imreducehaze(I_inv)  (remove the "haze")
%    3. INVERT: I_enhanced = 255 - I_dehazed     (back to enhanced low-light)
%
%  ── Dehazing function: imreducehaze ──────────────────────────────────────────
%  imreducehaze implements the Approximate Dark Channel Prior (approxdcp) method,
%  which estimates and removes the atmospheric veil from a hazy image.
%  The dark channel prior observes that in most local patches of haze-free outdoor
%  images, at least one color channel has very low intensity — the "dark channel".
%  Haze raises this dark channel uniformly; removing it restores clarity.
%
%  ── Parameters ───────────────────────────────────────────────────────────────
%  Method = 'approxdcp'           — approximate dark channel prior (efficient)
%  ContrastEnhancement = 'none'   — disables MATLAB's built-in post-dehazing
%                                   contrast boost to avoid double-processing.
%                                   The dehazing step itself restores contrast;
%                                   additional stretching would be redundant and
%                                   could over-saturate the result.
%
%  ── Characteristics ─────────────────────────────────────────────────────────
%  Strengths:   Simple three-step pipeline; leverages mature dehazing algorithms.
%  Limitations: The low-light ↔ haze analogy is approximate and breaks down for
%               images with strong local brightness variation. May introduce slight
%               color shifts at the inversion stage.
%
%  ── Reference ───────────────────────────────────────────────────────────────
%  Dong X., Pang Y., Wen J.
%  "Fast Efficient Algorithm for Enhancement of Low Lighting Video."
%  ACM SIGGRAPH Posters, 2010.
%
%  MATLAB implementation reference:
%  https://www.mathworks.com/help/images/low-light-image-enhancement.html
%
%  ── Input ───────────────────────────────────────────────────────────────────
%  img_low  - Low-light image (uint8, H × W × 3)
%
%  ── Output ──────────────────────────────────────────────────────────────────
%  enhanced - Enhanced image (uint8, H × W × 3)

    % Step 1: Invert the low-light image
    % imcomplement computes 255 - I for uint8, transforming the dark image
    % into a bright, hazy-looking one suitable for dehazing
    img_inv = imcomplement(img_low);

    % Step 2: Apply dehazing
    % imreducehaze estimates and removes the "atmospheric veil" introduced by
    % the inversion. ContrastEnhancement='none' prevents double contrast boosting.
    dehazed = imreducehaze(img_inv, 'Method', 'approxdcp', ...
              'ContrastEnhancement', 'none');

    % Step 3: Invert back
    % Inverting the dehazed image recovers the enhanced low-light result
    enhanced = imcomplement(dehazed);

    % Ensure uint8 output — imcomplement on non-uint8 may return double
    if ~isa(enhanced, 'uint8')
        enhanced = im2uint8(enhanced);
    end

end