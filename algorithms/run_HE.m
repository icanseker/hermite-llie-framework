function enhanced = run_HE(img_low)
%RUN_HE  Histogram Equalization applied per color channel.
%
%  ── Method Overview ─────────────────────────────────────────────────────────
%  Histogram equalization redistributes pixel intensity values so that the
%  resulting histogram is approximately uniform across the full [0, 255] range.
%  This maximizes global contrast by stretching the tonal range.
%
%  For a grayscale image, histogram equalization computes a mapping function
%  T(r) = (L-1) · CDF(r), where CDF is the cumulative distribution function
%  of the pixel intensities and L = 256 for 8-bit images. Each input intensity
%  r is remapped to T(r), spreading values across the full range.
%
%  ── Per-channel application ──────────────────────────────────────────────────
%  For RGB images, equalization is applied independently to each channel (R, G, B).
%  This maximizes contrast in each channel separately but may introduce color
%  shifts because the relative balance between channels is altered.
%
%  ── Characteristics ─────────────────────────────────────────────────────────
%  Strengths:   Very fast, simple, globally improves contrast in dark images.
%  Limitations: No spatial adaptivity — treats dark and bright regions equally,
%               which can cause over-enhancement in already-bright areas.
%               Per-channel equalization may alter color rendition.
%
%  ── Parameters ──────────────────────────────────────────────────────────────
%  None — histogram equalization is fully parameter-free.
%
%  ── Reference ───────────────────────────────────────────────────────────────
%  Gonzalez & Woods. "Digital Image Processing" (4th ed.), Chapter 3, 2018.
%
%  ── Input ───────────────────────────────────────────────────────────────────
%  img_low  - Low-light image (uint8, H × W × 3 or H × W)
%
%  ── Output ──────────────────────────────────────────────────────────────────
%  enhanced - Histogram-equalized image (uint8, same size as input)

    % Initialize output with a copy of the input (correct size/type allocation)
    enhanced = img_low;

    % Apply histeq independently to each channel.
    % histeq() computes the equalization mapping using the full-image histogram
    % of that channel and applies it uniformly across all pixels.
    for ch = 1:size(img_low, 3)
        enhanced(:, :, ch) = histeq(img_low(:, :, ch));
    end

end