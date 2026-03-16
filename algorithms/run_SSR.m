function enhanced = run_SSR(img_low)
%RUN_SSR  Single-Scale Retinex enhancement (sigma = 80).
%
%  ── Method Overview ─────────────────────────────────────────────────────────
%  The Retinex model, inspired by the human visual system's color constancy,
%  decomposes an image I into illumination L and reflectance R:
%
%    I(x,y) = L(x,y) · R(x,y)
%
%  The goal is to recover R, which represents the true scene appearance
%  independent of lighting conditions. In the log domain:
%
%    log(R) = log(I) - log(L)
%
%  Single-Scale Retinex estimates the illumination L as a Gaussian-blurred
%  version of I (the "surround" function):
%
%    SSR(x,y) = log(I(x,y)) - log(G_σ * I(x,y))
%
%  where G_σ is a 2D Gaussian kernel with standard deviation σ, and * denotes
%  convolution. The intuition is that low-frequency (blurred) content captures
%  the slowly-varying illumination, and subtracting it in the log domain removes
%  the illumination component, leaving the high-frequency reflectance.
%
%  ── Parameter: σ = 80 ────────────────────────────────────────────────────────
%  σ = 80 is the standard value from the original SSR paper (Jobson et al. 1997).
%  Larger σ captures broader illumination variations but blurs finer details.
%  Smaller σ preserves more detail but may not fully remove slow illumination trends.
%
%  ── Per-channel application ──────────────────────────────────────────────────
%  SSR is applied independently to each color channel (R, G, B). The log-domain
%  output spans a range that does not directly correspond to [0, 255]. Linear
%  normalization (min-max rescaling) per channel maps each channel to [0, 255].
%
%  ── Characteristics ─────────────────────────────────────────────────────────
%  Strengths:   Adaptive to local illumination variation; good detail recovery.
%  Limitations: May introduce halo artifacts at strong edges; single scale
%               cannot simultaneously handle both fine and coarse illumination.
%
%  ── Reference ───────────────────────────────────────────────────────────────
%  Jobson D.J., Rahman Z., Woodell G.A.
%  "Properties and Performance of a Center/Surround Retinex."
%  IEEE Transactions on Image Processing, 6(3), 451–462, 1997.
%
%  ── Input ───────────────────────────────────────────────────────────────────
%  img_low  - Low-light image (uint8, H × W × 3 or H × W)
%
%  ── Output ──────────────────────────────────────────────────────────────────
%  enhanced - SSR-enhanced image (uint8, same size as input)

    sigma = 80;  % Gaussian surround scale (standard value from original paper)

    % ── Convert to double and add +1 offset ───────────────────────────────────
    % log(0) is undefined; adding 1 ensures log(I+1) is always finite.
    % This is standard practice in Retinex implementations.
    img_d = double(img_low) + 1.0;
    [rows, cols, channels] = size(img_d);

    % Pre-allocate output in the log-domain (before normalization)
    ssr_out = zeros(rows, cols, channels);

    % ── Compute SSR per channel ───────────────────────────────────────────────
    for ch = 1:channels

        % Compute Gaussian surround (estimated illumination for this channel)
        % imgaussfilt applies a 2D Gaussian filter with the given sigma
        surround = imgaussfilt(img_d(:, :, ch), sigma);

        % SSR = log(I) - log(G_σ * I)
        % In the log domain, this is equivalent to log(I / surround)
        ssr_out(:, :, ch) = log(img_d(:, :, ch)) - log(surround);

    end

    % ── Per-channel linear normalization to [0, 255] ──────────────────────────
    % The log-domain SSR output spans a range that can be negative and does not
    % correspond to standard 8-bit values. Linear normalization maps each
    % channel's min to 0 and max to 255, preserving relative intensity differences.
    enhanced = zeros(rows, cols, channels);
    for ch = 1:channels
        ch_data = ssr_out(:, :, ch);
        ch_min  = min(ch_data(:));
        ch_max  = max(ch_data(:));
        if ch_max > ch_min
            % Standard min-max normalization
            enhanced(:, :, ch) = (ch_data - ch_min) / (ch_max - ch_min) * 255;
        else
            % All pixels have the same value (flat channel) — scale to 255
            enhanced(:, :, ch) = ch_data * 255;
        end
    end

    enhanced = uint8(enhanced);

end