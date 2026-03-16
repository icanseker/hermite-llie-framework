function result = apply_convolution(img, a2, a3)
%APPLY_CONVOLUTION  Enhance image using an 8-directional polynomial coefficient kernel.
%
%  This function is the core spatial operator of the proposed method.
%  Given coefficient values (a1=1, a2, a3) derived from the polynomial
%  coefficient bounds, it constructs eight 3×3 directional kernels (one per
%  principal spatial direction), collapses them into a single Master Kernel
%  via the linearity of convolution, and applies it once per color channel.
%
%  ── Why 8 directions? ──────────────────────────────────────────────────────
%  A single oriented kernel would enhance edges aligned with its direction
%  while attenuating orthogonal ones — introducing directional bias.
%  By covering all cardinal (0°, 90°, 180°, 270°) and diagonal
%  (45°, 135°, 225°, 315°) orientations, the enhancement becomes isotropic:
%  every spatial structure is treated equally regardless of orientation.
%  This is the key mechanism behind the method's strong SSIM performance.
%
%  ── Role in the optimization pipeline ──────────────────────────────────────
%  This function is called ONCE per image to produce the final enhanced output
%  using the optimal (ν*, t*) found by the two-phase optimization.
%
%  ── Master Kernel — distributive identity ─────────────────────────────────
%  The 8 directional kernels collapse into a single Master Kernel via the
%  distributive property of the convolution operator:
%
%      (I * h1) + (I * h2) + ... + (I * h8)  ≡  I * (h1 + h2 + ... + h8)
%
%  The 8 kernels are pre-averaged into a single Master Kernel:
%
%      K_master = (1/8) · Σ hᵢ
%
%  This reduces the convolution count from 8C to C with zero precision loss.
%  For the standard RGB case (C=3): from 24 to 3 convolutions per call.
%
%  The Master Kernel has a closed-form structure. Summing all 8 kernels:
%    - Each corner and edge position receives exactly (a1 + a3) total weight
%    - The center receives 8·a2
%  After dividing by 8:
%
%      K_master =  [ (a1+a3)/8,  (a1+a3)/8,  (a1+a3)/8 ]
%                  [ (a1+a3)/8,    a2,        (a1+a3)/8 ]
%                  [ (a1+a3)/8,  (a1+a3)/8,  (a1+a3)/8 ]
%
%  Since a1 = 1:
%
%      K_master =  [ (1+a3)/8,  (1+a3)/8,  (1+a3)/8 ]
%                  [ (1+a3)/8,    a2,       (1+a3)/8 ]
%                  [ (1+a3)/8,  (1+a3)/8,  (1+a3)/8 ]
%
%  ── Kernel structure ─────────────────────────────────────────────────────────
%  All 8 directional kernels share (a1, a2, a3) but in different positions:
%    - a2 is always at the center [2,2] — acts as the gain factor
%    - a1 and a3 are placed at the two ends of each oriented axis
%    - Off-axis positions are zero
%
%  ── Brightness offset ────────────────────────────────────────────────────────
%  After convolution, +50 is added to every pixel to compensate for the
%  very low dynamic range of dark input images (I_mean < 64).
%
%  ── Inputs ──────────────────────────────────────────────────────────────────
%  img - Low-light input image (H × W × C, uint8)
%  a2  - Second coefficient from polynomial bounds (scalar)
%  a3  - Third coefficient from polynomial bounds (scalar)
%
%  ── Output ──────────────────────────────────────────────────────────────────
%  result - Enhanced image (H × W × C, uint8), pixel values clamped to [0, 255]

    % a₁ = 1 fixed by bi-univalent normalization (analytically confirmed)
    a1 = 1;

    % ── Construct 8 directional kernels ──────────────────────────────────────
    %
    %  h1 (0°, right):          h2 (45°, down-right):
    %  [ 0   0   0  ]           [ 0   0   a3 ]
    %  [ a1  a2  a3 ]           [ 0   a2  0  ]
    %  [ 0   0   0  ]           [ a1  0   0  ]
    %
    %  h3 (90°, down):          h4 (135°, down-left):
    %  [ 0   a1  0  ]           [ a1  0   0  ]
    %  [ 0   a2  0  ]           [ 0   a2  0  ]
    %  [ 0   a3  0  ]           [ 0   0   a3 ]
    %
    %  h5 (180°, left):         h6 (225°, up-left):
    %  [ 0   0   0  ]           [ a3  0   0  ]
    %  [ a3  a2  a1 ]           [ 0   a2  0  ]
    %  [ 0   0   0  ]           [ 0   0   a1 ]
    %
    %  h7 (270°, up):           h8 (315°, up-right):
    %  [ 0   a3  0  ]           [ 0   0   a1 ]
    %  [ 0   a2  0  ]           [ 0   a2  0  ]
    %  [ 0   a1  0  ]           [ a3  0   0  ]

    h1 = [0,  0,  0;  a1, a2, a3; 0,  0,  0 ];  % 0°   (right)
    h2 = [0,  0,  a3; 0,  a2, 0;  a1, 0,  0 ];  % 45°  (down-right)
    h3 = [0,  a1, 0;  0,  a2, 0;  0,  a3, 0 ];  % 90°  (down)
    h4 = [a1, 0,  0;  0,  a2, 0;  0,  0,  a3];  % 135° (down-left)
    h5 = [0,  0,  0;  a3, a2, a1; 0,  0,  0 ];  % 180° (left)
    h6 = [a3, 0,  0;  0,  a2, 0;  0,  0,  a1];  % 225° (up-left)
    h7 = [0,  a3, 0;  0,  a2, 0;  0,  a1, 0 ];  % 270° (up)
    h8 = [0,  0,  a1; 0,  a2, 0;  a3, 0,  0 ];  % 315° (up-right)

    % ── Compute Master Kernel ─────────────────────────────────────────────────
    % Sum all 8 kernels and divide by 8 (pre-average).
    % This is mathematically equivalent to applying all 8 individually and
    % averaging, but requires only 1 convolution per channel instead of 8.
    kernels     = cat(3, h1, h2, h3, h4, h5, h6, h7, h8);
    master_kernel = sum(kernels, 3) / 8;

    % ── Apply Master Kernel (1 convolution per channel) ───────────────────────
    img_d = double(img);
    [rows, cols, ch] = size(img_d);
    acc = zeros(rows, cols, ch);

    for c = 1:ch
        % conv2 'same' preserves image dimensions (no border shrinkage)
        acc(:, :, c) = conv2(img_d(:, :, c), master_kernel, 'same');
    end

    % ── Brightness offset + clamp ─────────────────────────────────────────────
    % +50 compensates for dark input range (I_mean < 64).
    % Fixed offset isolates the kernel's contribution independently of
    % post-processing normalization. 64 = 128/2 is the low-light threshold
    % (one exposure stop below the neutral midpoint 128).
    % Clamp to [0, 255] and cast back to uint8.
    result = uint8(min(max(acc + 50, 0), 255));

end