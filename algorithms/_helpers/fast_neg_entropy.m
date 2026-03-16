function val = fast_neg_entropy(x, img_d, I_edge, coeff_func, nu_min, nu_max, t_min, t_max)
%FAST_NEG_ENTROPY  Objective function for Nelder-Mead entropy maximization.
%
%  MATLAB's fminsearch minimizes its objective function. Since we want to
%  MAXIMIZE the Shannon entropy of the enhanced image, this function returns
%  the NEGATIVE entropy — so that minimizing -entropy is equivalent to
%  maximizing entropy.
%
%  ── Mathematical basis: kernel decomposition ────────────────────────────────
%  The Master Kernel separates into two static, parameter-independent components:
%
%      K_master = a₂ · δ  +  e · H_edge
%
%  where:
%      δ      = [0,0,0; 0,1,0; 0,0,0]   — Kronecker delta (identity filter)
%      H_edge = [1,1,1; 1,0,1; 1,1,1]   — 8-neighbor ring mask (static)
%      e      = (1 + a₃) / 8            — scalar coefficient (varies per eval)
%
%  By the distributive property of convolution and the identity I ∗ δ = I:
%
%      I ∗ K_master = a₂ · (I ∗ δ) + e · (I ∗ H_edge)
%                   = a₂ · img_d   + e · I_edge
%
%  This is an EXACT algebraic identity — not an approximation.
%  I_edge = I ∗ H_edge depends only on the image, not on (ν, t).
%  It is precomputed once in run_hermite.m before all optimization loops.
%
%  Per-evaluation cost in this function:
%    - 2 scalar-matrix multiplications: a₂·img_d and e·I_edge
%    - 1 element-wise addition
%    - 1 entropy calculation
%    - 0 conv2 calls, 0 imfilter calls, 0 kernel constructions
%
%  Complexity reduction:
%    Complexity: O(W·H·K²) + O(N · W·H)
%                — spatial K² cost fully isolated from the N-evaluation loop
%
%  ── Penalty for infeasible parameters ───────────────────────────────────────
%  If (ν, t) is out of bounds or produces invalid coefficients, a large penalty
%  (1e6) is returned, steering the Nelder-Mead simplex away from infeasible
%  regions.
%
%  ── Inputs ──────────────────────────────────────────────────────────────────
%  x          - [ν, t] parameter vector (1×2)
%  img_d      - double(img_low), precomputed before the optimization loop
%  I_edge     - imfilter(img_d, H_edge, 'replicate'), precomputed before loop
%               H_edge = [1,1,1; 1,0,1; 1,1,1]
%  coeff_func - Handle to coefficient function (e.g. @hermite_coefficients)
%               Signature: [a2, a3, valid] = coeff_func(nu, t)
%  nu_min/max - Valid ν range; values outside → penalty returned
%  t_min/max  - Valid t range; values outside → penalty returned
%
%  ── Output ──────────────────────────────────────────────────────────────────
%  val - Negative Shannon entropy of the enhanced image (scalar)
%        OR 1e6 (penalty) if parameters are infeasible
    PENALTY = 1e6;

    nu = x(1);
    t  = x(2);

    % ── Boundary check ───────────────────────────────────────────────────────
    if nu < nu_min || nu > nu_max || t < t_min || t > t_max
        val = PENALTY;
        return;
    end

    % ── Coefficient validity check ───────────────────────────────────────────
    [a2, a3, valid] = coeff_func(nu, t);
    if ~valid
        val = PENALTY;
        return;
    end

    % ── Fast enhanced image: no conv2 call ───────────────────────────────────
    % Equivalent to apply_convolution(img_low, a2, a3) but uses precomputed
    % spatial components. Each call: 2 scalar multiplications + element-wise add.
    edge_factor = (1 + a3) / 8;
    acc = a2 * img_d + edge_factor * I_edge + 50;
    candidate = uint8(min(max(round(acc), 0), 255));

    % ── Return negative entropy ──────────────────────────────────────────────
    val = -calc_entropy(candidate);

end