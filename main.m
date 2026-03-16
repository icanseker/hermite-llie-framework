%% main.m — Low-Light Image Enhancement: Hermite Polynomial Method
%
%  Paper: "A Computationally Optimized and Dynamically Refined Framework for
%          Structure-Preserving Low-Light Image Enhancement via Hermite
%          Polynomial Analytic Functions"
%
%  ── Overview ───────────────────────────────────────────────────────────────
%  This script orchestrates the full experiment:
%    1. Scans source/low/ for all image files
%    2. For each image, computes mean intensity and applies a threshold gate
%    3. Runs the proposed Hermite method (if I_mean < threshold) and all
%       9 comparison algorithms
%    4. Computes evaluation metrics for each result
%    5. Saves enhanced images and metric reports to target/
%
%  ── Architecture ───────────────────────────────────────────────────────────
%  main.m handles ALL orchestration: directory scanning, loop control,
%  metric computation, file I/O, and console reporting.
%
%  Each algorithm file (run_*.m) is a PURE FUNCTION:
%    Input:  single low-light image + preset string (proposed method)
%             or single low-light image only (comparison methods)
%    Output: enhanced image (and optimal params + evals for proposed method)
%  Algorithm functions know nothing about directories, metrics, or file I/O.
%  This separation keeps each component independently readable and testable.
%
%  ── Dataset ────────────────────────────────────────────────────────────────
%  Default images: LOL (Low-Light) evaluation set, 15 paired images (400×600).
%  All 15 images have I_mean well below the threshold of 64, so all 15 are
%  processed by the proposed Hermite method and all comparison algorithms.
%
%  Custom images: Place any images in source/low/. If matching ground truth
%  exists in source/high/ (same filename), full-reference metrics are computed.
%  Without a reference, PSNR/MSE/SSIM are reported as N/A.
%
%  ── Output ─────────────────────────────────────────────────────────────────
%  target/{AlgorithmName}/{id}_enhanced.png   — enhanced image
%  target/{AlgorithmName}/{id}_metrics.txt    — metric values + optimal params
%
%  See README.md for full documentation.

clc; clear; close all;

%% ═══════════════════════════════════════════════════════════════════════════
%  CONFIGURATION
%  Adjust these parameters to control experiment behavior and speed/quality.
%  ═══════════════════════════════════════════════════════════════════════════

% Directory paths
src_low_dir  = fullfile(pwd, 'source', 'low');   % Input low-light images
src_high_dir = fullfile(pwd, 'source', 'high');  % Ground truth references (optional)
target_dir   = fullfile(pwd, 'target');          % Output directory (auto-created)

% ── Brightness threshold ───────────────────────────────────────────────────
% LOW-LIGHT DEFINITION: I_mean < 64 (= 128/2, one stop below neutral midpoint 128).
% Images at or above this threshold are SKIPPED by the proposed method —
% a skip notice is written to the metrics file instead of an enhanced image.
% Comparison algorithms (HE, SSR, etc.) run on ALL images regardless.
I_MEAN_THRESHOLD = 64;   % 64 = 128/2: one stop below neutral midpoint 128

% ── Precision Preset ─────────────────────────────────────────────────────
% Controls the full optimization pipeline: Phase 1 grid density AND
% Phase 2 Nelder-Mead tolerances are both resolved inside run_hermite.m.
% No other edits needed to switch modes.
%
% All presets benefit from spatial precomputation (I_edge computed once
% before the optimization loop). The freed budget is used for finer grids,
% not fewer evaluations — same wall-clock time, higher parameter resolution.
%
%   'high_precision'  — nu=0.015, t=0.05  (~5000 grid pts + 5 NM runs)
%                       Covers parameter space 6× more densely than before.
%                       Use for all published results.
%   'balanced'        — nu=0.030, t=0.08  (~1500 grid pts + 3 NM runs)
%                       Development, validation, non-paper experiments.
%   'fast'            — nu=0.100, t=0.20  (~200 grid pts + 1 NM run)
%                       Real-time, live video, large-batch deployment.
%
% See run_hermite.m for exact parameter values and timing estimates.
optimization_preset = 'high_precision';


%% ═══════════════════════════════════════════════════════════════════════════
%  ALGORITHM REGISTRY
%  Each entry: {name, function_handle, type}
%    'polynomial' : proposed method — takes (img_low, preset), resolves all
%                   optimization params internally, returns [enhanced, best_nu, best_t, total_evals]
%    'comparison' : fixed-parameter baseline — returns enhanced image only
%  ═══════════════════════════════════════════════════════════════════════════
algorithms = {
    'Hermite', @run_hermite, 'polynomial';   % Proposed: Hermite coefficients + 8-dir convolution
    'HE',      @run_HE,      'comparison';   % Baseline: Histogram Equalization
    'SSR',     @run_SSR,     'comparison';   % Baseline: Single-Scale Retinex (sigma=80)
    'Gamma',   @run_gamma,   'comparison';   % Baseline: Gamma Correction (gamma=0.4)
    'LIME',    @run_LIME,    'comparison';   % Baseline: LIME ADMM solver (alpha=0.15, gamma=0.8)
    'Dong',    @run_dong,    'comparison';   % Baseline: Inverted Dehazing (approxdcp)
    'BIMEF',        @run_BIMEF,         'comparison';   % Modern: Bio-Inspired Multi-Exposure Fusion (Ying 2020)
    'NPE',          @run_NPE,           'comparison';   % Modern: Naturalness Preserved Enhancement (Wang 2013)
    'LECARM',       @run_LECARM,        'comparison';   % Modern: Camera Response Model Enhancement (Ying 2017)
    'EnlightenGAN', @run_EnlightenGAN,  'comparison';   % Deep: Unsupervised GAN (Jiang 2021, IEEE TIP)
};


%% ═══════════════════════════════════════════════════════════════════════════
%  CLEAN TARGET DIRECTORY
%  If target/ already exists and contains results from a previous run,
%  clear it entirely before starting. This prevents old and new results
%  from mixing — especially important when re-running with different
%  preset or a different image set.
%  ═══════════════════════════════════════════════════════════════════════════
if exist(target_dir, 'dir')
    % Remove the entire target directory tree and recreate it empty
    rmdir(target_dir, 's');
    fprintf('Cleared previous results from: %s\n', target_dir);
end
mkdir(target_dir);


%% ═══════════════════════════════════════════════════════════════════════════
%  ADD PATHS
%  Structure:
%    algorithms/          — run_*.m wrappers (one per algorithm)
%    algorithms/_helpers/ — shared utilities: apply_convolution, fast_neg_entropy
%    algorithms/bimef/    — BIMEF.m + BIMEF.p (official files, baidut/BIMEF)
%    algorithms/lime/     — 12 .m files (official files, estija/LIME)
%    algorithms/npe/      — NPEA.p + supporting .p files (baidut/BIMEF)
%    algorithms/lecarm/   — official LECARM source (unchanged)
%    algorithms/enlightengan/ — EnlightenGAN Python source
%    metrics/             — calc_NIQE and other metric helpers
%  ═══════════════════════════════════════════════════════════════════════════
alg = fullfile(pwd, 'algorithms');
addpath(alg);
addpath(fullfile(alg, '_helpers'));
addpath(fullfile(alg, 'bimef'));
addpath(fullfile(alg, 'lime'));
addpath(fullfile(alg, 'npe'));
addpath(fullfile(alg, 'lecarm'));
addpath(fullfile(pwd, 'metrics'));

%% ═══════════════════════════════════════════════════════════════════════════
%  PYTHON DEPENDENCY CHECK (EnlightenGAN)
%  Checks whether onnxruntime and opencv-python are installed.
%  If either is missing, installs it automatically via pip before proceeding.
%  This runs once at startup and takes only a second if already installed.
%  ═══════════════════════════════════════════════════════════════════════════
fprintf('Checking Python dependencies for EnlightenGAN...\n');

% Detect Python command
if ispc
    py_candidates = {'py', 'python', 'python3'};
else
    py_candidates = {'python3', 'python'};
end
py_cmd = '';
for i = 1:numel(py_candidates)
    [s, ~] = system(sprintf('%s --version', py_candidates{i}));
    if s == 0, py_cmd = py_candidates{i}; break; end
end

if isempty(py_cmd)
    warning(['Python not found on PATH. EnlightenGAN will be skipped.\n' ...
             'Install Python from https://python.org to enable it.']);
else
    py_deps = {'onnxruntime', 'cv2'};
    py_pkgs = {'onnxruntime', 'opencv-python'};
    for i = 1:numel(py_deps)
        chk = sprintf('%s -c "import %s" 2>&1', py_cmd, py_deps{i});
        [s, ~] = system(chk);
        if s ~= 0
            fprintf('  Installing %s...\n', py_pkgs{i});
            system(sprintf('%s -m pip install %s', py_cmd, py_pkgs{i}));
        else
            fprintf('  %s: OK\n', py_pkgs{i});
        end
    end
    fprintf('Python dependencies ready.\n');
end
fprintf('\n');


%% ═══════════════════════════════════════════════════════════════════════════
%  SCAN SOURCE DIRECTORY
%  Find all image files in source/low/. The algorithm processes whatever
%  images it finds — it is not limited to the default LOL dataset.
%  ═══════════════════════════════════════════════════════════════════════════
supported_ext = {'*.png','*.jpg','*.jpeg','*.bmp','*.tif','*.tiff'};
file_cells = cell(length(supported_ext), 1);
for e = 1:length(supported_ext)
    file_cells{e} = dir(fullfile(src_low_dir, supported_ext{e}));
end
file_list = vertcat(file_cells{:});

if isempty(file_list)
    error('No images found in: %s\nPlace low-light images there and re-run.', src_low_dir);
end

num_images = length(file_list);
fprintf('=== Hermite Low-Light Enhancement ===\n\n');
fprintf('Source : %s  (%d images)\n', src_low_dir, num_images);
fprintf('Threshold : I_mean < %d  [= 128/2, one stop below neutral midpoint 128]\n', I_MEAN_THRESHOLD);
fprintf('Preset       : %s\n\n', optimization_preset);


%% ═══════════════════════════════════════════════════════════════════════════
%  MAIN LOOP: Per image, per algorithm
%  ═══════════════════════════════════════════════════════════════════════════
for i = 1:num_images

    % ── Identify this image ─────────────────────────────────────────────────
    fname              = file_list(i).name;
    [~, basename, ext] = fileparts(fname);
    low_path           = fullfile(src_low_dir,  fname);
    high_path          = fullfile(src_high_dir, fname);
    has_ref            = isfile(high_path);   % true if ground truth exists

    % ── Load input image ────────────────────────────────────────────────────
    img_low = imread(low_path);

    % ── Compute mean intensity (grayscale) ─────────────────────────────────
    % I_mean drives the threshold decision for the proposed method.
    % It is computed on the raw uint8 image values (0–255 range).
    I_mean = mean(double(img_low(:)));

    % ── Load ground truth (if available) ───────────────────────────────────
    % PSNR, MSE, and SSIM require a reference image.
    % If source/high/{fname} does not exist, these metrics are reported as N/A.
    img_high = [];
    if has_ref
        img_high = imread(high_path);
        fprintf('[%d/%d] %s  (I_mean=%.1f, ref=YES)\n', i, num_images, fname, I_mean);
    else
        fprintf('[%d/%d] %s  (I_mean=%.1f, ref=NO → PSNR/MSE/SSIM will be N/A)\n', ...
                i, num_images, fname, I_mean);
    end

    % ── Threshold gate for proposed method ─────────────────────────────────
    % The 8-directional convolution approach is designed for very dark images.
    % Low-light is defined as I_mean < 64 (= 128/2, one stop below neutral 128).
    % Images at or above 64 are outside the intended operating range.
    skip_polynomial = (I_mean >= I_MEAN_THRESHOLD);
    if skip_polynomial
        fprintf('  I_mean=%.1f >= %d → Hermite will be SKIPPED for this image\n', ...
                I_mean, I_MEAN_THRESHOLD);
    end

    % ── Per-algorithm processing ─────────────────────────────────────────
    for a = 1:size(algorithms, 1)
        alg_name = algorithms{a, 1};
        alg_func = algorithms{a, 2};
        alg_type = algorithms{a, 3};

        % Skip proposed method for images above the brightness threshold
        if strcmp(alg_type, 'polynomial') && skip_polynomial
            write_skipped(target_dir, alg_name, fname, basename, I_mean, I_MEAN_THRESHOLD);
            fprintf('  %-10s → SKIPPED (I_mean >= threshold)\n', alg_name);
            continue;
        end

        % ── Call the algorithm ─────────────────────────────────────────────
        % Polynomial (proposed) methods return optimal params and evaluation count.
        % Comparison methods return only the enhanced image.
        if strcmp(alg_type, 'polynomial')
            % Time the proposed method — records wall-clock seconds from
            % first coefficient evaluation to final enhanced image output.
            % This covers Phase 1 (grid search) + Phase 2 (Nelder-Mead) + final conv.
            t_start = tic;
            [enhanced, best_nu, best_t, total_evals] = ...
                alg_func(img_low, optimization_preset);
            elapsed_sec = toc(t_start);
        else
            % Comparison algorithms use fixed parameters — no optimization loop.
            enhanced = alg_func(img_low);
            elapsed_sec = NaN;  % Timing not recorded for comparison methods
        end

        % ── Compute no-reference metrics ──────────────────────────────────
        % Entropy, CII, and NIQE do not require ground truth, so they are
        % always computed. NIQE: lower = more natural. Entropy: higher = richer.
        entropy_val = calc_entropy(enhanced);
        cii_val     = calc_CII(img_low, enhanced);
        niqe_val    = calc_NIQE(enhanced);

        % ── Compute full-reference metrics (if ground truth available) ─────
        % PSNR, MSE, SSIM all require the reference image.
        % If no reference, these are stored as NaN and reported as N/A.
        if has_ref
            psnr_val = calc_PSNR(enhanced, img_high);
            mse_val  = calc_MSE(enhanced, img_high);
            ssim_val = calc_SSIM(enhanced, img_high);
        else
            psnr_val = NaN; mse_val = NaN; ssim_val = NaN;
        end

        % ── Save enhanced image ────────────────────────────────────────────
        % Each algorithm gets its own subdirectory under target/.
        % Output filename preserves the original image ID.
        alg_dir = fullfile(target_dir, alg_name);
        if ~exist(alg_dir, 'dir'), mkdir(alg_dir); end
        out_ext = ext;
        if isempty(out_ext), out_ext = '.png'; end
        imwrite(enhanced, fullfile(alg_dir, [basename '_enhanced' out_ext]));

        % ── Write metrics report ───────────────────────────────────────────
        % One .txt file per (image, algorithm) pair.
        % Includes optimal (nu, t) for the proposed method.
        fid = fopen(fullfile(alg_dir, [basename '_metrics.txt']), 'w');
        fprintf(fid, 'Image: %s\n', fname);
        fprintf(fid, 'Algorithm: %s\n', alg_name);
        fprintf(fid, 'I_mean: %.2f\n', I_mean);
        fprintf(fid, 'Status: PROCESSED\n');
        if strcmp(alg_type, 'polynomial')
            fprintf(fid, 'Optimal_nu: %.4f\n', best_nu);
            fprintf(fid, 'Optimal_t: %.4f\n', best_t);
            fprintf(fid, 'Total_evals: %d\n', total_evals);
            % Preset and timing — key for precision-vs-speed analysis
            fprintf(fid, 'Preset: %s\n', optimization_preset);
            fprintf(fid, 'Processing_time_sec: %.4f\n', elapsed_sec);
        else
            fprintf(fid, 'Optimal_nu: -\n');
            fprintf(fid, 'Optimal_t: -\n');
        end
        fprintf(fid, '\n--- Metrics ---\n');
        fprintf(fid, 'Entropy: %.4f\n', entropy_val);
        fprintf(fid, 'CII: %.4f\n', cii_val);
        fprintf(fid, 'NIQE: %.4f\n', niqe_val);
        if has_ref
            fprintf(fid, 'PSNR: %.4f\n', psnr_val);
            fprintf(fid, 'MSE: %.4f\n', mse_val);
            fprintf(fid, 'SSIM: %.4f\n', ssim_val);
        else
            fprintf(fid, 'PSNR: N/A\n');
            fprintf(fid, 'MSE: N/A\n');
            fprintf(fid, 'SSIM: N/A\n');
            fprintf(fid, 'Note: No matching reference image found in source/high/.\n');
            fprintf(fid, '      PSNR, MSE, SSIM require a ground truth reference image.\n');
            fprintf(fid, '      Entropy and CII are no-reference metrics and computed above.\n');
        end
        fclose(fid);

        % ── Console summary line ───────────────────────────────────────────
        if has_ref
            fprintf('  %-10s → SSIM=%.4f  PSNR=%.2f dB  Entropy=%.4f  NIQE=%.2f', ...
                    alg_name, ssim_val, psnr_val, entropy_val, niqe_val);
        else
            fprintf('  %-10s → Entropy=%.4f  CII=%.2f  NIQE=%.2f', ...
                    alg_name, entropy_val, cii_val, niqe_val);
        end
        if strcmp(alg_type, 'polynomial')
            fprintf('  |  nu=%.4f  t=%.4f  evals=%d', best_nu, best_t, total_evals);
        end
        fprintf('\n');

    end % algorithm loop
    fprintf('\n');

end % image loop

fprintf('=== Experiment complete. Results saved to: %s ===\n', target_dir);


%% ═══════════════════════════════════════════════════════════════════════════
%  LOCAL HELPER: write_skipped
%  Writes a skip-notice metrics file when a polynomial method is not applied
%  because the image's mean intensity is above the threshold.
%  ═══════════════════════════════════════════════════════════════════════════
function write_skipped(target_dir, alg_name, fname, basename, I_mean, threshold)
%WRITE_SKIPPED  Write a metrics file indicating the image was skipped.
%
%  This occurs when the image's mean intensity (I_mean) is >= the threshold
%  (default 64 = 128/2). Low-light is defined as I_mean < 64. The proposed Hermite
%  images; applying it to moderately or well-lit images is outside its
%  intended operating range and may produce over-enhanced or distorted results.

    alg_dir = fullfile(target_dir, alg_name);
    if ~exist(alg_dir, 'dir'), mkdir(alg_dir); end

    fid = fopen(fullfile(alg_dir, [basename '_metrics.txt']), 'w');
    fprintf(fid, 'Image: %s\n', fname);
    fprintf(fid, 'Algorithm: %s\n', alg_name);
    fprintf(fid, 'I_mean: %.2f\n', I_mean);
    fprintf(fid, 'Status: SKIPPED\n');
    fprintf(fid, 'Reason: I_mean (%.2f) >= threshold (%d).\n', I_mean, threshold);
    fprintf(fid, 'Note: Low-light is defined as I_mean < %d (= 128/2, one stop below neutral 128).\n', threshold);
    fclose(fid);
end