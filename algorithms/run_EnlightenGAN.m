function enhanced = run_EnlightenGAN(img_low)
%RUN_ENLIGHTENGAN  Comparison baseline: EnlightenGAN (unsupervised deep enhancement).
%
%  Jiang Y., Gong X., Liu D., Cheng Y., Fang C., Shen X., Yang J., Zhou P., Wang Z.
%  "EnlightenGAN: Deep Light Enhancement without Paired Supervision."
%  IEEE Transactions on Image Processing, 30, 2340-2349, 2021.
%  arXiv: 1906.06972
%  Official code: https://github.com/VITA-Group/EnlightenGAN
%
%  Inference package: arsenyinfo/EnlightenGAN-inference (ONNX, CPU)
%  https://github.com/arsenyinfo/EnlightenGAN-inference
%  Located at: algorithms/enlightengan/enlighten_inference/
%
%  Integration pattern:
%    Writes img_low to a temporary PNG, calls run_enlightengan.py via
%    system(), reads the result back, returns it as uint8 — identical
%    to every other comparison algorithm in the pipeline.
%
%  Input:
%    img_low  - Low-light image (uint8, H x W x 3)
%
%  Output:
%    enhanced - Enhanced image (uint8, H x W x 3)

    % ── Detect Python command (Windows: py / python; Unix: python3 / python) ──
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
        error('run_EnlightenGAN: Python not found on PATH.');
    end

    % ── Paths ─────────────────────────────────────────────────────────────────
    script_dir = fileparts(mfilename('fullpath'));
    py_script  = fullfile(script_dir, 'enlightengan', 'run_enlightengan.py');
    tmp_in     = fullfile(tempdir, sprintf('egan_in_%d.png',  randi(1e8)));
    tmp_out    = fullfile(tempdir, sprintf('egan_out_%d.png', randi(1e8)));

    % ── Write temporary input image ───────────────────────────────────────────
    imwrite(img_low, tmp_in);

    % ── Call Python inference script ──────────────────────────────────────────
    cmd    = sprintf('%s "%s" "%s" "%s"', py_cmd, py_script, tmp_in, tmp_out);
    status = system(cmd);

    % ── Clean up input temp file ──────────────────────────────────────────────
    if exist(tmp_in, 'file'), delete(tmp_in); end

    if status ~= 0
        if exist(tmp_out, 'file'), delete(tmp_out); end
        error(['run_EnlightenGAN: Python script failed (exit code %d).\n' ...
               'Make sure algorithms/enlightengan/enlighten_inference/ exists\n' ...
               'and onnxruntime + opencv-python are installed.'], status);
    end

    % ── Read enhanced output ──────────────────────────────────────────────────
    enhanced = imread(tmp_out);
    if exist(tmp_out, 'file'), delete(tmp_out); end

    % Ensure uint8 RGB output
    if ~isa(enhanced, 'uint8')
        if max(enhanced(:)) <= 1
            enhanced = uint8(double(enhanced) * 255);
        else
            enhanced = uint8(enhanced);
        end
    end
    if size(enhanced, 3) == 1
        enhanced = repmat(enhanced, [1, 1, 3]);
    end

end