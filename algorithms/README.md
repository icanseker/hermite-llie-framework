# Algorithms — Source Documentation

This file documents every algorithm in this directory: where the code came from,
what was included, what was excluded, and what (if anything) was modified.

---

## `_helpers/` — Shared Utilities

### `apply_convolution.m`
Applies the 8-directional Master Kernel to a single image. Called **once per image** to produce the final enhanced output using the optimal (ν*, t*). Not called inside the optimization loop.

The Master Kernel is a mathematically derived 3×3 filter equivalent to the average of 8 directional kernels:

```
K_master = [ (1+a₃)/8,  (1+a₃)/8,  (1+a₃)/8 ]
           [ (1+a₃)/8,    a₂,       (1+a₃)/8 ]
           [ (1+a₃)/8,  (1+a₃)/8,  (1+a₃)/8 ]
```

### `fast_neg_entropy.m`
High-speed objective function for the optimization loop. Uses precomputed spatial components `(img_d, I_edge)` to avoid `conv2` inside the loop entirely.

**Mathematical basis:**

```
I * K_master  =  a₂ · img_d  +  (1+a₃)/8 · I_edge

where:
  img_d  = double(img_low)            — precomputed in run_hermite.m (once)
  I_edge = conv(img_d, H_edge)        — precomputed in run_hermite.m (once)
  H_edge = [1,1,1; 1,0,1; 1,1,1]     — 8-neighbor static kernel
```

This is an exact algebraic identity. Each call reduces to two scalar multiplications and one element-wise addition. For ~5000 evaluations (high_precision), this eliminates ~5000 `conv2` calls from the loop.

---

---

## Folder Structure

```
algorithms/
├── run_hermite.m          Proposed method wrapper
├── run_HE.m               Written by us
├── run_SSR.m              Written by us
├── run_gamma.m            Written by us
├── run_LIME.m             Written by us (calls original lime_main_module())
├── run_dong.m             Written by us
├── run_BIMEF.m            Written by us (calls original BIMEF())
├── run_NPE.m              Written by us (calls NPEA.p from baidut/BIMEF)
├── run_LECARM.m           Written by us (calls original LECARM())
├── run_EnlightenGAN.m     Written by us (calls run_enlightengan.py via system())
│
├── _helpers/              Shared utilities (written by us)
│   ├── apply_convolution.m   Final-pass Master Kernel convolution (called once per image)
│   ├── fast_neg_entropy.m    High-speed objective for optimization loop (no conv2)
│
├── bimef/                 BIMEF official files (baidut/BIMEF, unchanged)
│   ├── BIMEF.m
│   └── BIMEF.p
│
├── lime/                  LIME official files (estija/LIME, unchanged)
│   └── *.m  (12 files: lime.m, lime_main_module.m, lime_trial.m, ...)
│
├── npe/                   NPE compiled .p files (baidut/BIMEF, unchanged)
│   ├── NPEA.p  BLT.p  BiFltL.p  Post.p  cbright.p  getextpic.p  getlocalmax.p
│
├── lecarm/                LECARM source (see below)
│   ├── LECARM.m
│   ├── CameraModel.m
│   ├── limeEstimate.m
│   └── +CameraModels/
│       ├── Sigmoid.m
│       ├── Beta.m
│       ├── BetaGamma.m
│       ├── Gamma.m
│       └── Preferred.m
│
└── enlightengan/          EnlightenGAN source (see below)
    ├── run_enlightengan.py
    └── enlighten_inference/
```

---

## Algorithm-by-Algorithm Source Notes

### HE, SSR, Gamma, Dong
Written entirely by us using MATLAB built-in functions.
No external code copied.

---

### LIME
**Paper:** Guo X., Li Y., Ling H. "LIME: Low-Light Image Enhancement via
Illumination Map Estimation." IEEE Transactions on Image Processing, 26(2),
982–993, 2017.

**Source:** estija/LIME — https://github.com/estija/LIME

Note: The official LIME code by the original authors is distributed as
compiled .p files (source not readable). estija/LIME is a clean open-source
MATLAB implementation of the same paper.

**What was copied from the repo (unchanged):**

| File | Role |
|------|------|
| `lime.m` | Main LIME function: initial map → ADMM → gamma correction |
| `lime_main_module.m` | Full pipeline including bilateral denoising |
| `lime_trial.m` | ADMM solver loop (50 iterations) |
| `initial_map.m` | Initial illumination map: T_hat = max(R,G,B) |
| `make_weight_matrix.m` | RTV-inspired weight matrix (Eq. 9) |
| `updateT.m` | T sub-problem solver (FFT domain) |
| `multiplyd.m` | Forward difference operator D*T |
| `multiplydtrans.m` | Adjoint operator D'*X |
| `shrinkage.m` | Soft thresholding for G sub-problem |
| `Tdenom.m` | FFT denominator for T update |
| `maked_alt.m` | Gradient matrix constructor |
| `gamma_corr.m` | Gamma correction on illumination map |

**Not copied:**
- `BM3D/` — only used for comparison with bilateral filter; we use `imbilatfilt`
- `imgs/`, `assets/` — demo images and figures
- `histeq_*.m`, `hist_sep.m`, `rgb2yuv.m`, `yuv2rgb.m` — comparison methods
- `lime_loop.m`, `lime_bf_loop.m` — parameter tuning scripts
- `README.md` — original repo README

No modifications were made to any copied file.

---

### BIMEF
**Paper:** Ying Z., Li G., Gao W. "A Bio-Inspired Multi-Exposure Fusion
Framework for Low-light Image Enhancement." IEEE Transactions on
Cybernetics, 50(6), 2400–2414, 2020. arXiv: 1711.00591

**Official code:** https://github.com/baidut/BIMEF

**What was copied from the repo (unchanged):**

| File | Source | Note |
|------|--------|------|
| `BIMEF.m` | `lowlight/BIMEF.m` | Full readable source code |
| `BIMEF.p` | `lowlight/BIMEF.p` | Compiled version — MATLAB uses .p when both exist |

`run_BIMEF.m` calls `BIMEF()` directly with default parameters.
No custom implementation — original files used as-is.

**Not copied:** `demo.m`, `experiments.m`, `startup.m`, `util/`, `quality/`,
`lowlight/` comparison methods (dong, lime, npe, etc.), `example.jpg`, `boxplot.jpg`.

---

### NPE
**Paper:** Wang S., Zheng J., Hu H.-M., Li B. "Naturalness Preserved
Enhancement Algorithm for Non-Uniform Illumination Images."
IEEE Transactions on Image Processing, 22(9), 3538–3548, 2013.
DOI: 10.1109/TIP.2013.2261309

**Source:** baidut/BIMEF — https://github.com/baidut/BIMEF
(`lowlight/NPE/` subfolder)

The original NPE code was never released by the authors as readable source.
It is distributed exclusively as compiled MATLAB `.p` files (encrypted
bytecode). The `.p` files in `algorithms/npe/` were obtained from the
baidut/BIMEF repository, which is the standard reference codebase for
low-light image enhancement benchmarking in the literature.

**What was copied (unchanged):**

| File | Role |
|------|------|
| `NPEA.p` | Top-level entry point — full NPE pipeline |
| `BLT.p` | Bi-log transformation (illumination mapping) |
| `BiFltL.p` | Bright-pass filter (illumination estimation) |
| `Post.p` | Post-processing (reflectance × mapped illumination) |
| `cbright.p` | Brightness computation helper |
| `getextpic.p` | Image extent/padding helper |
| `getlocalmax.p` | Local maximum computation helper |

**Authenticity verification:**
To confirm that these `.p` files produce results consistent with the
original Wang 2013 paper, we ran them on the paper's own test dataset
(Road, Birds, Rail, Skyscraper, Sculpture, Nightfall, Harbor, Parking Area)
and compared the output entropy values against Table I of the paper.
The entropy values matched within the expected range, confirming that
these are the original author's compiled files.

The test dataset was obtained from the BIMEF project's Google Drive:
https://drive.google.com/drive/folders/1YujA_GjKigwSJXbRJTkO1l2p1gMmm8ai
(BIMEF datasets folder — NPE subfolder)

---

### LECARM
**Paper:** Ying Z., Li G., Ren Y., Wang R., Wang W. "A New Image Contrast
Enhancement Algorithm Using Exposure Fusion Framework." Computer Analysis
of Images and Patterns (CAIP), 2017.

**Official code:** https://github.com/baidut/LECARM

**What was copied from the repo (`LECARM-master/`):**

| File | Copied as |
|------|-----------|
| `LECARM.m` | `algorithms/lecarm/LECARM.m` |
| `CameraModel.m` | `algorithms/lecarm/CameraModel.m` |
| `limeEstimate.m` | `algorithms/lecarm/limeEstimate.m` |
| `+CameraModels/Sigmoid.m` | `algorithms/lecarm/+CameraModels/Sigmoid.m` |
| `+CameraModels/Beta.m` | `algorithms/lecarm/+CameraModels/Beta.m` |
| `+CameraModels/BetaGamma.m` | `algorithms/lecarm/+CameraModels/BetaGamma.m` |
| `+CameraModels/Gamma.m` | `algorithms/lecarm/+CameraModels/Gamma.m` |
| `+CameraModels/Preferred.m` | `algorithms/lecarm/+CameraModels/Preferred.m` |

**Not copied:**
- `run.m` — interactive demo script, not relevant to our pipeline
- `image/` — demo images used by run.m
- `README.md` — original repo README
- `subjective.jpg` — comparison figure from the original paper

**One modification made to `limeEstimate.m`:**
Line 32 originally calls `gaussmf()` from the Fuzzy Logic Toolbox.
Replaced with the inline mathematical equivalent:
```matlab
% Original:  fil = gaussmf(temp, [sigma, 0]);
% Replaced:
fil = exp(-0.5 * (temp / sigma).^2);
```
`gaussmf(x,[sig,c]) = exp(-0.5*((x-c)/sig)^2)`. With c=0 this reduces
to `exp(-0.5*(x/sig)^2)`. Numerically identical, no toolbox required.
No other changes were made to any LECARM file.

---

### EnlightenGAN
**Paper:** Jiang Y., Gong X., Liu D., Cheng Y., Fang C., Shen X., Yang J.,
Zhou P., Wang Z. "EnlightenGAN: Deep Light Enhancement without Paired
Supervision." IEEE Transactions on Image Processing, 30, 2340–2349, 2021.
arXiv: 1906.06972

**Official code:** https://github.com/VITA-Group/EnlightenGAN

**Nothing was copied from the official repository.**

The official repo requires PyTorch, GPU, and manual weight downloads.
Instead, we use the ONNX-based inference package by arsenyinfo:
https://github.com/arsenyinfo/EnlightenGAN-inference

This package contains the same pretrained model converted to ONNX format
and only requires `onnxruntime` as the inference engine. It runs on CPU
with no GPU required.

**What was copied:**

The `enlighten_inference/` package from the arsenyinfo repo was copied
directly into `algorithms/enlightengan/enlighten_inference/`. This is the
package that would be installed via:
```
pip install git+https://github.com/arsenyinfo/EnlightenGAN-inference
```

| File | Description |
|------|-------------|
| `enlighten_inference/__init__.py` | `EnlightenOnnxModel` class — loads the ONNX model and runs inference |
| `enlighten_inference/enlighten.onnx` | Pretrained EnlightenGAN weights in ONNX format |

No modifications were made to either file.

`algorithms/enlightengan/run_enlightengan.py` was written by us. It is a
thin CLI wrapper: takes `<input_path>` and `<output_path>` as arguments,
calls `EnlightenOnnxModel.predict()`, and writes the result to disk.

**Integration pattern:**
`run_EnlightenGAN.m` writes `img_low` to a temporary PNG, calls
`run_enlightengan.py` via `system()`, reads the enhanced result back,
and returns it as `uint8` to `main.m` — identical to every other
comparison algorithm. No pre-computation step, no separate output
folder, and no `basename` parameter is needed.