# FDM Channel Encoding

Ossi Lehtinen, Ocon Oy — <ossi@ocon.fi>

A frequency-division-multiplexing–inspired input encoding for multi-signal
transformers. Each channel gets a non-overlapping block of embedding
dimensions, and its scalar value amplitude-modulates a channel-specific,
RoPE-style sinusoidal carrier. Blocks are concatenated rather than summed,
so channel identity is preserved by construction.

This repo contains:
- A PyTorch implementation of four input encoders (`sum`, `concat`, `fdm`,
  `fdm-learn`) sharing the same transformer backbone.
- A synthetic multi-signal benchmark where the outcome is a lagged,
  non-linear function of several channels, binned for a categorical
  generative head.
- A sweep runner, a `d_model` scaling study, a carrier-band grid search,
  a linear-probe channel-recovery experiment, a test-time channel-masking
  experiment, and a plotting script.
- A LaTeX manuscript (`paper/`) with all tables and figures regenerated
  from experiment outputs.

---

## Headline results

At C=4 channels, mean ± std over 5 seeds, 100 epochs with cosine LR decay,
same transformer backbone (~152K params) for all encoders. Metric is the
best val NLL achieved during training (effective early stopping).

Val NLL = validation negative log-likelihood (cross-entropy of the
categorical generative head); val acc = top-1 bin accuracy.

| encoder                          | class        | enc. params | val NLL ↓        | val acc ↑        |
| -------------------------------- | ------------ | ----------- | ---------------- | ---------------- |
| `sum` (baseline, shared W)       | summation    | 320         | 3.253 ± 0.007    | 0.092 ± 0.003    |
| `ci` (channel-independent)       | arch.        | 2,112       | 3.051 ± 0.008    | 0.140 ± 0.003    |
| `fdm` (block + carrier)          | block        | 0           | 2.854 ± 0.018    | 0.145 ± 0.004    |
| `cat` (channel-as-token, fixed mask) | arch.    | 320         | 2.738 ± 0.040    | 0.161 ± 0.006    |
| **`concat`** (block, no carrier) | block        | 128         | **2.429 ± 0.038**| **0.201 ± 0.007**|
| **`sum-perch`** (per-channel W_k) | summation   | 512         | **2.422 ± 0.021**| **0.202 ± 0.004**|
| **`sum-ortho`** (perch + λ=1e-2) | sum + reg    | 512         | **2.422 ± 0.021**| **0.202 ± 0.004**|

Random baseline: NLL = ln 32 ≈ 3.466, acc = 1/32 ≈ 0.031.

Linear-probe channel recovery (R² from a frozen hidden state back to the
raw input, mean over channels):

| encoder      | input layer | after 3 layers |
| ------------ | ----------- | -------------- |
| `sum`        | 0.240       | 0.234          |
| `fdm`        | 1.000       | 0.925          |
| `concat`     | 0.999       | **0.937**      |
| `sum-perch`  | 1.000       | 0.926          |
| `sum-ortho`  | 1.000       | 0.926          |

### Real-data validation: ETTh1

Same ordering reproduces on ETTh1 (7 variates, next-step bin prediction
of oil temperature):

| encoder      | val NLL | val acc |
| ------------ | ------- | ------- |
| `sum`        | 3.636 ± 0.060 | 0.008 ± 0.008 |
| `ci`         | 0.825 ± 0.046 | 0.678 ± 0.023 |
| `fdm`        | 0.720 ± 0.034 | 0.725 ± 0.013 |
| **`concat`** | **0.582 ± 0.015** | **0.787 ± 0.008** |
| **`sum-perch`** | **0.587 ± 0.014** | **0.793 ± 0.008** |
| **`sum-ortho`** | **0.588 ± 0.014** | **0.794 ± 0.009** |

`sum` fails catastrophically on real data; all three per-channel-`W_k`
encoders tie at the top.

**Three mechanisms tie for first**: hard **block partitioning** (`concat`),
plain **per-channel learned projection** (`sum-perch` — replace shared `W`
with per-channel `W_k` and sum), and `sum-perch` plus a **soft orthogonality
penalty** (`sum-ortho`, λ · Σ(W_i·W_j)² on the projections). They are
indistinguishable to four decimals of NLL.

A direct gram-matrix measurement (`fft_encode/gram_analysis.py`) shows
that `sum-perch`'s learned `W_k` already converge to mean pairwise cosine
0.054 (near-orthogonal) without any regulariser; the penalty tightens
this to 0.021 but doesn't change downstream metrics. **Per-channel
projections are the mechanism; the orthogonality penalty is decorative.**

`fdm`'s sinusoidal carrier is a faster-convergence inductive bias but a
lower ceiling; `sum`'s shared-W recipe never closes the gap (15× more
parameters moves NLL by <0.02).

### Scaling with model width

At C=4, val NLL / val acc (mean over 5 seeds):

| d_model | total params | sum           | fdm           | concat            |
| ------- | ------------ | ------------- | ------------- | ----------------- |
| 64      | 152K         | 3.253 / 0.092 | 2.854 / 0.145 | **2.429 / 0.201** |
| 128     | 600K         | 3.251 / 0.094 | 2.898 / 0.146 | **2.291 / 0.223** |
| 256     | 2.38M        | 3.255 / 0.091 | 2.930 / 0.139 | **2.322 / 0.221** |

- **`sum` is structurally capped**: 15× more parameters barely move it; no
  amount of width fixes rank-deficient encoding.
- **`concat` scales strongly** and starts overfitting at d_model=256.
- **`fdm` plateaus at d_block=16**, losing ground as width grows because the
  fixed sinusoidal template has fewer degrees of freedom than a learned
  per-block projection.

### Is the FDM gap a frequency-choice issue?

Two controls at d_model=128:

- **Learnable `ω_k`**: the `fdm-learn` variant plumbs as a Parameter with
  gradients but the ωs drift <1% under AdamW in 100 epochs. Metrics match
  fixed-ω to <10⁻³.
- **Grid search over 11 fixed (ω_min, ω_max) bands**: best NLL is
  **2.809** for ω ∈ [0.03, 0.5] (matching signal bandwidth), beating the
  default 2.900 by ~0.09 nats but still **0.52 nats behind `concat`
  (2.291)**. The gap is structural, not a hyperparameter miss.

---

## Quick start

```bash
uv sync
# run everything in one command (~1 hour on a single GPU):
uv run python -m fft_encode.run_all --out results_full --epochs 100 --seeds 5 --seeds-grid 3
# outputs: results_full/{main.json, dmodel.json, carrier.json, probe.json, summary.txt}
# figures:
uv run python -m fft_encode.plot
# if a late stage fails, rebuild the summary from existing JSONs:
uv run python -m fft_encode.rebuild_summary --dir results_full
```

Individual stage scripts are also available for debugging:

```bash
uv run python -m fft_encode.train --epochs 20           # single-encoder comparison
uv run python -m fft_encode.experiments --epochs 100    # main sweep only
uv run python -m fft_encode.scale_dmodel --epochs 100   # d_model sweep
uv run python -m fft_encode.carrier_grid --epochs 100   # carrier band grid
uv run python -m fft_encode.probe --epochs 100          # linear probes
```

Requires Python 3.11+ and a CUDA GPU is recommended (full sweep runs in
~90 s on a modern single-GPU workstation).

---

## Encoding formulae

For C channels at position t with values v_k(t), d_block = d_model / C:

**fdm** (this work, parameter-free):

```
e_k(t)[2i]   = v_k(t) · sin(ω_k · t · base^(-2i/d_block))
e_k(t)[2i+1] = v_k(t) · cos(ω_k · t · base^(-2i/d_block))
h(t)         = [e_0(t) ‖ e_1(t) ‖ … ‖ e_{C-1}(t)]
```

Carriers `{ω_k}` are log-spaced across `[0.5, 8.0]`; `base = 10000` as in
RoPE. Both `ω_k` and the per-dimension frequency ladder are fixed buffers.

**concat** (block partitioning only, ablation):

```
e_k(t) = W_k · v_k(t) + b_k
h(t)   = [e_0(t) ‖ … ‖ e_{C-1}(t)] + p(t)
```

with standard sinusoidal positional encoding `p(t)`.

**sum** (naive baseline):

```
h(t) = Σ_k (W · v_k(t) + e_k) + p(t)
```

with a shared scalar projection `W` and learned channel embeddings `e_k`.

---

## Benchmark task

Each series has C channels, each an independent sum-of-sinusoids plus
AR(1) noise, standardized. The outcome depends only on the first four
channels:

```
y_t = tanh(s_0(t-3) · s_1(t))
      + 0.6 · sin(1.3 · s_2(t-7))
      + 0.4 · 1[s_3(t) > 0] · s_0(t)
```

Any additional channels (k ≥ 4) are independent distractors. The
continuous outcome is binned into K=32 quantile bins; the model emits a
categorical distribution at every step and is trained with next-step
cross-entropy.

Signal frequencies lie in ≈ [0.03, 0.50] rad/sample; FDM carriers lie in
[0.5, 8.0] rad/sample, so the carrier band does not overlap the signal
band — the FDM gain is not a trivial resonance with the data.

---

## Repository layout

```
fft_encode/
  data.py             synthetic multi-signal dataset
  encodings.py        Sum, SumOrtho, Concat, FDM channel encoders
  baselines.py        channel-independent and channel-as-token models
  model.py            causal transformer + categorical head
  runner.py           shared training+tracing utility
  train.py            single-run trainer (legacy)
  experiments.py      main sweep (legacy)
  probe.py            linear-probe channel recovery (legacy)
  scale_dmodel.py     d_model scaling (legacy)
  carrier_grid.py     carrier band grid search (legacy)
  run_all.py          one-command reproducer (all stages with cosine LR)
  rebuild_summary.py  regenerate summary.txt from saved JSONs
  plot.py             paper figures
paper/
  main.tex         manuscript
  tables/*.tex     generated tables
  figures/*.pdf    generated figures
  main.pdf         built manuscript
results_full/      outputs from run_all.py (main, dmodel, carrier, probe, summary)
```

---

## Paper

Build with any TeX Live installation (needs `booktabs`, `multirow`,
`natbib`, `authblk`, `microtype`):

```bash
cd paper
pdflatex main.tex && pdflatex main.tex
```

The PDF is included in the repo.

---

## License

MIT.
