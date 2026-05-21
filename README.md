# FDM Channel Encoding

Ossi Lehtinen, Ocon Oy — <ossi@ocon.fi>

A frequency-division-multiplexing–inspired input encoding for multi-signal
transformers. Each channel gets a non-overlapping block of embedding
dimensions, and its scalar value amplitude-modulates a channel-specific,
RoPE-style sinusoidal carrier. Blocks are concatenated rather than summed,
so channel identity is preserved by construction.

This repo contains:
- A PyTorch implementation of three input encoders (`sum`, `concat`, `fdm`)
  sharing the same transformer backbone.
- A synthetic multi-signal benchmark where the outcome is a lagged,
  non-linear function of several channels, binned for a categorical
  generative head.
- A sweep runner, a linear-probe channel-recovery experiment, a test-time
  channel-masking experiment, and a plotting script.
- A LaTeX manuscript (`paper/`) with all tables and figures regenerated
  from experiment outputs.

---

## Headline results

At C=4 channels, mean ± std over 3 seeds, same transformer backbone
(≈152K params) for all three encoders:

Val NLL = validation negative log-likelihood (cross-entropy of the
categorical generative head); val acc = top-1 bin accuracy.

| encoder                        | encoder params | val NLL ↓        | val acc ↑        |
| ------------------------------ | -------------- | ---------------- | ---------------- |
| `sum` (naive baseline)         | 320            | 3.280 ± 0.009    | 0.088 ± 0.002    |
| `concat` (block, no carrier)   | 128            | 3.125 ± 0.018    | 0.116 ± 0.007    |
| **`fdm` (block + carrier)**    | **0**          | **3.061 ± 0.014**| **0.123 ± 0.001**|

Random baseline: NLL = ln 32 ≈ 3.466, acc = 1/32 ≈ 0.031.

Linear-probe channel recovery (R² from a frozen hidden state back to the
raw input, per channel):

| encoder  | input layer | after 3 layers |
| -------- | ----------- | -------------- |
| `fdm`    | 1.00        | 0.96           |
| `concat` | 0.997       | 0.93           |
| `sum`    | 0.24        | 0.24           |

Block partitioning alone (`concat`) accounts for ~70–80% of the gain over
`sum`; the carrier adds the remainder.

---

## Quick start

```bash
uv sync
# single-encoder comparison at C=4
uv run python -m fft_encode.train --epochs 10
# full paper sweep: 3 encoders × {4,8,16} channels × 3 seeds, plus masking
uv run python -m fft_encode.experiments --seeds 3 --epochs 12 --out results.json
# linear-probe channel recovery at C=4
uv run python -m fft_encode.probe --epochs 12 --out probe_results.json
# regenerate scaling figure
uv run python -m fft_encode.plot
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
  data.py         synthetic multi-signal dataset
  encodings.py    SumEncoding, ConcatEncoding, FDMChannelEncoding
  model.py        causal transformer + categorical head
  train.py        single-run trainer
  experiments.py  main sweep (table 1 + scaling + channel mask)
  probe.py        linear-probe channel recovery
  plot.py         scaling figure for the paper
paper/
  main.tex        manuscript
  tables/*.tex    generated tables
  figures/*.pdf   generated figures
  main.pdf        built manuscript
results.json        full experiment output
probe_results.json  probe output
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
