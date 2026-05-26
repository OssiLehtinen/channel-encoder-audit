# Input Encoders for Multi-Channel Signal Transformers

Ossi Lehtinen, Ocon Oy — <ossi@ocon.fi>

> **Recommendation.** Default to `nn.Linear(C, d_model)` — it's the
> simplest thing that works and we never decisively beat it, because
> the per-channel projections spontaneously orthogonalise when the
> task requires it. For a small but consistent ${\sim}2\%$ NLL gain,
> use `linear-ppe` (the same per-channel projection plus a learned
> linear projection of the sinusoidal positional encoding).

An empirical audit of how transformers should embed $C$ simultaneous scalar
channels at the input layer. Eight encoders compared on a controlled
synthetic benchmark and on the public ETTh1 dataset, with paired-difference
analysis at 20 seeds for headline configurations.

The headline result is that **the standard per-channel linear projection
`nn.Linear(C, d_model)`** is hard to dislodge. Block partitioning
(`concat`) and orthogonality regularisation (`linear-ortho`) tie it
within seed noise; a nonlinear MLP stem matches it at low $C$ and edges
it narrowly at $C{=}16$; and **`linear-ppe`** — projecting the
sinusoidal positional encoding through a learned linear layer — gives
a small but consistent ${\sim}2\%$ NLL edge at every $C$ tested. The
shared-scalar baseline `sum` (information-theoretic collapse) and the
channel-independent baseline `ci` (overfits universally on the synthetic
benchmark, underperforms on both) lose decisively; the channel-as-token
baseline `cat` sits behind the per-channel-$W_k$ tier on the synthetic
benchmark but ties it on ETTh1. A direct geometric probe identifies
`linear-ppe`'s mechanism as positional-channel orthogonalisation (not
subspace compression).

---

## Headline results (20 seeds, 300 epochs, best val NLL)

**Synthetic, $C{=}4$ channels:**

| Encoder          | Val NLL ↓             | Val acc ↑             |
| ---------------- | --------------------- | --------------------- |
| `sum`            | $3.257 \pm 0.011$     | $0.091 \pm 0.003$     |
| `ci`             | $3.053 \pm 0.013$     | $0.136 \pm 0.004$     |
| `cat`            | $2.348 \pm 0.060$     | $0.212 \pm 0.011$     |
| `mlp`            | $2.177 \pm 0.022$     | $0.242 \pm 0.007$     |
| `concat`         | $2.170 \pm 0.025$     | $0.243 \pm 0.006$     |
| `linear`         | $2.155 \pm 0.019$     | $0.247 \pm 0.006$     |
| `linear-ortho`   | $2.155 \pm 0.020$     | $0.247 \pm 0.006$     |
| **`linear-ppe`** | **$2.114 \pm 0.029$** | **$0.256 \pm 0.009$** |

Random baseline: NLL $= \ln 32 \approx 3.466$, acc $= 1/32 \approx 0.031$.
At $C{=}16$, `mlp` takes the lowest mean NLL but ties `linear-ppe`
under paired analysis ($p{=}0.053$); see the paper for the full scaling
table.

**ETTh1 (7 variates, next-step bin of oil temperature):**

| Encoder          | Val NLL ↓             | Val acc ↑             |
| ---------------- | --------------------- | --------------------- |
| `sum`            | $3.668 \pm 0.063$     | $0.016 \pm 0.020$     |
| `ci`             | $0.865 \pm 0.038$     | $0.664 \pm 0.018$     |
| `mlp`            | $0.585 \pm 0.020$     | $0.788 \pm 0.011$     |
| `linear-ppe`     | $0.573 \pm 0.014$     | $0.776 \pm 0.009$     |
| `concat`         | $0.571 \pm 0.019$     | $0.783 \pm 0.013$     |
| `linear`         | $0.561 \pm 0.017$     | $0.786 \pm 0.008$     |
| `linear-ortho`   | $0.561 \pm 0.017$     | $0.784 \pm 0.009$     |
| **`cat`**        | **$0.551 \pm 0.019$** | $0.785 \pm 0.010$     |

`sum` and `ci` are the decisive losers. The per-channel-$W_k$ family
clusters within seed noise. `cat` posts the lowest mean NLL — but paired
analysis at 20 seeds puts it *statistically tied* with `linear` and
`linear-ortho` ($p > 0.10$) while decisively above `mlp` and
`linear-ppe`. The honest reading is that `cat`'s synthetic-benchmark
disadvantage closes on ETTh1, where all seven variates are plausibly
informative measurements of the same underlying electrical system, but
its claimed lead does not survive a paired test against the closest
competitors.

---

## What each encoder is

For $C$ channels with values $v_k(t)$ at position $t$, embedded into
$\mathbb{R}^{d_{\text{model}}}$, with $\mathbf{p}(t)$ a fixed sinusoidal
positional encoding:

| Name           | Definition |
|----------------|-----------|
| `sum`          | shared scalar projection $W$, per-channel bias: $h(t) = \sum_k (W v_k(t) + e_k) + \mathbf{p}(t)$ |
| `linear`       | per-channel projection $W_k$, summed: $h(t) = \sum_k (W_k v_k(t) + b_k) + \mathbf{p}(t)$ — i.e. `nn.Linear(C, d_model)` |
| `linear-ortho` | `linear` plus auxiliary loss $\lambda \sum_{i < j}(W_i \cdot W_j)^2$ ($\lambda=10^{-2}$) |
| `mlp`          | two-layer MLP on the channel vector with GELU nonlinearity |
| `linear-ppe`   | `linear` channel side plus a learned linear projection of $\mathbf{p}(t)$ |
| `concat`       | per-channel projection into $d_{\text{model}}/C$ dims, concatenated (block partitioning) |
| `ci`           | channel-independent (PatchTST-spirit): shared backbone runs per channel |
| `cat`          | channel-as-token (iTransformer-spirit): each $(t, k)$ pair is a token |

Six of the eight are encoder swaps that share the I/O signature
$\mathbb{R}^{B\times T\times C} \to \mathbb{R}^{B\times T\times d_{\text{model}}}$
and feed into the same causal transformer backbone. `ci` and `cat` are
full-architecture alternatives that reshape the token sequence the
backbone consumes.

---

## Synthetic benchmark

Each series has $C$ channels, each an independent sum-of-three-sinusoids
with frequencies in $[0.005, 0.08]$ cycles/sample plus AR(1) noise,
standardised. The outcome depends on the first four channels:

$$
y_t = \tanh(s_0(t{-}3)\, s_1(t)) + 0.6\sin(1.3\, s_2(t{-}7))
      + 0.4\,\mathbb{1}[s_3(t) > 0]\, s_0(t),
$$

binned into $K=32$ quantile bins. Channels $k \ge 4$ are independent
distractors. The benchmark is deliberately designed so that an encoder that
mixes channels at the input layer cannot recover the interaction structure.

ETTh1 (`fft_encode/real_data.py`) is fetched from the ETDataset GitHub
mirror on first use and cached under `$FFT_ENCODE_CACHE` (default
`~/.cache/fft_encode/`).

---

## Reproducibility

The single entry point is `fft_encode.reproduce`. From a clean clone:

```bash
uv sync
uv run python -m fft_encode.reproduce --out results/
```

This runs every experiment reported in the paper at 300 epochs, 20 seeds
for headline configurations (5 seeds for a few diagnostics, labelled as
such), cosine LR decay, writing one JSON per stage plus a human-readable
`summary.txt`:

```
results/main.json            # synthetic, C in {4,8,16} x 8 encoders
results/dmodel.json          # d_model in {64,128,256}, 6 top-tier encoders
results/geometry.json        # W_k norms + off-diagonal Gram + variance fractions
results/probe.json           # linear-probe channel recovery (R²)
results/mask.json            # test-time channel masking
results/etth1.json           # ETTh1 real-data validation
results/bias.json            # channel-bias ablation (linear vs. linear-nobias)
results/geom_largen.json     # large-N geometry (distractor noise floor)
results/main_largen.json     # C=16 top-tier at 10× training data
results/main_mse.json        # MSE/regression head loss-family check
results/mlp_geometry.json    # Gram on MLP's W^(1) columns
results/pospro_geometry.json # positional projection geometry
results/convergence.json     # epochs-to-target, derived from main traces
results/summary.txt          # aggregate of everything above
```

Wall-clock on a single modern GPU is roughly 10–12 hours.

The optional `extra_seeds` stage is an open-ended round-robin that loops
until interrupted, writing per-(stage, seed) JSON files under
`results/extra_seeds/`. It is *not* in the default `--stages` set; opt in
explicitly:

```bash
uv run python -m fft_encode.reproduce --out results/ --stages extra_seeds
```

`fft_encode.analyze` aggregates the canonical and extra-seed JSONs into
paired-difference tests and bootstrap CIs:

```bash
uv run python -m fft_encode.analyze --out results/
```

writes `results/analysis_<N>seeds.{json,txt}` driving the paper's
$p$-values and intervals.

### Useful flags

```bash
# smoke test (30 epochs, 2 seeds)
uv run python -m fft_encode.reproduce --out smoke/ --epochs 30 --seeds 2

# run only one stage
uv run python -m fft_encode.reproduce --out results/ --stages etth1

# subset of encoders / channels
uv run python -m fft_encode.reproduce --out results/ \
    --encoders sum linear concat --Cs 4 8
```

### Reference results

The exact JSON outputs that produced the numbers in the paper are committed
under `results_paper/`. The reproducer writes to `results/` by default so
reference numbers stay untouched.

### Figures

```bash
uv run python -m fft_encode.plot \
    --results results_paper/main.json \
    --dmodel-results results_paper/dmodel.json
```

writes `paper/figures/scaling.pdf` and `paper/figures/dmodel_scaling.pdf`.

### Paper

```bash
cd paper
pdflatex main.tex && pdflatex main.tex
```

Build dependencies: any TeX Live with `booktabs`, `multirow`, `natbib`,
`authblk`, `microtype`.

---

## Repository layout

```
fft_encode/
  reproduce.py     ← one-command reproducer (start here)
  analyze.py       paired tests + bootstrap CIs across merged seed pool
  data.py          synthetic multi-signal dataset
  real_data.py     ETTh1 adapter
  encodings.py     linear / linear-ortho / linear-ppe / concat / mlp / sum
  baselines.py     ci (channel-independent), cat (channel-as-token)
  model.py         causal transformer + categorical head + build_model()
  runner.py        train-and-trace utility used by reproduce.py
  experiments.py   RunCfg + LR scheduler + evaluate (training primitives)
  probe.py         closed-form ridge probe (collect_hidden, fit_probe)
  plot.py          render paper figures from results JSONs
paper/             LaTeX manuscript + tables + figures
results_paper/     reference numbers committed alongside the paper
TODO.md            deferred refactor items from the code review
RESEARCH_LOG.md    chronological narrative of how the project evolved
```

The class implementing `linear`, `linear-ortho`, `linear-ppe`, and
`linear-nobias` is `SumOrthoEncoding` — a name kept from before the
rename for checkpoint compatibility; new code should refer to the
paper-facing encoder names.

---

## Citation

If you use this work, please cite the manuscript at `paper/main.pdf`.

## License

MIT.
