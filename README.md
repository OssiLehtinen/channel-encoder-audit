# Input Encoders for Multi-Channel Signal Transformers

Ossi Lehtinen, Ocon Oy — <ossi@ocon.fi>

An empirical audit of how transformers should embed $C$ simultaneous scalar
channels at the input layer. Eight encoders, one shared transformer backbone,
on a controlled synthetic benchmark and on the public ETTh1 dataset.

The headline result is that **the standard per-channel linear projection
`nn.Linear(C, d_model)`** matches every alternative we test (block
partitioning, orthogonality regularisers, nonlinear MLP stems,
channel-independent and channel-as-token architectures). The one variant
showing a small further improvement is **`linear-ppe`** — projecting the
sinusoidal positional encoding through a learned linear layer — but the gap
is at the edge of what five seeds resolve, and the underlying mechanism
(positional/channel orthogonalisation vs.\ positional subspace compression)
is not pinned down here.

---

## Headline results (5 seeds, 300 epochs, best val NLL)

**Synthetic, $C=4$ channels:**

| Encoder          | Val NLL ↓             | Val acc ↑             |
| ---------------- | --------------------- | --------------------- |
| `sum`            | $3.252 \pm 0.009$     | $0.092 \pm 0.003$     |
| `ci`             | $3.054 \pm 0.007$     | $0.139 \pm 0.003$     |
| `cat`            | $2.360 \pm 0.073$     | $0.210 \pm 0.014$     |
| `concat`         | $2.183 \pm 0.030$     | $0.240 \pm 0.005$     |
| `mlp`            | $2.171 \pm 0.029$     | $0.244 \pm 0.008$     |
| `linear-ortho`      | $2.170 \pm 0.016$     | $0.245 \pm 0.006$     |
| `linear`         | $2.170 \pm 0.016$     | $0.243 \pm 0.005$     |
| **`linear-ppe`** | **$2.116 \pm 0.034$** | **$0.257 \pm 0.009$** |

Random baseline: NLL $= \ln 32 \approx 3.466$, acc $= 1/32 \approx 0.031$.

**ETTh1 (7 variates, next-step bin of oil temperature):**

| Encoder      | Val NLL ↓             | Val acc ↑         |
| ------------ | --------------------- | ----------------- |
| `sum`        | $3.633 \pm 0.049$     | $0.018 \pm 0.024$ |
| `ci`         | $0.853 \pm 0.022$     | $0.674 \pm 0.013$ |
| `mlp`        | $0.577 \pm 0.010$     | $0.790 \pm 0.007$ |
| `linear-ortho`  | $0.572 \pm 0.022$     | $0.787 \pm 0.008$ |
| `linear-ppe` | $0.570 \pm 0.015$     | $0.773 \pm 0.005$ |
| `linear`     | $0.566 \pm 0.019$     | $0.789 \pm 0.008$ |
| `concat`     | $0.560 \pm 0.010$     | $0.777 \pm 0.018$ |
| **`cat`**    | **$0.541 \pm 0.011$** | $0.787 \pm 0.008$ |

`sum` fails catastrophically; `ci` underperforms; the per-channel-$W_k$
family clusters within seed noise. `cat` posts the lowest mean NLL on
real data — but its confidence interval barely separates from
`concat`/`linear` and best-bin accuracies across the top tier are
indistinguishable, so the honest reading is that `cat`'s
synthetic-benchmark disadvantage closes on ETTh1, not that it
decisively wins. The most plausible reason: ETTh1's seven variates are
all informative measurements of one physical system, which rewards the
fine-grained cross-variate attention `cat` enables.

---

## What each encoder is

For $C$ channels with values $v_k(t)$ at position $t$, embedded into
$\mathbb{R}^{d_{\text{model}}}$, with $\mathbf{p}(t)$ a fixed sinusoidal
positional encoding:

| Name         | Definition |
|--------------|-----------|
| `sum`        | shared scalar projection $W$, per-channel bias: $h(t) = \sum_k (W v_k(t) + e_k) + \mathbf{p}(t)$ |
| `linear`     | per-channel projection $W_k$, summed: $h(t) = \sum_k (W_k v_k(t) + b_k) + \mathbf{p}(t)$ — i.e. `nn.Linear(C, d_model)` |
| `linear-ortho`  | `linear` plus auxiliary loss $\lambda \sum_{i \ne j}(W_i \cdot W_j)^2$ ($\lambda=10^{-2}$) |
| `mlp`        | two-layer MLP on the channel vector with GELU nonlinearity |
| `linear-ppe` | `linear` channel side plus a learned linear projection of $\mathbf{p}(t)$ |
| `concat`     | per-channel projection into $d_{\text{model}}/C$ dims, concatenated (block partitioning) |
| `ci`         | channel-independent (PatchTST-spirit): shared backbone runs per channel |
| `cat`        | channel-as-token (iTransformer-spirit): each $(t, k)$ pair is a token |

---

## Reproducibility

The single entry point is `fft_encode.reproduce`. From a clean clone:

```bash
uv sync
uv run python -m fft_encode.reproduce --out results/
```

This runs every experiment reported in the paper at 300 epochs, 5 seeds,
cosine LR decay, writing one JSON per stage plus a human-readable
`summary.txt`:

```
results/main.json        # synthetic, C in {4,8,16} x 8 encoders x 5 seeds
results/dmodel.json      # d_model in {64,128,256}, sum + concat
results/geometry.json    # W_k norms + off-diagonal Gram + variance fractions
results/probe.json       # linear-probe channel recovery (R²)
results/mask.json        # test-time channel masking
results/etth1.json       # ETTh1 real-data validation
results/convergence.json # epochs-to-target, derived from main traces
results/summary.txt      # aggregate of everything above
```

Wall-clock on a single modern GPU is roughly 10–12 hours.

### Useful flags

```bash
# smoke test (30 epochs, all stages)
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

## Repository layout

```
fft_encode/
  reproduce.py     ← one-command reproducer (start here)
  data.py          synthetic multi-signal dataset
  real_data.py     ETTh1 adapter
  encodings.py     Sum, SumOrtho, Concat, MLP encoders
  baselines.py     channel-independent, channel-as-token architectures
  model.py         causal transformer + categorical head + build_model()
  runner.py        train-and-trace utility used by reproduce.py
  experiments.py   RunCfg + LR scheduler + evaluate (training primitives)
  probe.py         closed-form ridge probe (collect_hidden, fit_probe)
  plot.py          render paper figures from results JSONs
paper/             LaTeX manuscript + tables + figures
results_paper/     reference numbers committed alongside the paper
```

---

## Citation

If you use this work, please cite the manuscript at `paper/main.pdf`.

## License

MIT.
