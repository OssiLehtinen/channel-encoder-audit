# Reproducing the paper

Every number in the paper comes from `fft_encode.reproduce` (the
training/diagnostic pipeline) and `fft_encode.analyze` (the
paired-test/bootstrap aggregator). From a clean clone of the repo:

```bash
uv sync
uv run python -m fft_encode.reproduce --out results/
uv run python -m fft_encode.analyze   --out results/
```

The first command runs all 13 finite stages at the paper defaults
(300 epochs, 20 seeds for headline configurations, cosine LR decay),
writing one JSON per stage plus `results/summary.txt`. The second
command merges the canonical and any extra-seed JSONs into paired-
difference tests and bootstrap CIs, writing `analysis_<N>seeds.json`
and `analysis_<N>seeds.txt`. The output schema is identical to
`results_paper/`, which holds the exact files produced for the
submitted manuscript.

## What runs and what comes out

| Stage             | Output file                | What it produces |
|-------------------|----------------------------|------------------|
| `main`            | `main.json`                | Synthetic sweep: $C \in \{4,8,16\}$ × 8 encoders × 20 seeds. Feeds Table 1 and Figure 1. |
| `dmodel`          | `dmodel.json`              | $d_{\text{model}} \in \{64,128,256\}$, 6 top-tier encoders ({sum, concat, linear, linear-ortho, mlp, linear-ppe}), 20 seeds. Feeds the $d_{\text{model}}$ scaling table. |
| `geometry`        | `geometry.json`            | `linear` at all $C$, `linear-ortho` at $C{=}4$, 20 seeds. Reports $W_k$ norms, off-diagonal Gram entries, variance fractions. Feeds the gram and encoding-space tables. |
| `probe`           | `probe.json`               | Closed-form ridge probe recovering raw channels from hidden states (layer 0 and 3). 5 seeds. |
| `mask`            | `mask.json`                | Per-channel zeroing at test time. 5 seeds. |
| `etth1`           | `etth1.json`               | ETTh1 real-data validation, all encoders × 20 seeds at $d_{\text{model}}{=}56$. ETTh1.csv is fetched on first use and cached under `~/.cache/fft_encode/`. |
| `bias`            | `bias.json`                | Channel-bias ablation: `linear` vs.\ `linear-nobias` at $C{=}4$, 5 seeds. |
| `geom_largen`     | `geom_largen.json`         | Large-$N$ geometry probe: `linear` at $C{=}8$ with $10\times$ training data (distractor-norm noise-floor check). |
| `main_largen`     | `main_largen.json`         | Top-tier encoders at $C{=}16$ with $10\times$ training data, 20 seeds. Tests whether the C=16 ranking persists when not data-limited. |
| `main_mse`        | `main_mse.json`            | Loss-family check: same encoders at $C \in \{4, 16\}$ with a scalar regression head + MSE loss. 5 seeds. |
| `mlp_geometry`    | `mlp_geometry.json`        | Gram-matrix analysis on `mlp`'s first-layer columns, $C \in \{4, 8, 16\}$, 5 seeds. |
| `pospro_geometry` | `pospro_geometry.json`     | Geometric mechanism probe for `linear-ppe`: effective rank of positional basis, fraction of $P$'s energy inside $\mathrm{span}(W)$, paired-seed against `linear`, 20 seeds. |
| `convergence`     | `convergence.json`         | Analysis-only — derives epochs-to-target from the `main` traces. |

`summary.txt` aggregates everything in human-readable form.

### Optional opt-in stage

| Stage         | Output                                                     | What it does |
|---------------|------------------------------------------------------------|--------------|
| `extra_seeds` | `extra_seeds/{stage}_s{NNN}.json` (per-stage, per-seed)    | Open-ended round-robin: for each new seed $s$ starting from `--extra-seeds-start` (default 5), runs one full cycle of the paired-test-sensitive stages and writes per-(stage, seed) files. **Loops until interrupted.** Not in the default `--stages` set; opt in explicitly. |

```bash
uv run python -m fft_encode.reproduce --out results/ --stages extra_seeds
```

The default 20-seed paper numbers used 15 cycles of `extra_seeds`
(seeds 5–19) on top of the canonical seeds 0–4.

## Wall time

Roughly **12 GPU-hours** for the full default `--stages` run on a single
modern GPU (RTX 4090 / A100 class). The expensive pieces are:

- `etth1` with `cat`: ~25 min/seed × 20 seeds (channel-as-token
  attention is $\mathcal{O}((CT)^2)$ at $C{=}7, T{=}160$).
- `dmodel` at $d_{\text{model}}{=}256$: ~2× the cost of $d{=}128$.
- `main` with `cat` at $C{=}8$: ~3.6 s/epoch × 300 epochs × 20 seeds.

Everything else is in the noise. The `extra_seeds` round-robin adds
roughly **1 GPU-day per cycle** at the same compute scale; for the
paper's 15 cycles, that was an overnight + a day.

## Running individual stages

```bash
# only ETTh1
uv run python -m fft_encode.reproduce --out results/ --stages etth1

# only the synthetic main sweep, fewer channels
uv run python -m fft_encode.reproduce --out results/ \
    --stages main --Cs 4 8

# quick smoke test (~5 minutes, sanity check only)
uv run python -m fft_encode.reproduce --out smoke/ \
    --epochs 30 --seeds 2 --Cs 4 \
    --encoders sum linear concat --stages main convergence
```

## Regenerating figures

After `main` and `dmodel` have written their JSONs (or pointing at
`results_paper/`):

```bash
uv run python -m fft_encode.plot \
    --results results_paper/main.json \
    --dmodel-results results_paper/dmodel.json
```

writes `paper/figures/scaling.pdf` and `paper/figures/dmodel_scaling.pdf`.
`plot.py` accepts both the new list-shaped output and the legacy
`{"runs": [...]}` wrapper.

## Building the paper

```bash
cd paper && pdflatex lehtinen2026_channel_encoders.tex && pdflatex lehtinen2026_channel_encoders.tex
```

Build dependencies: any TeX Live with `booktabs`, `multirow`, `natbib`,
`authblk`, `microtype`, `tikz`.

## Determinism notes

- `torch.manual_seed(seed)` is set per run. Seeded `random_split`
  makes the train/val partition reproducible. We do not enable
  `torch.use_deterministic_algorithms(True)`; CUDA matmul
  nondeterminism contributes to seed-level variation in the runs.
- Re-running `fft_encode.reproduce` from scratch will reproduce the
  paper's numbers within seed std but not bit-exactly.
- Two legacy encoder names are aliased in `build_model` for
  backward compatibility with older JSON outputs:
  `sum-perch` → `linear`, and `sum-ortho` → `linear-ortho`. JSON
  files in `results_paper/` (some of which use the legacy names in
  `cfg.encoder`) can be loaded and analysed alongside fresh runs
  from the renamed code. Both `fft_encode.analyze` and the summary
  writer canonicalise encoder names on read so the merged tables
  stay consistent.
