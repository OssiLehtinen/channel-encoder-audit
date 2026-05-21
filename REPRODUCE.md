# Reproducing the paper

Every number in the paper comes from `fft_encode.reproduce`. From a clean
clone of the repo:

```bash
uv sync
uv run python -m fft_encode.reproduce --out results/
```

This runs all stages at the paper defaults (300 epochs, 5 seeds, cosine LR
decay), writing one JSON per stage plus `results/summary.txt`. The
output schema is identical to `results_paper/`, which holds the exact
files produced for the submitted manuscript.

## What runs and what comes out

| Stage         | Output file                | What it produces |
|---------------|----------------------------|------------------|
| `main`        | `main.json`                | Synthetic sweep: $C \in \{4,8,16\}$ × 8 encoders × 5 seeds. Feeds Table 1 and Figure 1. |
| `dmodel`      | `dmodel.json`              | $d_{\text{model}} \in \{64,128,256\}$, `sum` + `concat`, 5 seeds. Feeds the $d_{\text{model}}$ scaling table. |
| `geometry`    | `geometry.json`            | `linear` at all $C$, `sum-ortho` at $C=4$. Reports $W_k$ norms, off-diagonal Gram entries, variance fractions. Feeds the gram and encoding-space tables. |
| `probe`       | `probe.json`               | Closed-form ridge probe recovering raw channels from hidden states (layer 0 and 3). |
| `mask`        | `mask.json`                | Per-channel zeroing at test time. |
| `etth1`       | `etth1.json`               | ETTh1 real-data validation, all encoders × 5 seeds at $d_{\text{model}}=56$. ETTh1.csv is fetched on first use and cached under `~/.cache/fft_encode/`. |
| `convergence` | `convergence.json`         | Analysis-only — derives epochs-to-target from the `main` traces. |

`summary.txt` aggregates everything in human-readable form.

## Wall time

Roughly 10–12 GPU-hours total on a single modern GPU (RTX 4090 / A100 class).
The expensive pieces are:

- `etth1` with `cat`: ~25 min/seed × 5 seeds (channel-as-token attention is
  $\mathcal{O}((CT)^2)$ at $C=7, T=160$).
- `dmodel` at $d_{\text{model}}=256$: ~2× the cost of $d=128$.
- `main` with `cat` at $C=8$: ~3.6 s/epoch.

Everything else is in the noise.

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
cd paper && pdflatex main.tex && pdflatex main.tex
```

Build dependencies: any TeX Live with `booktabs`, `multirow`, `natbib`,
`authblk`, `microtype`.

## Determinism notes

- `torch.manual_seed(seed)` is set per run. Seeded `random_split` makes the
  train/val partition reproducible. We do not enable
  `torch.use_deterministic_algorithms(True)`; CUDA matmul nondeterminism
  contributes to seed-level variation in the runs.
- Re-running `fft_encode.reproduce` from scratch will reproduce the
  paper's numbers within seed std but not bit-exactly.
- The legacy encoder name `sum-perch` is aliased to `linear` in
  `build_model`, so JSON files in `results_paper/` (which use `sum-perch`)
  can be loaded and analysed alongside fresh runs from the new script.

## Pending

- `cat` on ETTh1 at 300 epochs is the one stage not yet committed to
  `results_paper/etth1.json` (the run was aborted mid-stream). Running
  `fft_encode.reproduce --stages etth1` fills it in; expect roughly two
  hours of wall time for those 5 seeds alone.
