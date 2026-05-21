# TODO

Deferred items from the code review (May 2026). The priority-1 cluster
(scipy dependency, `extra_seeds` infinite-loop default,
`ENCODERS_DMODEL` partial sweep, stale labels and docstrings, asserts,
dead variable) was fixed in commit `b0ced6b`. The remaining items are
medium-effort refactors that are worth doing if the code keeps
evolving but were not blocking the paper.

Item numbers below match the code review for traceability.

## Refactors

- [ ] **#7 — Consolidate the two ETTh1 training paths.**
      `fft_encode.reproduce._train_etth1` re-implements
      `fft_encode.runner.train_and_trace` for the ETTh1 dataset.
      Subtle divergences have already crept in (no `convergence_flag`,
      different `Trace` shape, hard-coded `log_every`). Right shape:
      parameterise `train_and_trace` by a dataset factory so the only
      ETTh1-specific code is `lambda: (ETTh1Dataset("train"),
      ETTh1Dataset("val"))`.

- [ ] **#8 — Factor `stage_extra_seeds`.**
      `fft_encode/reproduce.py:622-859` duplicates per-seed inner loops
      for seven stages (~250 lines). Each canonical stage function
      could expose a `one_seed(s, args, device) -> dict` entry point;
      `stage_extra_seeds` iterates over `(s, stage_fn)` pairs and
      writes per-seed files. Removes the "fix it in two places" hazard.

- [ ] **#9 — Merge `train_and_trace_mse` into `train_and_trace`.**
      `runner.py:143-249` is a ~200-line near-copy of
      `train_and_trace` that swaps `return_continuous=True`,
      `n_bins=1`, `mse_loss`/`evaluate_mse`, and the metric field names.
      Should be a `target_type="categorical"|"regression"` parameter
      on the single function.

- [ ] **#10 — Split or rename `SumOrthoEncoding`.**
      `encodings.py:35-136` implements five paper encoders (`linear`,
      `linear-ortho`, `linear-ppe`, `linear-lpe`, `linear-nobias`) via
      flags. The class name is a leftover from the original
      "sum-ortho" variant. Either:
      - rename to `LinearEncoding` and keep `SumOrthoEncoding =
        LinearEncoding` for old checkpoints, or
      - split into a base class + four thin subclasses (one per paper
        encoder), so `build_model(kind=...)` constructs an
        obviously-correct module per name.

- [ ] **#11 — Deduplicate `_canon` / `_ENCODER_ALIAS`.**
      Triplicated in `reproduce.py:961-967`, `analyze.py:82-96`, and
      `plot.py:29-33`. Pull into `fft_encode/_aliases.py`:
      ```python
      ENCODER_ALIAS = {"sum-perch": "linear",
                       "sum-ortho": "linear-ortho"}
      def canon(enc: str) -> str:
          return ENCODER_ALIAS.get(enc, enc)
      ```
      and import from the three consumers.

- [ ] **#12 — Split `fft_encode/reproduce.py` (1397 lines).**
      Natural shape:
      - `reproduce/__main__.py` — CLI + stage dispatcher
      - `reproduce/stages.py` — the 13 `stage_X` functions
      - `reproduce/extra_seeds.py` — round-robin loop
      - `reproduce/summary.py` — `write_summary` + helpers

## Maintainability / minor

- [ ] **#14 — Hash-verify the ETTh1 download.**
      `fft_encode/real_data.py:34-40` fetches the CSV from a GitHub
      mirror's `main` branch with no integrity check. Add an
      SHA-256 verification on first download and fail loudly if it
      doesn't match the expected hash. Locks reproducibility against
      upstream drift.

- [ ] **#15 — Make ETTh1 dimensions configurable.**
      `reproduce.py:286-291` hard-codes `T=160, K=32, d_model=56,
      n_heads=7`. None of these are tied to CLI or `RunCfg`, so
      sweeping `d_model` or `T` on ETTh1 requires editing code.

- [ ] **#16 — Document or name-constant the convergence-flag thresholds.**
      `runner.py:108-122` uses magic numbers `0.005` (flat-tail
      threshold for "converged") and `0.01` ("overfitting" trigger).
      Pull into named constants with a short docstring explaining the
      choice; the values are diagnostic-only but currently undocumented.

- [ ] **#17 — Move MLP gram analysis onto the encoder.**
      `MLPEncoding` has no `gram_stats()` method; gram analysis on
      $W^{(1)}$ is done inline in `reproduce.py:583` via
      `model.encoder.mlp[0].weight`. Adding a `gram_stats()` method
      mirroring `SumOrthoEncoding.gram_stats` would make the
      diagnostic uniformly available across encoders.

- [ ] **#18 — Trim dead diagnostics from `stage_pospro_geometry`.**
      Principal angles between $\mathrm{span}(W)$ and $\mathrm{span}(P)$
      are computed and saved per run (`reproduce.py:524-526`), but the
      paper concludes they are degenerate at this dimensionality and
      not the right tool. The numbers go into the JSON anyway. Either
      drop, or wrap behind a `--with-principal-angles` flag.

- [ ] **#20 — Re-export common names from `fft_encode/__init__.py`.**
      Currently empty. Re-exporting `RunCfg`, `train_and_trace`,
      `build_model`, and the encoder classes would let callers write
      `from fft_encode import RunCfg` instead of
      `from fft_encode.experiments import RunCfg`.
