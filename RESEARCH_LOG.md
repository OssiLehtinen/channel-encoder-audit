# Research log

How this project evolved from an FDM-inspired channel encoding paper into
an empirical audit of subspace geometry in multi-channel signal
transformers. Reconstructed from the commit history.

---

## Phase 1 — The original premise: FDM channel encoding (April 13, 2026)

The project began with a specific architectural hunch: that the right way
to embed $C$ scalar channels into one $d_{\text{model}}$-dimensional vector
per time step was **frequency-division multiplexing**. Each channel gets
a non-overlapping block of embedding dimensions, and its scalar value
amplitude-modulates a channel-specific, RoPE-style sinusoidal carrier
inside the block. Blocks are concatenated, so channel identity is
preserved by construction.

The initial commit (`acd442c`) shipped a complete first-pass paper around
this premise: four encoders (`sum`, `concat`, `fdm`, `fdm-learn`), a
synthetic multi-signal benchmark with deliberate cross-channel
interactions, a sweep runner, linear-probe channel-recovery and
test-time channel-masking diagnostics, and a manuscript with all tables
regenerated from JSON outputs.

The original title was *FDM Channel Encoding*; the headline result was
that `fdm` (NLL 2.85) beat `sum` (3.25) and lost narrowly to `concat`
(2.43). The hypothesis was that the *carrier* — the sinusoidal modulation
inside each per-channel block — was buying the model something.

Same-day follow-ups (`7df0143`, `5bfccc3`, `7852867`, `04489d8`) added
authorship and expanded NLL/RoPE/FFT abbreviations on first use. `8188aa0`
added a `d_model` scaling study and a carrier-band grid search, both
aimed at characterising FDM's behaviour.

## Phase 2 — Sharper training, broader baselines (April 14–16)

The 12-epoch initial training turned out to be undertrained for `concat`;
a 100-epoch cosine sweep (`7d89cc8`) substantially rewrote the headline
table. Three new baselines came in alongside:

- `sum-ortho`: `sum` with a soft orthogonality regulariser on the learned
  per-channel projections.
- `ci` (channel-independent, PatchTST-spirit): shared-weight transformer
  per channel.
- `cat` (channel-as-token, iTransformer-spirit): each $(t, k)$ a token.

Headline at this point: `concat` and `sum-ortho` tied at NLL 2.42, both
cleanly above `fdm` (2.85), with `sum` structurally capped at 3.25
regardless of width. `ci` and `cat` underperformed at matched compute.

ETTh1 was added as a real-data check (`f5146be`) and an off-diagonal Gram
analysis confirmed the learned per-channel projections were already
near-orthogonal under `sum-ortho`.

At this point the paper was still framed around FDM as the headline
encoder, with `concat`/`sum-ortho` presented as alternatives that happen
to perform similarly.

## Phase 3 — The pivot: from FDM to subspace geometry (May 11)

This was the largest single shift. The premise that *FDM specifically* was
doing something interesting started to look implausible. A direct
question — *is the SUM-PERCH approach really novel?* — exposed that
`sum-perch` (per-channel projection $W_k$ then sum) is mathematically
just `nn.Linear(C, d_model)`. It's the most obvious encoder a PyTorch
user would write. There was no architectural novelty there.

`34d2085` ("paper: full rewrite as subspace-geometry audit") reframed
the manuscript around a geometric reading: distinct input streams need
approximately orthogonal subspaces in $d_{\text{model}}$ so the model
can recover stream-of-origin. The shared-projection `sum` baseline is the
one that violates this; everything else (`linear`, `sum-ortho`, `concat`,
new `mlp` stem, etc.) is some way of preserving channel identity at the
input layer.

`fdm` was dropped from the manuscript entirely as a dead end that didn't
serve the paper's actual story. The headline became: **the standard
per-channel linear projection matches every alternative; only the shared
scalar projection loses.**

New encoders added during this phase:

- `mlp`: two-layer MLP on the channel vector. Tests whether nonlinearity
  helps. (It doesn't, at small C with small width.)
- `linear-ppe`: a learned linear projection of the sinusoidal positional
  encoding, on top of `linear`. Came from the question "wouldn't its
  own embedding allow the temporal component to get orthogonal with the
  rest?" — which turned out to be experimentally true.

`12eaeec` gitignored Claude Code session scratch files. `0fcf1b9`
softened overclaims from small differences across several Results
subsections (this is a recurring theme: tightening framing as the paired
tests later sharpened the picture).

`79fe54e` consolidated the experiments into a single
`fft_encode.reproduce` entry point with stage flags (`main`, `dmodel`,
`geometry`, `probe`, `mask`, `etth1`, `convergence`), replacing two
overlapping scripts (`run_all.py` and `run_300ep_all.py`). FDM was
stripped from the code along with several dead-code modules
(`carrier_grid.py`, `scale_dmodel.py`, etc.). The top-level repo got a
cleanup pass — 22 stale logs, loose result JSONs, and abandoned
`results_full*/` directories deleted.

## Phase 4 — The overnight reproducer run and its consequences (May 12)

`644da2b` was an overnight run of `fft_encode.reproduce --out
results_paper/` at the new uniform 300-epoch budget. Everything in the
paper from here on is derivable from the JSONs in `results_paper/`. Most
findings reproduced within seed noise. One surprise: **`cat` posted the
lowest mean NLL on ETTh1**, even though it underperformed on the
synthetic benchmark. We added it to the paper with deliberately hedged
framing (overlapping CIs vs `concat`/`linear`, indistinguishable
accuracies in the top tier) — a domain-dependent gap narrowing, not a
clean win.

Several substantive deepening commits followed in quick succession:

- `ec9203c` added a Discussion paragraph on the residual stream as the
  reason channel identity survives downstream (probes at layer 3 recover
  $R^2 \ge 0.84$ — that's an architectural consequence, not something
  attention/FFN learns).
- `f121adb` expanded the Methods section to give proper recipes for the
  Gram, encoding-space, and linear-probing analyses, and added a §3.5
  bias ablation showing the per-channel biases on `linear` are inessential
  to the orthogonalisation argument (they collapse to a single
  $d_{\text{model}}$-dim offset). The training-protocol section grew a
  paired-seed paragraph that becomes load-bearing later.
- `fd801b9` and `0569ec9` added a `geom_largen` stage probing the
  distractor-norm noise-floor hypothesis: at $C=8$ with 10× training
  data, distractor norms drop from 0.55 to 0.22, confirming the
  finite-data noise-floor reading.
- `0b075a7` sharpened the `sum` failure paragraph in the Discussion to
  deliver an explicit information-theoretic argument: the encoder output
  is a function of the pointwise sum $S(t)$ only, so by the data-
  processing inequality no downstream model can recover the lost channel
  identity. The probe's $R^2 \approx 0.24$ for `sum` at $C=4$ is exactly
  the theoretical floor of $1/C$.
- `dd166c3` added a TikZ architecture diagram (Figure 1) showing the
  swappable encoder block in the middle of an otherwise fixed
  transformer pipeline.
- `2dd6db6` audited the encoder parameter counts in Table 1 against the
  actual implementations — `linear-ppe` was off by 64 (missed the
  positional-projection bias); `ci`/`cat` numbers were stale from an
  earlier code version. Numbers corrected, dagger footnote added
  explaining that "encoder params" is a slightly awkward fit for the
  architectural baselines.
- `be3c504` was the methodological turning point on small effects:
  applying paired-difference analysis to the existing per-seed JSONs
  showed the `linear-ppe` gap at $C=4$ is well above zero (paired
  $t = 4.81$, $p = 0.009$, 5/5 paired diffs positive), not "borderline"
  as the unpaired CIs suggested. Several places in the paper softened
  during `0fcf1b9` were then resharpened with the proper test.
- `9bb2fc8` extended paired tests to `cat` on ETTh1 (decisive vs
  `mlp`/`linear-ppe`, borderline vs `linear`/`concat`/`sum-ortho`) and
  `mlp` at $C=16$ (decisive vs the whole linear family, $p \in \{0.0003,
  0.008\}$, 5/5). The same commit added an explicit policy paragraph to
  §3.3 — report unpaired CIs, test paired where the unpaired comparison
  is borderline — naming the three call-out cases. It also added a
  `main_largen` stage scaffolding for a follow-up sweep.

## Phase 5 — Stress-testing findings under more data (May 12–13)

`694ed29` folded in the results of the `main_largen` 5×5 run at $C=16$,
$N_{\text{series}}=5120$ (10× the main-sweep data). Key result:

- `mlp`'s lead at $C=16$ **persists** in the data-rich regime (still 5/5
  paired diffs favouring `mlp`, $p \in \{0.0003, 0.018\}$).
- But the magnitude **shrinks ~5×**: $\Delta$NLL drops from 0.068 at
  $N=512$ to 0.014 at $N=5120$ against `linear`.
- All encoders gained ~0.4 NLL from the extra data, confirming the
  $N=512$ main sweep is data-limited at $C=16$ (same story we'd already
  seen at $C=4$ via `geom_largen`).

The conclusion: the encoder landscape isn't completely flat (different
encoders lead at different $C$ and dataset), but the practical gaps shrink
substantially when the model isn't data-limited.

`7a1a828` extended the `dmodel` sweep from sum+concat to all top-tier
encoders at $d_{\text{model}} \in \{64, 128, 256\}$. Three findings:

1. `linear-ppe` leads at *every* $d_{\text{model}}$ (14/15 paired diffs
   across the three widths favour it).
2. `mlp` *degrades* with $d_{\text{model}}$ at $C=4$ — it's the worst of
   the top tier at $d=128$ and $d=256$. Its $d^2$ encoder params
   overfit hardest.
3. All encoders flag as overfitting at $d \ge 128$ under the main-sweep
   $N$; the reported best-NLL numbers are effective-early-stopping
   minima, not converged trajectories. $C=4$ with $N=512$ doesn't justify
   a 256-dim residual stream.

The "Nonlinearity is redundant" Discussion paragraph was rewritten as
"Nonlinearity is redundant at small $C$, harmful with width at small
$C$, useful at large $C$" — the first heading that openly acknowledged
the encoder rankings interact with $C$ and $d_{\text{model}}$.

## Phase 6 — Settling the `linear-ppe` mechanism (May 15)

By this point the open question was: *why* does `linear-ppe` help?
Two candidate readings had been floating in the paper:

- **Compression**: the learned $W_{\text{pos}}$ rotation pushes the
  positional encoding into fewer effective directions, freeing
  residual-stream capacity.
- **Orthogonalisation**: it rotates positional dirs to be orthogonal to
  the channel subspace $\text{span}(W_k)$.

`e473d93` added a `pospro_geometry` stage that measures both directly,
on five seeds each of `linear` and `linear-ppe` at $C=4$. The verdicts
were opposite:

- The fraction of $P$'s Frobenius energy lying inside $\text{span}(W)$
  drops $\sim 4.5\times$ under `linear-ppe` (3.2% → 0.7%) — the
  rotation actively pushes $P$ out of the channel subspace.
  **Orthogonalisation confirmed.**
- $P$'s effective rank under `linear-ppe` is *higher* (9.37 vs 7.59),
  not lower. **Compression contradicted** — the rotation spreads
  positional energy across more directions, not fewer.

A third diagnostic — principal angles — turned out degenerate at this
dimensionality (the 4-dim $\text{span}(W)$ is trivially contained in
the rank-22 $\text{span}(P)$ in 64-D ambient space, so all principal
angles come out 0° regardless of encoder). A bug in the QR-based
implementation surfaced this; the SVD-based fix didn't change the
qualitative conclusion (the principal-angles metric remains
uninformative here, but the energy-fraction metric resolved the
mechanism cleanly).

With orthogonalisation as the confirmed mechanism, the $C$-dependent
collapse of the `linear-ppe` gap (from $\Delta = 0.054$ at $C=4$ to
$\Delta = 0.008$ at $C=16$) admits a natural reading: $W_{\text{pos}}$
must rotate $P$ out of a $C$-dimensional channel span using a fixed
$d_{\text{model}}^2$ parameter budget, and the rotation becomes
increasingly over-constrained as $C$ grows. This is the only one of the
three speculative explanations the paper had previously listed that
directly invokes the empirically-supported mechanism.

## Phase 7 — Cleanup (May 15)

`ee03e89` un-tracked a recent-token-rotation file that was accidentally
committed (`ghtoken_old`); a follow-up `git filter-branch` + force-push
scrubbed it from history entirely (the token was already expired, so
this was hygiene not security).

`69b8b94` was the editorial pass: reorder the Discussion paragraphs
into a clean grouping (why linear works → where linear isn't unique →
practical → extrapolation), move §3.10 Pospro to sit just before
§3.11 ETTh1 so the Results flow reads main → scalings → diagnostics →
convergence/cost → linear-ppe deep dive → real-data validation, and
trim the more speculative "Reading for non-numerical inputs" paragraph
in the Discussion.

## Where we ended up

A 20-page manuscript whose headline is: **at convergence, the standard
per-channel linear projection `nn.Linear(C, d_model)` matches every
alternative we tested.** Eight encoders on a controlled synthetic
benchmark + ETTh1 real-data validation, with paired-difference
statistics where small effects sit at the resolution boundary.

Three encoders deviate from the linear family under paired analysis:

- **`linear-ppe`** wins by $\sim 0.05$ NLL at $C=4$ (paired $p = 0.009$);
  gap shrinks with $C$ and is absent by $C=16$. Mechanism: the learned
  positional projection rotates the positional encoding into directions
  orthogonal to the channel subspace (confirmed by direct geometric
  measurement, not compression as one might have guessed).
- **`mlp`** leads at $C=16$ (paired $p \in \{0.0003, 0.008\}$, 5/5
  favouring `mlp` against every linear-family member). The lead persists
  at 10× data but shrinks to $\sim 0.015$ NLL — the original
  measurement is partly amplified by the data-limited regime.
- **`cat`** posts the lowest mean NLL on ETTh1 (paired tests decisive vs
  `mlp`/`linear-ppe`, borderline vs the rest). Domain-dependent
  narrowing of the synthetic gap, not a robust win.

The `sum` failure is also given a tight argument: the encoder output
algebraically simplifies to $W \cdot S(t) + \text{const} + \mathbf{p}(t)$
where $S(t) = \sum_k v_k(t)$, so the data-processing inequality bounds
what any downstream model can recover. The empirical probe $R^2
\approx 0.24$ for `sum` at $C=4$ is exactly the theoretical floor of
$1/C$.

The reproducer (`fft_encode.reproduce`) covers ten stages and produces
the canonical `results_paper/` JSONs in roughly 12 GPU-hours on a single
GPU. Every number in the paper is derivable from those JSONs.

## What changed about the work as it evolved

The biggest shift was the May 11 pivot. The original framing positioned
FDM as the architectural contribution and treated `sum`/`concat`/`linear`
as comparisons. That framing was unstable: FDM didn't actually win,
`sum-perch` (per-channel projection then sum, which renamed to `linear`)
isn't a novel proposal but the default an experienced PyTorch user
would write, and the interesting story turned out to be *why* the default
works. The pivot reframed the paper as an audit of an obvious-but-
unexamined default rather than a proposal of a new encoder.

The second-biggest shift was methodological: applying paired-difference
analysis to the existing seed data (May 12) tightened several findings
that had been hedged as "borderline." This wasn't a new experiment —
it was a new way of reading data that already existed. The paper's
training-protocol section now describes the report-vs-test policy
explicitly so this isn't a one-off.

The mechanism question for `linear-ppe` survived the longest as a
hedged open question. It sat as "we discuss possible mechanisms but
neither is pinned down" for about three weeks before the
`pospro_geometry` measurement (May 15) settled it — orthogonalisation,
not compression. The fix wasn't expensive (~10 minutes of GPU time and
~100 lines of code) but it took that long to arrive at a clean
discriminator metric and a working implementation. The principal-angles
detour during the same commit is a useful reminder that the obvious
geometric diagnostic isn't always the right one for the question.
