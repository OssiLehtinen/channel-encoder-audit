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

The initial commit (`5d125f7`) shipped a complete first-pass paper around
this premise: four encoders (`sum`, `concat`, `fdm`, `fdm-learn`), a
synthetic multi-signal benchmark with deliberate cross-channel
interactions, a sweep runner, linear-probe channel-recovery and
test-time channel-masking diagnostics, and a manuscript with all tables
regenerated from JSON outputs.

The original title was *FDM Channel Encoding*; the headline result was
that `fdm` (NLL 2.85) beat `sum` (3.25) and lost narrowly to `concat`
(2.43). The hypothesis was that the *carrier* — the sinusoidal modulation
inside each per-channel block — was buying the model something.

Same-day follow-ups (`06463e5`, `3545aa8`, `523e65b`, `90dddee`) added
authorship and expanded NLL/RoPE/FFT abbreviations on first use. `3ea310e`
added a `d_model` scaling study and a carrier-band grid search, both
aimed at characterising FDM's behaviour.

## Phase 2 — Sharper training, broader baselines (April 14–16)

The 12-epoch initial training turned out to be undertrained for `concat`;
a 100-epoch cosine sweep (`51abac8`) substantially rewrote the headline
table. Three new baselines came in alongside:

- `linear-ortho`: `sum` with a soft orthogonality regulariser on the learned
  per-channel projections.
- `ci` (channel-independent, PatchTST-spirit): shared-weight transformer
  per channel.
- `cat` (channel-as-token, iTransformer-spirit): each $(t, k)$ a token.

Headline at this point: `concat` and `linear-ortho` tied at NLL 2.42, both
cleanly above `fdm` (2.85), with `sum` structurally capped at 3.25
regardless of width. `ci` and `cat` underperformed at matched compute.

ETTh1 was added as a real-data check (`0b1dc44`) and an off-diagonal Gram
analysis confirmed the learned per-channel projections were already
near-orthogonal under `linear-ortho`.

At this point the paper was still framed around FDM as the headline
encoder, with `concat`/`linear-ortho` presented as alternatives that happen
to perform similarly.

## Phase 3 — The pivot: from FDM to subspace geometry (May 11)

This was the largest single shift. The premise that *FDM specifically* was
doing something interesting started to look implausible. A direct
question — *is the SUM-PERCH approach really novel?* — exposed that
`sum-perch` (per-channel projection $W_k$ then sum) is mathematically
just `nn.Linear(C, d_model)`. It's the most obvious encoder a PyTorch
user would write. There was no architectural novelty there.

`0b9cbc6` ("paper: full rewrite as subspace-geometry audit") reframed
the manuscript around a geometric reading: distinct input streams need
approximately orthogonal subspaces in $d_{\text{model}}$ so the model
can recover stream-of-origin. The shared-projection `sum` baseline is the
one that violates this; everything else (`linear`, `linear-ortho`, `concat`,
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

`2978dc6` gitignored Claude Code session scratch files. `4cee7cb`
softened overclaims from small differences across several Results
subsections (this is a recurring theme: tightening framing as the paired
tests later sharpened the picture).

`510d86b` consolidated the experiments into a single
`fft_encode.reproduce` entry point with stage flags (`main`, `dmodel`,
`geometry`, `probe`, `mask`, `etth1`, `convergence`), replacing two
overlapping scripts (`run_all.py` and `run_300ep_all.py`). FDM was
stripped from the code along with several dead-code modules
(`carrier_grid.py`, `scale_dmodel.py`, etc.). The top-level repo got a
cleanup pass — 22 stale logs, loose result JSONs, and abandoned
`results_full*/` directories deleted.

## Phase 4 — The overnight reproducer run and its consequences (May 12)

`efb8843` was an overnight run of `fft_encode.reproduce --out
results_paper/` at the new uniform 300-epoch budget. Everything in the
paper from here on is derivable from the JSONs in `results_paper/`. Most
findings reproduced within seed noise. One surprise: **`cat` posted the
lowest mean NLL on ETTh1**, even though it underperformed on the
synthetic benchmark. We added it to the paper with deliberately hedged
framing (overlapping CIs vs `concat`/`linear`, indistinguishable
accuracies in the top tier) — a domain-dependent gap narrowing, not a
clean win.

Several substantive deepening commits followed in quick succession:

- `4ef0ba1` added a Discussion paragraph on the residual stream as the
  reason channel identity survives downstream (probes at layer 3 recover
  $R^2 \ge 0.84$ — that's an architectural consequence, not something
  attention/FFN learns).
- `baf43cf` expanded the Methods section to give proper recipes for the
  Gram, encoding-space, and linear-probing analyses, and added a §3.5
  bias ablation showing the per-channel biases on `linear` are inessential
  to the orthogonalisation argument (they collapse to a single
  $d_{\text{model}}$-dim offset). The training-protocol section grew a
  paired-seed paragraph that becomes load-bearing later.
- `d61f07b` and `3f43bcd` added a `geom_largen` stage probing the
  distractor-norm noise-floor hypothesis: at $C=8$ with 10× training
  data, distractor norms drop from 0.55 to 0.22, confirming the
  finite-data noise-floor reading.
- `8191b9d` sharpened the `sum` failure paragraph in the Discussion to
  deliver an explicit information-theoretic argument: the encoder output
  is a function of the pointwise sum $S(t)$ only, so by the data-
  processing inequality no downstream model can recover the lost channel
  identity. The probe's $R^2 \approx 0.24$ for `sum` at $C=4$ is exactly
  the theoretical floor of $1/C$.
- `41e5c99` added a TikZ architecture diagram (Figure 1) showing the
  swappable encoder block in the middle of an otherwise fixed
  transformer pipeline.
- `bd150cf` audited the encoder parameter counts in Table 1 against the
  actual implementations — `linear-ppe` was off by 64 (missed the
  positional-projection bias); `ci`/`cat` numbers were stale from an
  earlier code version. Numbers corrected, dagger footnote added
  explaining that "encoder params" is a slightly awkward fit for the
  architectural baselines.
- `fd0a646` was the methodological turning point on small effects:
  applying paired-difference analysis to the existing per-seed JSONs
  showed the `linear-ppe` gap at $C=4$ is well above zero (paired
  $t = 4.81$, $p = 0.009$, 5/5 paired diffs positive), not "borderline"
  as the unpaired CIs suggested. Several places in the paper softened
  during `4cee7cb` were then resharpened with the proper test.
- `a72d7ca` extended paired tests to `cat` on ETTh1 (decisive vs
  `mlp`/`linear-ppe`, borderline vs `linear`/`concat`/`linear-ortho`) and
  `mlp` at $C=16$ (decisive vs the whole linear family, $p \in \{0.0003,
  0.008\}$, 5/5). The same commit added an explicit policy paragraph to
  §3.3 — report unpaired CIs, test paired where the unpaired comparison
  is borderline — naming the three call-out cases. It also added a
  `main_largen` stage scaffolding for a follow-up sweep.

## Phase 5 — Stress-testing findings under more data (May 12–13)

`03bdadb` folded in the results of the `main_largen` 5×5 run at $C=16$,
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

`74b2f2b` extended the `dmodel` sweep from sum+concat to all top-tier
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

`fb7af33` added a `pospro_geometry` stage that measures both directly,
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

`a8a0a0c` was the editorial pass: reorder the Discussion paragraphs
into a clean grouping (why linear works → where linear isn't unique →
practical → extrapolation), move §3.10 Pospro to sit just before
§3.11 ETTh1 so the Results flow reads main → scalings → diagnostics →
convergence/cost → linear-ppe deep dive → real-data validation, and
trim the more speculative "Reading for non-numerical inputs" paragraph
in the Discussion.

## Phase 8 — Scaling up to 20 seeds (May 15–18)

The 5-seed runs had been good enough to detect the large effects but
left several small ones (linear-ppe at $C=16$, mlp at $C=16$ vs.
linear-ppe) sitting at the resolution boundary of the paired test.
Rather than dropping those findings into the noise, we added more
seeds.

`a216aac` introduced an `extra_seeds` stage to the reproducer: an
open-ended round-robin that, starting from a configurable seed
(default 5, so canonical seeds 0–4 are preserved), runs one full
cycle of the paired-test-sensitive stages and writes per-(stage, seed)
JSON files. Each cycle is independent and resumable; killing the
process at any point leaves a balanced set of additional runs on
disk. The same commit also added per-example validation NLLs to the
trace dicts so cluster-bootstrap (resample seeds, pool their
per-example diffs) becomes a downstream option, not just seed-level
bootstrap.

`63a2f6a` extended the round-robin to also rotate through `geometry`
and `pospro_geometry`, so the Gram-matrix and positional-projection
diagnostics get the same paired-seed treatment as the headline NLL
comparisons.

`f270d30` is the analysis pipeline (`fft_encode.analyze`): merges
canonical and extra-seed JSONs into per-stage flat lists, runs every
paired comparison the paper makes, and reports both paired
$t$-tests and bootstrap CIs (seed-level and cluster). The output is
a single `analysis_<N>seeds.{json,txt}` file that drives the paper's
$p$-values and intervals.

`bc7850d` is just the result of running the round-robin for 15
cycles (seeds 5–19), giving a 20-seed pool across every headline
configuration. The full sweep took about a day of GPU time
overnight; nothing else changed except more JSON files appearing
under `results_paper/extra_seeds/`.

`e55ad53` ports the paper to the new numbers. Several effects that
had been borderline at 5 seeds now resolve cleanly: linear-ppe at
$C=8$ goes from "marginal" to $p=3.7\times 10^{-4}$, linear-ppe at
$C=16$ goes from "null" to detectable ($p=0.036$, just above zero),
mlp at $C=16$ vs. the linear family is decisive but mlp vs.
linear-ppe stays borderline ($p=0.053$). The phrase "practical
near-equivalence" enters the abstract here — the 20-seed analysis
*resolves* small inter-encoder gaps as statistically clear, but
they're small relative to typical task-design variation, and the
honest summary is that the encoder choice is, in any practical
sense, close to a free parameter inside the per-channel-$W_k$
family.

## Phase 9 — Loss-family check and MLP geometry (May 18)

Two follow-up experiments emerged from the 20-seed reading.

`c3313b2` flagged a real thing while looking at the convergence
diagnostics: `ci` doesn't just underperform on synthetic — it
*overfits universally*. All 20 seeds at every $C \in \{4, 8, 16\}$
flag "overfitting", best val NLL is reached at epoch ${\sim}40$–60,
and val NLL drifts upward by 0.12–0.29 nats afterwards. The
reported `ci` NLL is the effective-early-stopping minimum, not a
converged plateau, and the paper now says so. The same commit also
introduced a `main_mse` stage to test whether the encoder ordering
is sensitive to the categorical-vs.-regression head choice — could
the bin head specifically be advantaging encoders that resolve fine
target quantiles?

`60749cf` is the rename plus the new diagnostic. The "ortho" in
`sum-ortho` sat on per-channel $W_k$ rows, not on the summation
step — calling it `sum-ortho` was a leftover from before `sum-perch`
was renamed to `linear`. The encoder is now `linear-ortho` across
paper, tables, code, and outputs, with backward-compat aliases on
every read path so older JSONs (which have `sum-ortho` baked into
`cfg.encoder`) continue to load. The same commit added
`mlp_geometry`, which trains `mlp` at $C \in \{4, 8, 16\}$ and
applies the off-diagonal cosine diagnostic to $W^{(1)}$'s columns
(the per-channel input directions before the GELU). Finding:
spontaneous near-orthogonality is real for `mlp` too but
consistently looser than the linear family's $W_k$ (mean $|\cos|$
$0.075$ vs.\ $0.042$ at $C=4$; widens to $0.105$ / max $0.46$ at
$C=16$). The qualitative claim — *task loss + per-channel input
stem $\Rightarrow$ near-orthogonal channel directions* — survives;
the quantitative cleanest geometry is the bare linear family's.

The MSE check came back unambiguous: with a scalar regression head
in place of the categorical 32-bin head, the encoder ordering is
preserved at both $C \in \{4, 16\}$ tested. `sum` collapses to
$R^2 \approx 0.13$ at $C=4$ and $\approx 0.01$ at $C=16$,
consistent with the same information-theoretic ceiling; the
per-channel-$W_k$ tier clusters tight; `linear-ppe` edges `linear`
at $C=4$ and `mlp` edges `linear` at $C=16$, mirroring the
categorical sweep. The label-binning choice is not what's driving
the encoder ordering.

## Phase 10 — Editorial passes (May 18–21)

By this point the paper had accumulated the kind of redundancy
that happens when several small additions are stacked one on top
of another. Four editorial passes, in order:

`f995c00` is the first big restructuring. Abstract reduced to a
qualitative summary (the dense paired-$p$ values were not earning
their abstract real estate); "Open questions" reframed in the
introduction (sum is now positioned as a verified floor, not an
open question — its failure mode is information-theoretic and the
only quantity of interest is how much downstream capacity fails to
recover); contribution paragraph reorganised to match the
question/answer pairing; ETTh1 framing hedged ("the lowest mean
NLL but statistically tied with linear" rather than "decisively
wins").

`56e3060` is the aggressive Discussion cut: seven of the Discussion
paragraphs essentially restated Results subsections with the same
numbers (Orthogonality is spontaneous; Importance weighting is
automatic; When nonlinearity helps; Projected positional encoding;
Channel-independent; Channel-as-token; No convergence-time
penalty), so the Discussion was made to do its proper job —
interpretation and generalisation — by keeping only the five
paragraphs that aren't restatements (linear-is-not-new,
what-fails-about-sum, residual-stream-end-to-end, reading for
non-numerical inputs, practical-near-equivalence within the top
tier). The Discussion shrank by ${\sim}40\%$ and the Conclusion was
compressed in the same pass.

`2c9eb9a` added five citations identified during the editorial
re-read: Ke et al. 2021 ("Rethinking Positional Encoding") is the
closest prior art for `linear-ppe` — they untie positional
embeddings from content embeddings in language transformers via a
separate projection, which is exactly the principle `linear-ppe`
applies to the channel-position competition; TFT (Lim et al. 2021)
introduces Variable Selection Networks as an explicit
channel-importance gating mechanism, which our
spontaneous-importance-weighting result is partly negative
evidence against in the homogeneous-numerical setting; TSMixer
(Chen et al. 2023) is the all-MLP architecture whose channel-mixing
block our `mlp` encoder is essentially a one-layer version of;
Elhage et al. 2021 ("A Mathematical Framework for Transformer
Circuits") formalises the residual-stream view that the deep-probe
argument relies on; Alain and Bengio 2017 is the canonical
probing-methodology reference.

`9034197` is the third editorial pass: spotted a numerical
inconsistency in the Discussion (stale 5-seed PPE $\Delta$ values
0.054/0.008 that contradicted the 20-seed 0.041/0.016 reported in
the Results PPE subsection), removed via the cut; §2.5's seed
default ("five seeds unless stated otherwise") flipped to lead with
20 seeds and name the 5-seed exceptions; the bias-ablation
subsection collapsed into a paragraph at the end of §3.5 (a 2-row
table for a one-paragraph note was over-billed); the C=16 10× data
sub-paragraph promoted to its own §3.3 "Data-richness check"
subsection. Plus a handful of phrasing tics (no more "Read
carefully"; no more single-word "Three readings." sentence;
"actually halt" demoted to "halt"; bibliography sorted
alphabetically).

`c4f4dcd` is the bundled honesty pass spread across three reading
passes in the same evening: (i) `ci` and `cat` are
*full-architecture alternatives*, not encoder swaps — the §2.1
opener and the Figure 1 caption were oversold the shared-backbone
claim because `ci` runs the transformer per-channel on
$(B \cdot C, T, d_{\text{model}})$ and `cat` embeds a $C \cdot T$-long
sequence over $(B, C \cdot T, d_{\text{model}})$; the diagram +
caption now scope the encoder-swap framing to the six variants
that actually share the I/O signature, and identify `ci`/`cat` as
architectural alternatives. (ii) `ci` is identified as a *second
decisive loser* alongside `sum` — the previous "the only encoder
that loses decisively is sum" understated `ci`'s ${\sim}0.88$ NLL
gap to the top tier and its only ${\sim}0.2$ NLL above sum's
information-theoretic floor; `ci` is closer to sum than to the
linear baseline. (iii) The no-warmup choice gets a one-sentence
pre-LN-stability defense citing Xiong et al. 2020. (iv) `tab:gram`
gains $\pm$std notation for visual parity with `tab:main`
(previously it was easy to miss that the two tables report the
same quantity from the same 20-seed runs); `tab:mlp-gram` caption
and the bias-ablation paragraph both note their 5-seed sub-sweep
status against the 20-seed `tab:main` to forestall the
why-are-these-numbers-different question. (v) Sentences claiming
linear-ppe achieves the lowest val NLL at every $C$ were
overreaches — at $C=16$ `mlp` has the lower mean ($2.231$ vs.
$2.252$, paired $p=0.053$). §3.2 and §3.10 now say so explicitly;
the "linear-ppe leads the linear family" claim is preserved where
it's actually true (i.e. against {linear, linear-ortho, concat}).
Manual stylistic polishing on top of all the above: introduction
restructured to flow as prose rather than `\paragraph`-headed
structure; author footnote acknowledges Claude Code and Opus
4.x assistance; sharpening in §2.2.

## Phase 11 — Code review (May 19)

By this point the paper had stabilised and we did one
pass over the code with the same critical eye.

`e94d4dd` fixed the priority-1 cluster of issues:

- **scipy** was imported by `analyze.py:27` but not declared in
  `pyproject.toml` dependencies — a clean `uv sync` followed by
  `uv run python -m fft_encode.analyze` would fail with
  `ModuleNotFoundError`. Added `scipy>=1.11`.
- **`extra_seeds` was in the default `--stages` set**, so
  `python -m fft_encode.reproduce --out results/` would run every
  finite stage and then loop forever in the round-robin (which is
  KeyboardInterrupt-driven by design). A `DEFAULT_STAGES =
  ALL_STAGES \ {extra_seeds}` split makes the canonical
  reproduction command terminate.
- **`ENCODERS_DMODEL` defaulted to `["sum", "concat"]`**, which
  doesn't reproduce Table 3 of the paper (which covers all six
  top-tier encoders + sum). Fixed.
- **Three stale comparison labels in `analyze.py`** still said
  "sum-ortho" after the rename — the encoder argument was correct
  (`"linear-ortho"`) but the human-readable label flowed through
  to the analysis output and would confuse readers.
- **`ChannelAsTokenTransformer` docstring** described a strict
  left-to-right within-time mask and last-channel-token
  prediction, but the actual `_causal_mask` is bidirectional
  within a time step and the `forward` mean-pools. Docstring
  rewritten to match the code.
- **Three runtime `assert`s** (`data.py`, `encodings.py`,
  `reproduce.py`) were guarding user-input preconditions —
  asserts are stripped under `python -O`, so they were replaced
  with `raise ValueError(...)`.
- **A dead `layer` variable** in `SignalTransformer.__init__`
  was a leftover from when this used
  `nn.TransformerEncoder(layer, num_layers=...)` before the
  switch to a manual `ModuleList` (which is what lets the probe
  capture per-layer hidden states).

`85ebb57` added `TODO.md` capturing the deferred items: refactors
(consolidate the two ETTh1 training paths;
factor `stage_extra_seeds` so each stage exposes a one-seed entry
point; merge `train_and_trace_mse` into `train_and_trace` via a
`target_type` parameter; rename `SumOrthoEncoding` to `LinearEncoding`
or split into per-paper-encoder subclasses; dedupe the `_canon`
alias helper across three files; split `reproduce.py` into a
subpackage) and minor maintainability (hash-verify the ETTh1
download against upstream drift; document or name-constant the
convergence-flag thresholds; move MLP gram analysis onto the
encoder class). These are medium-effort and not blocking; they
sit in TODO.md so a future-me can pick them up.

## Phase 12 — Repo move and post-rename polish (May 21–22)

The repository was renamed from `FDM-encoding` (a leftover from the
original frequency-domain-multiplexing framing dropped in May 11) to
`channel-encoder-audit`, which matches the paper's title and audit
framing. The new repo was seeded by replaying the previous repo's
commits via `git format-patch --root` plus `git am --ignore-date`,
scrubbing the single token-misstep commit (the
`ghtoken_old`-gitignore one) along the way and resetting author and
committer dates to the move date. 42 commits made it across; the
author email was rewritten from the placeholder to the canonical
`ossi@ocon.fi` in the same pass.

Several small follow-up edits then landed in the new repo.

The in-paper URL was updated to point at the new repo. The bare
`\url{...}` after "Code:" in the abstract was first replaced with a
footnote (clean reading text, URL tucked away as
`\footnote{\url{...}}`), and the URL itself changed to
`channel-encoder-audit`.

The author footnote was rendering badly: a `\thanks` + `\footnote`
combination put the asterisk and dagger marks on top of each other
at title-page font size, and a leading `\\` inside the footnote body
had pushed the dagger label onto its own line. Two attempts to use
two clean `\thanks` calls instead didn't fix the mark overlap. The
working fix was a single `\thanks` containing both the email and
the Claude-Code disclosure separated by a period — one marker, no
overlap possible. Also dropped a `\texttt{}` wrapper around the
disclosure's running prose (typewriter for a sentence of English
text is ugly).

§2.2's `linear-ppe` definition was rewritten in light of a
clarifying question: is `linear-ppe` *enforcing* orthogonalisation
via an architectural prior, or *enabling* a degree of freedom that
the optimiser then uses? The latter is the correct reading, and the
existing wording had additionally described the rotation going the
wrong direction — "rotate the positional subspace **to** a channel
subspace better compatible with the signal channel encodings",
which reads like rotating `p(t)` *into* `span(W)` even though §3.10's
measurement shows the rotation pushes `p(t)` *out of* `span(W)`.
The new wording makes both corrections: the sinusoidal basis is
otherwise fixed at initialisation and cannot move; `linear-ppe`'s
only addition is the missing degree of freedom, and the cross-stream
gradient pressure already present in vanilla `linear` (the same
pressure that organises `W` toward near-orthogonality) does the rest
once the position side has a parameter to move with.

README and REPRODUCE.md were brought up to date with the current
paper state. Both had been frozen at roughly the May 14 state, so
they still claimed 5 seeds throughout, listed only 7 of the
reproducer's now 13 finite stages, framed `ci` as "underperforming"
rather than as a decisive loser, and described the `linear-ppe`
mechanism as "not pinned down here". Numbers replaced with the
20-seed values from `tab:main` / `tab:etth1`, stage table extended
to the full 13 plus the opt-in `extra_seeds`, framing aligned with
the current Discussion. The `dmodel` stage's encoder default — which
the code-review pass (Phase 11) had also fixed — now says "6
top-tier encoders" instead of the stale "sum + concat".

## Where we ended up

A 21-page manuscript whose headline is: **the standard per-channel
linear projection `nn.Linear(C, d_model)` matches every alternative
we tested up to small, statistically real but practically modest
differences.** Eight encoders on a controlled synthetic benchmark +
ETTh1 real-data validation, at 20 paired seeds for headline
comparisons (with a few diagnostics at 5 seeds, labelled as such),
paired-difference statistics and bootstrap CIs where small effects
matter.

Two encoders lose decisively:

- **`sum`** (shared-scalar baseline) collapses for
  information-theoretic reasons: the encoder output algebraically
  simplifies to $W \cdot S(t) + \text{const} + \mathbf{p}(t)$ with
  $S(t) = \sum_k v_k(t)$, the data-processing inequality bounds
  what any downstream model can recover, and the empirical layer-0
  probe sits at $R^2 \approx 0.24 \approx 1/C$ exactly at the
  theoretical floor.
- **`ci`** (channel-independent) underperforms throughout (${\sim}0.88$
  NLL below the top tier at $C=4$ — closer to `sum`'s floor than
  to the linear baseline), overfits universally at every $C$, and
  costs ${\sim}8\times$ wall clock at $C=16$. Two contributing
  failures: the head-side bottleneck (combining $C$ per-channel
  hidden streams into a single prediction loses the cross-channel
  interaction the task is built on) and the universal overfitting
  (best val at epoch ${\sim}40$–60, drift up by 0.12–0.29 nats
  afterwards).

Three encoders deviate from the linear family under paired analysis
without losing the practical-near-equivalence headline:

- **`linear-ppe`** wins by ${\sim}0.041$ NLL at $C=4$ (paired
  $p = 2.6 \times 10^{-6}$, 19/20); the lead shrinks with $C$ to
  $0.026$ at $C=8$ ($p = 3.7 \times 10^{-4}$) and $0.016$ at $C=16$
  ($p = 0.036$). The 20-seed analysis pushed the $C=16$ gap above
  zero where the 5-seed reading had been null. Mechanism: the
  learned positional projection rotates the positional encoding
  into directions orthogonal to the channel subspace (direct
  geometric measurement; the fraction of $P$'s energy inside
  $\mathrm{span}(W)$ drops ${\sim}6.3\times$). Not compression, as
  one might have guessed — $P$'s effective rank is *higher* under
  `linear-ppe`, not lower. At $C=16$ `mlp` takes the lower mean
  ($2.231$ vs.\ `linear-ppe`'s $2.252$), so `linear-ppe` is the
  *linear-family* leader rather than the absolute leader at the
  largest $C$ we test.
- **`mlp`** edges the linear family at $C=16$ (paired
  $p \in [3 \times 10^{-4}, 5 \times 10^{-12}]$) but ties
  `linear-ppe` ($p = 0.053$). The lead persists at 10× data with
  decisive paired statistics against `linear-ppe`
  ($p = 2.4 \times 10^{-4}$) but shrinks ${\sim}3\times$ in
  magnitude — the original measurement is partly amplified by the
  data-limited regime. At $C=4, d_{\text{model}} \ge 128$ `mlp`
  is the *worst* of the top tier (overfits hardest because its
  encoder parameters scale as $d_{\text{model}}^2$).
- **`cat`** posts the lowest mean NLL on ETTh1 ($0.551$), paired
  $p = 1.4 \times 10^{-7}$ vs. `mlp` and $p = 3 \times 10^{-4}$ vs.
  `linear-ppe`, but statistically tied with `linear` ($p = 0.14$)
  and `linear-ortho` ($p = 0.10$). Domain-dependent narrowing of
  the synthetic-benchmark gap, not a robust win.

The reproducer (`fft_encode.reproduce`) covers 13 finite stages
(plus the optional open-ended `extra_seeds` round-robin) and
produces the canonical `results_paper/` JSONs in roughly 12
GPU-hours on a single GPU. Every number in the paper is derivable
from those JSONs via `fft_encode.analyze`.

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

The 20-seed expansion (Phase 8) was the third methodological shift,
and the one that pinned down the "practical near-equivalence" frame
that became the paper's headline. At 5 seeds we had two clear
findings (sum collapses; linear-family clusters at the top), one
solid finding (linear-ppe edges linear at $C=4$), and a handful of
borderline ones (linear-ppe at $C=8, 16$; mlp vs linear at $C=16$;
mlp vs linear-ppe at $C=16$). The instinct on borderline findings
is usually to drop them or hedge harder, but the right move turned
out to be: get more seeds. Each borderline gap that resolved at 20
seeds added a small quantitative claim the paper can defend;
together they let us *say* "small but statistically real, practically
modest" rather than the looser "we can't quite resolve this".

The big lesson from the editorial passes (Phase 10) is that
Discussion sections tend to grow by accumulation — each new
sub-experiment adds a Discussion paragraph that interprets it, and
those paragraphs accumulate without anyone noticing they're
restating the corresponding Results subsection with the same
numbers. The aggressive cut in `56e3060` dropped seven such
paragraphs and the Discussion got *better* for it. Worth doing
proactively on future papers rather than waiting for someone to
notice.

The "ci as second decisive loser" item in Phase 10's bundle
(`c4f4dcd`) is the kind of thing that's only obvious in retrospect.
We'd been calling `ci` "underperforming" for weeks because that's
what the original framing said, and nobody re-checked the gap
against the top tier until a fresh reading. At $C=4$ `ci` is at
NLL 3.053 — ${\sim}0.88$ below the linear family and only
${\sim}0.20$ above the `sum` floor. By any reasonable definition
that's decisive, but the verb "underperforming" was sticky enough
that two editorial passes had let it through. The lesson: when
revisiting language that's been around for a while, re-check the
underlying numbers against the language even if nothing about the
numbers has changed.

The code review (Phase 11) found seven real bugs that had survived
multiple read-throughs. The most consequential — the infinite-loop
default in `--stages` — was a footgun that would have hit any new
user running the canonical command, and it was created in
`extra_seeds`'s commit a few weeks before; nobody had run the
canonical command since because we'd been doing targeted
`--stages X` invocations the whole time. Worth re-running the
literal command in the README/REPRODUCE.md after any change to the
stage dispatcher, exactly the way a new user would.

The §2.2 `linear-ppe` rewrite in Phase 12 is a recurrence of the
"sticky-language" lesson from Phase 10's `ci`-as-decisive-loser
finding. The §2.2 definition had been carrying the wrong directional
description of the rotation since the encoder was introduced — the
prose said `p(t)` rotates *to* a channel-compatible subspace while
the §3.10 results section measured `p(t)` rotating *out of*
`span(W)`. Nobody noticed for weeks: the substance was always right
(the §3.10 measurement) and the prose-level description had drifted
without anyone re-reading it against the result. The same kind of
drift produced the `ci`-"underperforming" framing that Phase 10
sharpened to "decisive loser". Recurring lesson: re-read every
encoder *definition* against its *results* subsection at submission
time. Substance and language drift in opposite directions when only
one of them is being actively edited.
