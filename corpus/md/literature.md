# Literature synthesis — what 5 papers say about our wall

**TFM · Multimodal prediction and search of YouTube video performance**

> A **deep, critical read** of 5 papers, read *against our concrete problem*
> (not generic summaries). Canonical reference config: frozen SigLIP 2 base,
> real thumbnail, **2717 videos / 86 channels**, target
> `z = (log10(views) − μ_channel)/σ_channel`. The full journey and dead-ends are
> in [`README.md`](README.md). This document is **additive**; it replaces nothing.


## 0 · Why this review reframes the project (it is not giving up)

After 12+ of our own attacks, the cross-channel wall is structural: in-dist
Spearman ≈ **0.26–0.28**, cross-channel (LOCO) ≈ **0.04–0.10**, and the
intra-channel z ≈ white noise (AR(1)≈0). The open question was *"what else can we
do to break it?"*. The literature's answer is uncomfortable but **valuable for the
thesis**: the ceiling we measure **is not our failure — it is the known ceiling of
the problem**, and the high numbers that get published come from measuring
different (and easier) things. This turns the wall from *"we didn't make it"* into
*"we characterized it and corroborated it with peer-reviewed literature"*, which is
exactly the gradable deliverable.

### The 5 papers

| # | Short cite | What it predicts | Target | Why it matters to us |
|---|---|---|---|---|
| P1 | Trzciński & Rokita 2017, *IEEE TMM* (arXiv 1510.06223) | Absolute 30-day views from early trajectory | absolute views (Spearman) | **Ceiling anchor:** visual features (incl. ResNet-152) → ρ=**0.234** content-only, no channel normalization (easier than ours) |
| P2 | Wu, Rizoiu, Xie 2018, *AAAI* (arXiv 1709.02541) "Beyond Views" | Relative engagement (watch %) | relative percentile within duration bin | Their **predictable quantity is intra-channel stable**; our z is **intra-channel white noise** → z ≈ their *popularity* (the unpredictable part), not their *engagement* |
| P3 | Cui, Chung, Peng & Wang 2024, *J. Business Research* | 1-week views from title+thumbnail sentiment | absolute views (negative binomial) | The dominant effect is **log-subscribers (β≈+0.55)** = the between-channel variance our z **erases by construction**. They never test intra-channel |
| P4 | Nisa et al. 2021, *Electronics* 10(23) | "Popular / not popular" | binary (with like/comment leakage) | **Methodological foil:** 88% "accuracy" = leakage + imbalance + non-temporal random split |
| P5 | Ou 2025 (arXiv 2505.10664) | Real vs AI (images) | binary | Architecture analogue (frozen emb + light head + few-shot); works **only because the axis is universal** — which we do NOT have |


## 1 · Convergent finding #1 — the ~0.23–0.28 content ceiling is in the literature

Three independent papers, with comparable or larger datasets and more powerful
features, **all land on the same content-signal ceiling**:

- **P1 (IEEE TMM, peer-reviewed).** With 1,820 Facebook videos, deep visual
  features (ResNet-152) + thumbnail-popularity model + clutter + scene dynamics,
  **all visual features combined → Spearman 0.234 ± 0.017** (Table II, p.7). And
  that is an **easier** task than ours: absolute views, **no** channel
  normalization (the channel-size variance — trivially rankable — is still in).
  Even so they don't beat ~0.23 with content. Our in-dist ≈0.26–0.28 **is not
  anomalously low: it is consistent with the literature ceiling.**
  > *"visual features can be computed before the publication, while the others
  > cannot be obtained until the video is published and it is too late to modify
  > its contents"* (p.7) — and even so content caps at ~0.23.

- **P2 (AAAI, "Beyond Views").** Their **predictable** quantity (relative
  engagement, R²=0.77) is so **because it is intra-channel stable** (§3.4: 92.7%
  of videos change <0.1 between day 7 and 30). Our z is, by construction and
  verified (AR(1)≈0), **the opposite: intra-channel white noise**. Read against
  our data, this paper says our z is structurally **aligned with their
  *popularity*** ("unstable over time and driven by external promotions"), not
  with their *engagement*. It is the citation that **legitimizes reporting a
  ceiling instead of an R²=0.8** and separating "predicting engagement" from
  "predicting popularity".
  > *"engagement measures of a video are stable over time, thus separating the
  > concerns for modeling engagement and those for popularity – the latter is
  > known to be unstable over time and driven by external promotions"*
  > (Abstract, p.1).

- **P3 (J. Business Research, "Clicks for money").** The dominant, stable
  predictor across **all** specifications is **log-subscribers (β≈+0.546,
  p<0.01)** — purely **between-channel** variance, exactly what our z removes.
  Its sentiment effects (β≈0.2–0.5) are **an order of magnitude smaller** than
  the channel effect and are **never estimated intra-channel** (the channel
  enters as a single scalar, no fixed effects). Their Spearman 0.73 is a
  between-channel prediction dominated by subscribers → **not comparable** with
  our 0.28/0.07. That is: **the most on-topic paper that exists is, read
  rigorously, evidence IN FAVOR of content→performance being channel-conditional.**

**Conclusion §1 (for the README):** the ~0.28 in-dist and ~0.07 cross-channel
**are not bad** — they are the content ceiling reported by peer-reviewed
literature, measured by us under a **more honest and harder** protocol
(per-channel relative z, temporal cutoff, channel groups).


## 2 · Convergent finding #2 — why the literature's high numbers are NOT comparable

Every high published number is explained by **an easiness we deliberately
rejected**:

| Source | "High" number | Where it really comes from | Which easiness it uses that we reject |
|---|---|---|---|
| P1 | ρ=0.90–0.94 | From the **early view trajectory** (`N(tᵢ)−N(tᵢ₋₁)`, ~6 days) | We are **pre-publication**; no trajectory. The 0.23→0.93 jump *is* the size of the exogenous component |
| P3 | ρ=0.73 | From **log-subscribers (β≈+0.55)**, between-channel variance | Our z **erases per-channel μ/σ** |
| P4 | acc=88% | From **engagement features** (likes/dislikes/comments) ≈ predicting views with views; 75/25 classes; random split | No engagement leakage, relative target, temporal+group split |
| P5 | acc≈95% | From a **universal axis** real-vs-fake pre-baked into CLIP | No universal content→z axis exists (we proved it) |

**Strongest presentation artifact for the thesis (foil table).** A single table
reconciles "the literature says 0.73" with "we say 0.28/0.07": run our *same*
features predicting (i) absolute log-views with a random split → expected ~0.5–0.7
(replicates the easy regime), (ii) per-channel z in-dist → ~0.28, (iii)
per-channel z LOCO → ~0.07. The drop **is not the model's, it is honest framing**.
(Idea from P3-E5 / P4-E3; **not executed without an OK**.)

> P3, p.11: *"our findings should not be misconstrued as indicating causal
> relationships ... other factors ... may contribute to video views"* and *"we used a
> cross-sectional analysis ... future research [should] adopt a longitudinal
> approach"* — the paper itself admits the limitation our design fixes.


## 3 · Convergent finding #3 — channel-relative novelty is **triply** motivated

Our strongest finding (§08: kNN to the channel's own prior corpus → ρ=**−0.40**;
the operative signal is **novelty vs the channel norm**, not similarity) **is not
a quirk of ours**: three papers motivate it via independent routes.

- **P2 "Beyond Views"** builds an *engagement map*: non-parametric quantile
  regression that **residualizes** a nuisance covariate (duration) and predicts
  the **relative rank within group**. It is the same operation as our z (group =
  channel) — and it suggests a leakage-safe analogue: a *per-channel novelty map*
  built **only from prior videos**.
- **P1** wins with **RBF similarity to a set of representative patterns** (support
  vectors). Direct repurpose: a kernel head whose features are RBF between the new
  video and the channel's **own prior videos** → turns the −0.40 (ad-hoc kNN, with
  leakage from using K=all) into a **trained, signed, regularized** signal.
- **P3** gives the ingredient we were missing: a **cheap, channel-agnostic affect
  vector** (title sentiment via multilingual `pysentimiento` — the paper itself
  cites it, p.8 —, facial valence/arousal, caps/emoji/length). Novelty can be
  measured in **affect/style** vs the channel's historical centroid, not only in
  raw embedding.

**Leakage-safe specification (the only one that attacks the measured cause).** For
each video *v* of channel *c*: embedding (and/or affect) centroid of *c*'s videos
**published strictly before** *v*; feature = distance / signed deviation of *v*
from that centroid; the **correct sign is empirical** (more novel → better z? —
our −0.40 says yes). By construction it is **channel-relative**, so it is the only
candidate with a real chance of transferring to *"the creator changes style"* (it
is "how much you deviate from YOUR norm", not the channel's identity). The original
−0.40 used K=all (includes the future) → **with leakage**; all validity depends on
recomputing with a strictly-prior corpus, and honestly reporting if it shrinks
to ~0.


## 4 · Parametric few-shot + per-channel affine recalibration (from P5)

P5 (frozen CLIP + tiny head + 20% few-shot → ~85%) is our architecture analogue.
Read rigorously, it gives two things:

1. **A warning (it reinforces our negative result).** Their method works because
   "real vs fake" is a **universal axis** already baked into CLIP (they cite
   Cozzolino, and Gemini *zero-shot* gets ≈84–87% — the axis pre-exists). They
   never test *zero-shot* on an unseen generator; they always inject 20% of the
   target. For us the reading is **pessimistic**: their success is evidence of the
   property we **proved we do NOT have** (a universal content→z axis). Do not
   expect cross-channel generalization "for free".
2. **A concrete bet (the actionable part).** What IS per-channel and learnable
   from few samples is the **offset/scale (μ,σ) and the novelty axis**. The
   transferable move is not "recover the universal axis" but **"learn this
   channel's novelty axis from a few of its videos + per-channel affine
   recalibration"** — and *affine recalibration* is natural because **z is defined
   affinely per channel**. P5 *omits* calibration: that is exactly our
   opportunity. This is the parametric cousin of our few-shot kNN (§08) and of the
   −0.40.


## 5 · The rigor protocol the literature demands (turns the wall into a defended result)

The gradable deliverable is not breaking the wall; it is **defending it
scientifically**. The literature gives us the standard:

- **Spearman with 95% CI, and the *width* of the CI as a first-class result**
  (P1: their selling point over the baseline was a ~40% narrower CI, not just a
  higher mean). For cross-channel: report variance/CI across held-out channels.
- **Paired significance test per fold/channel** (P1, p.7: paired Student-t across
  folds, p<0.02 for ~1% deltas). Apply to V1 vs V4 vs V6 vs novelty-head → lets us
  rigorously claim *"embeddings do not significantly help cross-channel (p=...)"*.
- **AIC-style ablation ladder** (P3, specs 1→7): controls → +markers → +title
  sentiment → +facial affect → +congruence → +embeddings. Directly supports the
  Variant 1 vs 5 honesty check.
- **Performance vs number of channel references curve** (P1 Fig.3 / P2 §3.4): ρ ±
  CI vs K∈{0,1,2,5,10,all}. It is **the most publishable figure**: it visually
  separates "irreducible (K=0, zero-shot ≈0.05)" from "channel-conditional
  recoverable signal", and **bounds** the ceiling instead of pretending to beat
  it. (It also de-confounds the §08 curve by fixing the channel set.)
- **Predict-then-map + an accompanying percentile target** (P2: predicting the
  de-confounded quantity and mapping back beats predicting the interpretable one
  directly, R² 0.77 vs 0.69). For us: predict z, map to 1–10, and **report both
  metrics**; add an outlier-robust **per-channel percentile** as an accompanying
  target (peer-reviewed precedent).


## 6 · Master table — what to take and what NOT to take from each paper

| Paper | Transferable technique (actionable) | What does NOT transfer (and why) |
|---|---|---|
| **P1** Trzciński | RBF-exemplar head over the channel's prior corpus (signed, regularized novelty); CI + paired-t per fold; ρ-vs-K curve; the 0.234 ceiling-anchor table | UL/ML/MRBF/SVR over the **view trajectory** (we don't have it); mature-video selection (bias opposite to our policy) |
| **P2** Beyond Views | **Rank/percentile-within-group** normalization (robust accompanying target); predict-then-map; conditional entropy to screen tokens; the stable-vs-unstable argument that legitimizes the ceiling | Their metric (watch %) **requires API/watch-time** — impossible with `ytInitialData`; Freebase features (proprietary); per-channel CSP (tested with no lift even at their scale) |
| **P3** Clicks for money | **Channel-agnostic** affect block: `pysentimiento` (multilingual, they cite it) pos/neg/intensity + 7-class emotion; facial valence/arousal (FER); caps/emoji/?/length markers; **title↔thumbnail congruence** (their cleanest content effect, β≈1.7); ablation ladder | Their text pipeline **VADER (English only)**; negative binomial over **absolute views** with channel = 1 scalar; Spearman 0.73 dominated by subscribers |
| **P4** Nisa (XGBoost) | Only as a **foil**: shows the typical "high accuracy" = engagement leakage + random split. Reinforces tuning: raise regularization (`reg_alpha` L1, `min_child_weight`) and reduce embedding dim with small N | Its features (definition/duration/score) require **API or engagement** (leakage). LR 0.3–0.4 grid inappropriate for high-dim embeddings; **do not copy its random CV** |
| **P5** Ou (CLIP few-shot) | **Tiny parametric adapter** (1 layer + dropout, early-stop) over the channel's few videos; **per-channel affine recalibration (a·ŷ+b)**; embedding normalization/whitening (L2, centering by prior centroid, PCA 32–128) | Its success depends on a **universal axis** we don't have; "in-distribution" few-shot (20% of the same dataset) ≠ our LOCO; Conv1D over embedding dims (no inductive justification) |


## 7 · Prioritized experiment backlog (NOT executed without explicit OK)

> Standing user constraint: *"we don't move until I say so"*. This is only the
> literature-derived plan, ordered by expected value to **raise the cross-channel
> floor** or **defend the ceiling**.

1. **Leakage-safe, sign-corrected channel-relative novelty** (P1-E1 ∧ P2-E3 ∧
   P3-E4 ∧ §08). Embedding/affect distance to the channel's **strictly prior**
   centroid, as a feature in V1/V6 and as a trained RBF/kernel head. Test in-dist
   **and** LOCO **and** the pivot. *It is the only experiment that attacks the
   measured cause.* Risk: leakage if the centroid is not strictly prior.
2. **De-confound the §08 few-shot curve** with a **fixed channel set** at all K, +
   CI + paired-t per channel (P1-E2/E3, P2). Turns the confounded curve into the
   publishable ceiling figure.
3. **Channel-agnostic affect block** (multilingual `pysentimiento` + facial FER +
   SigLIP title↔thumbnail congruence) as a Variant 1 expansion (P3-A/B). Decisive:
   does it transfer cross-channel because it is channel-agnostic, or does it also
   collapse? (either way is a result).
4. **Decisive intra-channel test** (P3-C): do the sentiment effects survive
   per-channel demeaning (with prior videos only)? Refutes/contextualizes P3 inside
   our README.
5. **Few-shot parametric adapter + per-channel affine recalibration** (P5-E2/E4)
   vs the §08 kNN, with paired support and a fixed channel set.
6. **Foil table** (P3-E5/P4-E3): same features over absolute views (random split)
   vs z in-dist vs z LOCO. README rhetorical artifact; not a product.
7. **Cross-cutting rigor layer** (P1/P2/P3): CI, paired-t per channel, ablation
   ladder, predict-then-map, accompanying percentile. Post-hoc, no leakage.


## 8 · Conclusion — the thesis reframe

The literature **does not give us a technique to raise the 0.28 in-dist** (P1, P2
and P3 converge that content caps at ~0.23–0.28; the rest is exogenous: trajectory,
algorithm, subscribers, luck). What it **does** give is more valuable for a thesis:

1. **It externalizes and legitimizes the ceiling.** Three peer-reviewed papers
   (IEEE-TMM, AAAI, JBR), with more data and more powerful features, on easier
   tasks, reach the same ~0.23–0.28 of content. Our number **is not low: it is
   correct**, and our protocol is **more honest** than theirs.
2. **The most on-topic paper (P3) is, read rigorously, evidence in favor of our
   thesis** that content→performance is channel-conditional: its dominant effect
   (β≈+0.55) is exactly the variance our z erases, and they never test
   intra-channel.
3. **It keeps a single actionable lead**, now **triply motivated** by the
   literature: leakage-safe **channel-relative novelty** (plus per-channel affine
   recalibration and a channel-agnostic affect block) — the only family that, by
   construction, could survive a style change.

The wall stops being *"we failed to break it"* and becomes *"we measured a
structural ceiling, corroborated it with peer-reviewed literature, and left a
single, well-grounded actionable front"*. That **is** the deliverable.


## References

- **P1** — T. Trzciński, P. Rokita. *Predicting Popularity of Online Videos Using
  Support Vector Regression.* IEEE Transactions on Multimedia, 2017.
  arXiv:1510.06223v4. (`papers/1510.06223v4.pdf`)
- **P2** — S. Wu, M.-A. Rizoiu, L. Xie. *Beyond Views: Measuring and Predicting
  Engagement in Online Videos.* AAAI / ICWSM 2018. arXiv:1709.02541v4.
  (`papers/1709.02541v4.pdf`)
- **P3** — Y. Cui, J. Chung, X. Peng, X. Wang. *Clicks for money: Predicting video
  views through a sentiment analysis of titles and thumbnails.* Journal of Business
  Research 183 (2024) 114849. (`papers/1-s2.0-S0148296324003539-main.pdf`)
- **P4** — N. Nisa et al. *Optimizing Prediction of YouTube Video Popularity Using
  XGBoost.* Electronics 10(23):2962, 2021. (`papers/electronics-10-02962-v2.pdf`)
- **P5** — Z. Ou. *CLIP Embeddings for AI-Generated Image Detection: A Few-Shot
  Study with Lightweight Classifier.* 2025. arXiv:2505.10664v1.
  (`papers/2505.10664v1.pdf`)
