# Quick review 2026-11-23-2019 - skeptical peer reviewer (Codex GPT-6 Luna high)

## Observations

- The ledger tracks 142 claims, with 75 UNVERIFIED and many scope fields still TBD; the volume of results makes clear claim boundaries essential. [experiments/claims-status.md]
- C209 reports a 16.67% clean80 gain for A2s over E27, but A2s has more unique parameters; SQ419 is the running control intended to separate recipe effects from that confound. [C209; DEVLOG.md, v0.5.54]
- The queue puts Q15, a redesign of the invalid K3 attention control, next for CPU work; no GPU task is pickable while SQ419 runs. [experiments/QUEUE.md]

## Ideas

- **[speculation] Check whether C209’s gain is spread across packages.** Hypothesis: a few packages may drive the document-level result despite the existing seed and document bootstrap. Cheapest test: about 15 CPU minutes at 13M scale, using existing per-document scores to compute a package-grouped bootstrap interval. If the interval includes zero or changes the estimated gain by more than 4 percentage points, qualify C209 and require broader package coverage in the next comparison.

## Question

- Does the package-grouped interval exclude zero and remain close to the reported seed-level estimate?

## Triage
- Idea (is C209's gain spread across packages?): C209 is a clean80 result, and clean80 documents have no package structure (they are v1 val docs). Package grouping exists only for the SQ99 out-of-corpus set, where SQ400 already reports it for this pair. A2s vs E27 out of corpus is -11.26%, with paired-by-seed CI [-11.59, -10.93] and package-clustered bootstrap over 31 packages [-12.02, -10.56]. Both exclude 0 and agree within 1 point, so no few packages drive it. The in-corpus analogue of package grouping is source config, which is the SQ239 extension (C209 by source config, rule: one source reversing or a 2x spread qualifies C209). Not added again.
- Question (does the package-grouped interval exclude zero and stay close to the seed estimate?): yes, out of corpus: [-12.02, -10.56] vs the seed-level [-11.59, -10.93], both around -11.3%. In corpus, the per-source breakdown arrives with SQ239's write-up.
- Observations: agreed.
