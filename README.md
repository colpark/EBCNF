# EBCNF — Neural Fields for Event-Based Cameras

Neural-field reconstruction of event-camera (EBC) data, built around two ideas:

1. **Don't fit the ±1 events directly — fit the brightness field and constrain its
   slope.** Event-camera output is a sparse, ternary {−1,0,+1} spatiotemporal field
   (events only at moving edges). A vanilla coordinate INR regressing that field
   memorizes seen frames but collapses at unseen times, and is capped by stochastic
   noise. Instead we follow the **event-generation model** (à la EvINR, ECCV 2024):
   fit `F_θ(x,y,t) → log-intensity` and supervise its **analytic temporal derivative**
   with events: `∂_t F·Δt ≈ C·E`, plus a spatial-gradient regularizer.

2. **Amortized LAINR.** A 3-D encoder predicts locality-aware latent tokens from a
   scene's events in one forward pass; a multi-band, locality-biased decoder
   reconstructs the continuous field (smooth latent → sharp output). Trained across
   many simulated scenes so a new scene is one forward pass, not a per-scene fit.
   RGB anchor frames fix the per-pixel DC constant (events alone are relative-only).

See `PLAN.md` for the full design, milestones, and honest caveats.

## Contents
```
ebcnf/
  event_inr.py       diagnosis: vanilla FF-INR on the raw ±1 field (why it's hard)
  evinr.py           Test 1: EvINR-style event->intensity (SIREN + derivative loss)
  sim_dataset.py     simulated EBC scenes (on-the-fly, seeded; train/val disjoint)
  lainr_ebc.py       amortized LAINR field (3-D encoder + locality multi-band decoder)
  train_amortized.py amortized training (EvINR loss + spatial reg + RGB anchors)
  utils/             seeding, device auto-detect
configs/ebc_lainr.yaml   "reasonable amount" training config
PLAN.md, reports/        plan + run outputs/figures
```

## Setup & run (uv)
```bash
uv sync
uv run make ffinr         # the difficulty diagnosis
uv run make test1         # EvINR reconstruction on a simple scene
uv run make train-smoke   # amortized LAINR-EBC tiny sanity (CPU, seconds)
uv run make train         # amortized training on simulated data (auto-detects GPU)
```
`make train` reads `configs/ebc_lainr.yaml` (1024 train / 128 val scenes, 32³ grid).
Override on the remote GPU as needed (e.g. larger `n_train`/`steps`).

## Status
- Vanilla FF-INR diagnosis: reproduces the memorize-vs-interpolate failure.
- Test 1 (per-scene EvINR): event sign-accuracy 1.000, intensity corr ~0.996 on a
  simple fully-dynamic scene — the derivative-supervision machinery works.
- Amortized LAINR-EBC: pipeline runs end-to-end; full GPU training + held-out
  evaluation (sign-acc / AC corr / temporal-gap recovery / RGB-anchor sweep) is next.
