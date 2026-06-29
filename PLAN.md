# Plan — Amortized LAINR for Event-Camera (EBC) Brightness Fields

Bring together three threads: **LAINR** (amortized, locality-aware, multi-band INR),
the **EvINR event-generation model** (supervise the field's analytic temporal
derivative, not the ±1 events), and our **fusion findings** (RGB anchors fix the DC
constant of integration; amortization trades per-scene fidelity for cross-scene
generalization).

## Objective
Learn one **amortized** field `F_θ(x,y,t) → log-intensity` whose latent is **predicted
in a single forward pass** by an encoder from a scene's event stream (+ optional RGB
anchor), and that generalizes across scenes — so reconstructing a *new* scene is one
forward pass, not a per-scene optimization (the EvINR/Ev-NeRF regime).

## Why LAINR specifically here
- **Smooth latent → sharp output**: the multi-band coarse-to-fine Fourier decoder
  represents sharp moving edges from a smoothly-varying modulation (confirmed: that's
  how INRs reconstruct image edges).
- **Locality-aware decoder**: a query coordinate attends to *nearby* latent tokens, so
  fast local dynamics get a local modulation — the right prior for events.
- **Amortization** supplies a *learned data prior*, which is exactly what resolves the
  underdetermined held-out-time gap that sank the per-scene plain INR.

## Architecture
1. **Encoder (amortizes field-fitting).** Input = event voxel volume `E` (per-step
   signed polarity, shape (T−1, H, W)) [+ optional RGB anchor frames]. A small **3-D
   CNN** downsamples (t,x,y) into a grid of **latent tokens** `z ∈ R^{M×D}` with fixed
   coordinate **anchors** `a ∈ [−1,1]^{M×3}` (token centers). One pass → tokens.
2. **Decoder (LAINR locality-aware multi-band).** For query `(x,y,t)`:
   cross-attention from a Fourier embedding of the coord to the tokens, with a
   **locality bias** `−α·‖coord − anchor‖²`, gives a modulation `m`. Then a
   **multi-band coarse-to-fine** Fourier field (bands of increasing bandwidth, FiLM by
   `m`) outputs log-intensity. Differentiable → **analytic ∂_t, ∂_x, ∂_y** via autograd.
3. (Anisotropy note) Fourier bandwidth should be **higher in t** than (x,y) — events
   are µs-fast; start isotropic, add t-anisotropy as a knob.

## Losses (per scene, batched across scenes = amortized)
- **EvINR temporal (event) term**: `(∂_t F·Δt − C·E)²` at sampled `(x,y,t)`.
- **Spatial regularization**: `λ(∂_x F)² + (∂_y F)²` (denoise + natural-image prior).
- **RGB-anchor term (optional)**: `‖F(x,y,t_k) − log RGB_k‖²` to fix the per-pixel DC
  constant; sweep #anchors {0,1,few,dense} to quantify each modality's marginal value.
- Robust (Huber) option on the event term for stochastic sensor noise.

## Simulated data ("a reasonable amount")
On-the-fly, fully seeded (no disk needed; reproducible by seed):
- Scenes = broadband **textured fields translating** with random velocity/frequencies/
  contrast/noise (+ a fraction of clean gratings for an easy slice).
- Default scale: **1024 train / 128 val** scenes, grid **32×32×32**, ~1e3 events/scene.
- Disjoint seed ranges for train/val (no leakage), matching the main harness convention.

## Training protocol
- Adam, LR warmup + grad clip (reuse harness), GPU auto-detect. Batch of scenes →
  encode → sample query coords per scene → derivative loss + reg (+ anchor).
- Budget: amortized, so train to convergence over the dataset (this is the GPU job).
- Determinism seeded; configs + git SHA written per run.

## Evaluation (on held-out scenes)
- **event polarity sign-accuracy** (machinery), **AC intensity corr** (event-determined
  relative reconstruction), **absolute L corr** (with anchors).
- **Held-out-time-gap recovery** — the decisive test: does the amortized prior let the
  field predict events/intensity at unseen times where the per-scene INR collapsed?
- **Amortization gap**: amortized (1 forward pass) vs a few steps of per-scene
  finetuning of the latent — quantify the fidelity cost of amortization.

## Milestones
1. ✅ Test 1: per-scene EvINR machinery works on a simple scene (sign-acc 1.0, corr 0.996).
2. **This push**: amortized LAINR-EBC pipeline (dataset + model + train) that runs
   end-to-end (smoke) and is ready for the GPU run.
3. GPU run: train on 1024 scenes; report held-out sign-acc / AC corr / gap recovery.
4. RGB-anchor sweep (absolute vs relative; modality marginal value).
5. Amortized-vs-finetuned (amortization gap); anisotropic-t ablation.

## Risks / honest caveats
- **Derivative-only supervision underdetermines absolute level** → need ≥1 anchor for
  absolute (event-only stays relative).
- **Amortization blur**: encoder-predicted latents underfit sharp detail vs per-scene
  optimization — the gap we measure, not assume.
- **Magnitude calibration**: spatial-reg attenuates derivative magnitude (Test 1 showed
  sign perfect, magnitude low) → event *re-detection* needs threshold recalibration.
- **Sim-to-real**: simulated 1/fᵝ-style scenes are an abstraction; real sensor noise,
  hot pixels, refractory effects are not modeled here.

## How to run (after `uv sync`)
```
uv run make ebc-train-smoke          # tiny end-to-end sanity (CPU, seconds)
uv run make ebc-train                # reasonable dataset (auto-detects GPU)
# or: uv run python -m src.ebc.train_amortized --config configs/ebc_lainr.yaml
```
