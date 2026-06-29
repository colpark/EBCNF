"""Amortized LAINR-EBC training: encode each scene's events -> latent tokens (one
forward pass), decode a continuous log-intensity field, supervised by the EvINR
event-generation model (analytic temporal derivative) + spatial reg + optional RGB
anchors. Trained across many simulated scenes (amortized / generalizable).

Run:
  uv run python -m src.ebc.train_amortized --smoke          # tiny CPU sanity
  uv run python -m src.ebc.train_amortized --config configs/ebc_lainr.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .utils.device import auto_device
from .utils.seeding import set_seed
from .sim_dataset import EBCSimConfig, make_loaders
from .lainr_ebc import LAINR_EBC

_REPO = Path(__file__).resolve().parents[1]


def _sample_coords(E, thr, n, T, device, gen):
    """Sample n query coords per scene + EvINR derivative targets. Returns
    coords (B,n,3) [requires_grad], targets (B,n)."""
    B, Tm1, H, W = E.shape
    dtn = 2.0 / (T - 1)
    k = torch.randint(Tm1, (B, n), generator=gen)
    yi = torch.randint(H, (B, n), generator=gen)
    xj = torch.randint(W, (B, n), generator=gen)
    xn = 2 * xj.float() / (W - 1) - 1
    yn = 2 * yi.float() / (H - 1) - 1
    tn = 2 * (k.float() + 0.5) / (T - 1) - 1
    coords = torch.stack([xn, yn, tn], -1).to(device).requires_grad_(True)
    bidx = torch.arange(B).unsqueeze(1).expand(B, n)
    target = (thr.view(B, 1).to(device) * E[bidx, k, yi, xj].to(device)) / dtn
    return coords, target


def _anchor_loss(model, tokens, anchors, batch, n_pix, T, device, gen):
    idx = batch["anchor_idx"]                                # (B,A)
    val = batch["anchor_val"]                                # (B,A,H,W)
    B, A = idx.shape
    if A == 0:
        return torch.zeros((), device=device)
    H, W = val.shape[-2:]
    yi = torch.randint(H, (B, A, n_pix), generator=gen)
    xj = torch.randint(W, (B, A, n_pix), generator=gen)
    tn = (2 * idx.float() / (T - 1) - 1).unsqueeze(-1).expand(B, A, n_pix)
    xn = 2 * xj.float() / (W - 1) - 1
    yn = 2 * yi.float() / (H - 1) - 1
    coords = torch.stack([xn, yn, tn], -1).reshape(B, A * n_pix, 3).to(device)
    pred = model.field(coords, tokens, anchors)             # (B, A*n_pix)
    bi = torch.arange(B).view(B, 1, 1).expand(B, A, n_pix)
    ai = torch.arange(A).view(1, A, 1).expand(B, A, n_pix)
    tgt = val.to(device)[bi, ai, yi, xj].reshape(B, A * n_pix)
    return torch.mean((pred - tgt) ** 2)


@torch.no_grad()
def _ac_corr(Lpred, Ltrue):
    Pa = Lpred - Lpred.mean(0, keepdims=True)
    Ta = Ltrue - Ltrue.mean(0, keepdims=True)
    a, b = Pa.reshape(Pa.shape[0], -1), Ta.reshape(Ta.shape[0], -1)
    cs = []
    for i in range(a.shape[0]):
        if a[i].std() > 1e-6 and b[i].std() > 1e-6:
            cs.append(torch.corrcoef(torch.stack([a[i], b[i]]))[0, 1].item())
    return float(np.mean(cs)) if cs else float("nan")


def evaluate(model, loader, T, device, max_batches=1):
    model.eval()
    sign_accs, ac_corrs = [], []
    H = W = None
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        E = batch["E"]; B, Tm1, H, W = E.shape
        tokens, anchors = model.encode(E.unsqueeze(1).to(device))
        # event sign-accuracy on a sample of event voxels
        coords, target = _sample_coords(E, batch["thr"], 2048, T, device,
                                        torch.Generator().manual_seed(123))
        Fv = model.field(coords, tokens, anchors)
        dF = torch.autograd.grad(Fv.sum(), coords)[0][..., 2]
        ev = target != 0
        sign_accs.append((torch.sign(dF[ev]) == torch.sign(target[ev])).float().mean().item())
        # AC reconstruction corr on a dense grid (per scene)
        xs = torch.linspace(-1, 1, W); ys = torch.linspace(-1, 1, H); tsn = torch.linspace(-1, 1, T)
        ti, yj, xi = torch.meshgrid(tsn, ys, xs, indexing="ij")
        grid = torch.stack([xi, yj, ti], -1).reshape(1, -1, 3).expand(B, -1, 3).to(device)
        with torch.no_grad():
            Lp = model.field(grid, tokens, anchors).reshape(B, T, H, W).cpu()
        ac_corrs.append(_ac_corr(Lp, batch["L"]))
    return float(np.mean(sign_accs)), float(np.mean(ac_corrs))


def train(cfg: EBCSimConfig, n_train, n_val, batch_size, steps, lr, lam, n_query,
          D, hidden, sigma_xy, sigma_t, device, base_seed, out_dir, log_every):
    set_seed(base_seed)
    dev = auto_device(device)
    tr, va = make_loaders(cfg, n_train, n_val, batch_size, base_seed)
    model = LAINR_EBC(D=D, hidden=hidden, sigma_xy=sigma_xy, sigma_t=sigma_t).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(base_seed)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"amortized LAINR-EBC | device={dev} | params={n_params:,} | "
          f"scenes train/val={n_train}/{n_val} | grid {cfg.T}x{cfg.H}x{cfg.W} | "
          f"anchors={cfg.n_anchors}")
    warm = max(1, steps // 10)
    it = 0
    model.train()
    while it < steps:
        for batch in tr:
            if it >= steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr * min(1.0, (it + 1) / warm)
            E = batch["E"]
            tokens, anchors = model.encode(E.unsqueeze(1).to(dev))
            coords, target = _sample_coords(E, batch["thr"], n_query, cfg.T, dev, gen)
            Fv = model.field(coords, tokens, anchors)
            grads = torch.autograd.grad(Fv.sum(), coords, create_graph=True)[0]
            loss_t = torch.mean((grads[..., 2] - target) ** 2)
            loss_s = torch.mean(grads[..., 0] ** 2 + grads[..., 1] ** 2)
            loss = loss_t + lam * loss_s
            if cfg.n_anchors > 0:
                loss = loss + _anchor_loss(model, tokens, anchors, batch, 64, cfg.T, dev, gen)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            it += 1
            if it % log_every == 0 or it == steps:
                print(f"  step {it}/{steps}  loss={loss.item():.4f} "
                      f"(temp={loss_t.item():.4f} spat={loss_s.item():.4f})")
    sign_acc, ac_corr = evaluate(model, va, cfg.T, dev)
    print(f"\nVAL (held-out scenes): event sign-acc={sign_acc:.3f}  AC intensity corr={ac_corr:.3f}")
    out = {"device": dev, "n_params": n_params, "steps": steps, "n_train": n_train,
           "val_sign_acc": sign_acc, "val_ac_corr": ac_corr, "anchors": cfg.n_anchors}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ebc_lainr_results.json").write_text(json.dumps(out, indent=2))
    torch.save(model.state_dict(), out_dir / "ebc_lainr.pt")
    print(f"wrote {out_dir/'ebc_lainr_results.json'} and checkpoint")
    return out


def _load_yaml(path):
    import yaml
    return yaml.safe_load(Path(path).read_text()) if path else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    c = _load_yaml(a.config)

    if a.smoke:
        cfg = EBCSimConfig(H=16, W=16, T=16, n_tex=4, n_anchors=1)
        params = dict(n_train=16, n_val=8, batch_size=4, steps=20, lr=5e-4, lam=0.02,
                      n_query=512, D=32, hidden=64, sigma_xy=4.0, sigma_t=8.0,
                      base_seed=0, log_every=10)
    else:
        cfg = EBCSimConfig(H=c.get("H", 32), W=c.get("W", 32), T=c.get("T", 32),
                           n_tex=c.get("n_tex", 5), contrast=c.get("contrast", 1.2),
                           noise=c.get("noise", 0.02), n_anchors=c.get("n_anchors", 1))
        params = dict(n_train=c.get("n_train", 1024), n_val=c.get("n_val", 128),
                      batch_size=c.get("batch_size", 8), steps=c.get("steps", 4000),
                      lr=c.get("lr", 5e-4), lam=c.get("lam", 0.02),
                      n_query=c.get("n_query", 1024), D=c.get("D", 64),
                      hidden=c.get("hidden", 128), sigma_xy=c.get("sigma_xy", 4.0),
                      sigma_t=c.get("sigma_t", 8.0), base_seed=c.get("seed", 0),
                      log_every=c.get("log_every", 200))
    train(cfg, device=a.device, out_dir=_REPO / "reports" / "ebc", **params)


if __name__ == "__main__":
    main()
