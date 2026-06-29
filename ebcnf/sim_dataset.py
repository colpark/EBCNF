"""Simulated EBC dataset for amortized LAINR training (on-the-fly, fully seeded).

Each scene: a broadband texture (or grating) translating with random parameters, from
which we derive the log-intensity field L (T,H,W), the per-step signed event field
E (T-1,H,W) in {-1,0,+1}, the contrast threshold thr, and optional RGB anchor frames.
Scenes are generated deterministically from a seed, so "amount of data" = number of
scenes; train/val use disjoint seed ranges (no leakage).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

SPLIT_OFFSET = {"train": 0, "val": 5_000_000}


@dataclass
class EBCSimConfig:
    H: int = 32
    W: int = 32
    T: int = 32
    n_tex: int = 5            # texture components (1 => near-grating, easy)
    contrast: float = 1.2     # event threshold in units of dL std
    noise: float = 0.02       # fraction of stochastic (hot-pixel) events
    n_anchors: int = 1        # RGB anchor frames (evenly spaced); 0 => event-only


def _scene(cfg: EBCSimConfig, seed: int):
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(0, 1, cfg.H), np.linspace(0, 1, cfg.W), indexing="ij")
    freqs = rng.uniform(2.0, 9.0, (cfg.n_tex, 2))
    amps = rng.uniform(0.5, 1.0, cfg.n_tex)
    phase = rng.uniform(0, 2 * np.pi, cfg.n_tex)
    vx, vy = rng.uniform(-0.5, 0.5, 2)
    ts = np.linspace(0, 1, cfg.T)[:, None, None]
    L = np.zeros((cfg.T, cfg.H, cfg.W))
    for k in range(cfg.n_tex):
        L += amps[k] * np.sin(2 * np.pi * (freqs[k, 0] * (xx[None] - vx * ts) +
                                           freqs[k, 1] * (yy[None] - vy * ts) + phase[k]))
    L = (L - L.mean()) / (L.std() + 1e-8)
    dL = np.diff(L, axis=0)
    thr = float(cfg.contrast * dL.std())
    E = np.zeros_like(dL)
    E[dL > thr] = 1.0
    E[dL < -thr] = -1.0
    if cfg.noise > 0:
        flip = rng.random(E.shape) < cfg.noise
        E[flip] = rng.choice([-1.0, 1.0], int(flip.sum()))
    return L.astype(np.float32), E.astype(np.float32), thr


class EBCSimDataset(Dataset):
    def __init__(self, cfg: EBCSimConfig, split: str, n: int, base_seed: int = 0):
        self.cfg, self.split, self.n, self.base_seed = cfg, split, n, base_seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        seed = self.base_seed * 100_000_000 + SPLIT_OFFSET[self.split] + i
        L, E, thr = _scene(self.cfg, seed)
        # RGB anchors: evenly-spaced frames of L (log-intensity stand-in)
        if self.cfg.n_anchors > 0:
            idx = np.linspace(0, self.cfg.T - 1, self.cfg.n_anchors).round().astype(int)
        else:
            idx = np.array([], dtype=int)
        return {
            "E": torch.from_numpy(E),                 # (T-1,H,W) signed events
            "L": torch.from_numpy(L),                 # (T,H,W) ground-truth log-intensity
            "thr": torch.tensor(thr),
            "anchor_idx": torch.from_numpy(idx.astype(np.int64)),
            "anchor_val": torch.from_numpy(L[idx]) if len(idx) else torch.zeros(0, self.cfg.H, self.cfg.W),
        }


def make_loaders(cfg: EBCSimConfig, n_train: int, n_val: int, batch_size: int,
                 base_seed: int = 0):
    def collate(b):
        out = {}
        for k in ("E", "L", "thr"):
            out[k] = torch.stack([x[k] for x in b])
        out["anchor_idx"] = torch.stack([x["anchor_idx"] for x in b])   # (B, n_anchors)
        out["anchor_val"] = torch.stack([x["anchor_val"] for x in b])   # (B, n_anchors,H,W)
        return out
    tr = EBCSimDataset(cfg, "train", n_train, base_seed)
    va = EBCSimDataset(cfg, "val", n_val, base_seed)
    return (DataLoader(tr, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True),
            DataLoader(va, batch_size=batch_size, shuffle=False, collate_fn=collate))
