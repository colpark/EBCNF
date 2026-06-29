"""Amortized LAINR field for EBC: 3-D event encoder -> latent tokens; locality-aware
multi-band decoder F_theta(x,y,t) -> log-intensity (differentiable for EvINR loss).

Coordinate convention everywhere: (x, y, t), normalized to [-1, 1].
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EventEncoder3D(nn.Module):
    """Event voxel volume (B,1,T-1,H,W) -> latent tokens (B,M,D) + anchors (M,3)."""

    def __init__(self, D=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, D // 2, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv3d(D // 2, D, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv3d(D, D, 3, stride=1, padding=1), nn.GELU(),
        )
        self.D = D
        self._anchors = None  # cached (M,3) for a given token grid shape

    def forward(self, V):                                   # V: (B,1,T,H,W)
        f = self.net(V)                                     # (B,D,Tt,Hh,Ww)
        B, D, Tt, Hh, Ww = f.shape
        tokens = f.permute(0, 2, 3, 4, 1).reshape(B, Tt * Hh * Ww, D)   # (B,M,D)
        if self._anchors is None or self._anchors.shape[0] != Tt * Hh * Ww:
            tc = torch.linspace(-1, 1, Tt); yc = torch.linspace(-1, 1, Hh); xc = torch.linspace(-1, 1, Ww)
            ti, yi, xi = torch.meshgrid(tc, yc, xc, indexing="ij")
            self._anchors = torch.stack([xi, yi, ti], -1).reshape(-1, 3)  # (M,3) (x,y,t)
        return tokens, self._anchors.to(V.device)


def _gff(coords, B):                                        # coords (...,3), B (3,F)
    ang = 2 * np.pi * coords @ B                            # (...,F)
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class LocalityMultiBandDecoder(nn.Module):
    """Query (x,y,t) -> locality-aware cross-attention to tokens -> multi-band field."""

    def __init__(self, D=64, hidden=128, n_bands=4, feats_per_band=16,
                 q_freq=16, sigma_xy=4.0, sigma_t=8.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        # query Fourier embedding (anisotropic: higher bandwidth in t)
        Bq = torch.randn(3, q_freq, generator=g)
        Bq[:2] *= sigma_xy; Bq[2] *= sigma_t
        self.register_buffer("Bq", Bq)
        self.q_proj = nn.Linear(2 * q_freq, hidden)
        self.k_proj = nn.Linear(D, hidden)
        self.v_proj = nn.Linear(D, hidden)
        self.scale = hidden ** -0.5
        self.log_loc = nn.Parameter(torch.tensor(1.5))
        # multi-band field features (increasing bandwidth), anisotropic in t
        self.band_B = nn.ParameterList()
        for b in range(n_bands):
            base = 2.0 ** b
            Bb = torch.randn(3, feats_per_band, generator=g)
            Bb[:2] *= sigma_xy * base; Bb[2] *= sigma_t * base
            self.band_B.append(nn.Parameter(Bb, requires_grad=False))
        self.band_in = nn.ModuleList(nn.Linear(2 * feats_per_band, hidden) for _ in range(n_bands))
        self.band_film = nn.ModuleList(nn.Linear(hidden, 2 * hidden) for _ in range(n_bands))
        self.act = nn.GELU()
        self.out = nn.Linear(hidden, 1)
        self.n_bands = n_bands

    def forward(self, coords, tokens, anchors):             # coords (B,N,3), tokens (B,M,D)
        q = self.q_proj(_gff(coords, self.Bq))              # (B,N,hidden)
        K = self.k_proj(tokens); Vv = self.v_proj(tokens)   # (B,M,hidden)
        logits = torch.bmm(q, K.transpose(1, 2)) * self.scale            # (B,N,M)
        d2 = ((coords.unsqueeze(2) - anchors.view(1, 1, -1, 3)) ** 2).sum(-1)  # (B,N,M)
        logits = logits - F.softplus(self.log_loc) * d2
        attn = torch.softmax(logits, dim=-1)
        m = torch.bmm(attn, Vv)                              # (B,N,hidden) modulation
        h = torch.zeros_like(m)
        for b in range(self.n_bands):
            hb = self.band_in[b](_gff(coords, self.band_B[b]))
            s, sh = self.band_film[b](m).chunk(2, dim=-1)
            h = h + self.act((1.0 + s) * hb + sh)
        return self.out(h).squeeze(-1)                       # (B,N)


class LAINR_EBC(nn.Module):
    def __init__(self, D=64, hidden=128, n_bands=4, sigma_xy=4.0, sigma_t=8.0):
        super().__init__()
        self.encoder = EventEncoder3D(D=D)
        self.decoder = LocalityMultiBandDecoder(D=D, hidden=hidden, n_bands=n_bands,
                                                sigma_xy=sigma_xy, sigma_t=sigma_t)

    def encode(self, V):
        return self.encoder(V)                               # tokens (B,M,D), anchors (M,3)

    def field(self, coords, tokens, anchors):
        return self.decoder(coords, tokens, anchors)         # (B,N)
