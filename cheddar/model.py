"""CHEDDAR: heterogeneous atom graph with two edge relations and sum|mean|max readout.

Bond edges and 3D contact edges get separate encoders and message functions, so a
pair that is close in space but distant in bond topology -- the transannular
hydrogen-bond signature -- is expressible. Section 3.2, Eqs. (7)-(13).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RelMPLayer(nn.Module):
    """One message-passing layer over two relations, each with its own message net."""

    def __init__(self, dim, bond_dim, space_dim, dropout=0.1):
        super().__init__()
        mk = lambda ed: nn.Sequential(
            nn.Linear(2 * dim + ed, dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim))
        self.msg_bond = mk(bond_dim)
        self.msg_space = mk(space_dim)
        self.upd = nn.Sequential(nn.Linear(3 * dim, dim), nn.SiLU())
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def _agg(h, idx, m):
        """Mean aggregation over incoming edges; an empty neighbour set gives zero."""
        dst = idx[1]
        out = torch.zeros_like(h).index_add_(0, dst, m)
        cnt = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype).index_add_(
            0, dst, torch.ones_like(dst, dtype=h.dtype)).clamp(min=1)
        return out / cnt.unsqueeze(-1)

    def forward(self, h, bi, ba, si, sa):
        ab = self._agg(h, bi, self.msg_bond(
            torch.cat([h[bi[0]], h[bi[1]], ba], dim=-1))) if bi.shape[1] else torch.zeros_like(h)
        as_ = self._agg(h, si, self.msg_space(
            torch.cat([h[si[0]], h[si[1]], sa], dim=-1))) if si.shape[1] else torch.zeros_like(h)
        return self.norm(h + self.upd(torch.cat([h, ab, as_], dim=-1)))


class MechGraphNet(nn.Module):
    """Eq. (8) encoders, Eq. (9)-(10) message passing, Eq. (11)-(12) readout and bounded head."""

    # Eq. (12) bounds: the normalized target scale of Eq. (15).
    Y_LO, Y_HI = -1.0, 1.0
    # Eq. (11): the three poolings are concatenated and the head weights them.
    READOUT = ("sum", "mean", "max")

    def __init__(self, in_node, in_bond, in_space, hidden=128, layers=3,
                 dropout=0.5, head_hidden=128, trans_drop=0.2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_node, hidden), nn.SiLU())
        self.benc = nn.Sequential(nn.Linear(in_bond, hidden // 2), nn.SiLU())
        self.senc = nn.Sequential(nn.Linear(in_space, hidden // 2), nn.SiLU())
        self.mp = nn.ModuleList([RelMPLayer(hidden, hidden // 2, hidden // 2, dropout)
                                 for _ in range(layers)])
        # Occupancy-stratified edge dropout, Eq. (7): during training, drop each
        # occupancy-selected contact edge that the mean-distance criterion would
        # NOT have selected, with probability trans_drop.
        self.trans_drop = float(trans_drop)
        self.readout = list(self.READOUT)
        self.head = nn.Sequential(
            nn.Linear(len(self.readout) * hidden, head_hidden), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(head_hidden, 1))

    def _pool(self, h, b, n):
        """Eq. (11): concatenate sum, mean and max over the atoms of each graph."""
        d = h.shape[-1]
        parts = []
        s = torch.zeros(n, d, device=h.device, dtype=h.dtype).index_add_(0, b, h)
        c = torch.zeros(n, device=h.device, dtype=h.dtype).index_add_(
            0, b, torch.ones_like(b, dtype=h.dtype)).clamp(min=1)
        for r in self.readout:
            if r == "sum":
                parts.append(s)
            elif r == "mean":
                parts.append(s / c.unsqueeze(-1))
            elif r == "max":
                # scatter_reduce, not index_reduce: index_reduce has no deterministic
                # CUDA kernel, and max is order-independent so the swap is exact.
                mx = torch.full((n, d), float("-inf"), device=h.device, dtype=h.dtype)
                mx = mx.scatter_reduce(0, b.unsqueeze(1).expand(-1, d), h,
                                       reduce="amax", include_self=True)
                parts.append(torch.where(torch.isinf(mx), torch.zeros_like(mx), mx))
            else:
                raise ValueError(r)
        return torch.cat(parts, dim=-1)

    def forward(self, batch):
        h = self.enc(batch["x"])
        ba = self.benc(batch["bond_attr"])
        sa = self.senc(batch["space_attr"])
        b, n = batch["batch"], batch["n_graphs"]
        si, sa_ = batch["space_index"], sa
        if (self.training and self.trans_drop > 0
                and batch.get("trans_mask") is not None and si.shape[1]):
            tm = batch["trans_mask"].bool()
            keep = ~tm | (torch.rand(tm.shape[0], device=tm.device) >= self.trans_drop)
            si = si[:, keep]
            sa_ = sa_[keep]
        for layer in self.mp:
            h = layer(h, batch["bond_index"], ba, si, sa_)
        out = self.head(self._pool(h, b, n)).squeeze(-1)
        return self.Y_LO + (self.Y_HI - self.Y_LO) * torch.sigmoid(out)   # Eq. (12)
