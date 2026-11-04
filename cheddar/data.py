"""Dataset and collate for the CHEDDAR ensemble cache (Section 3.1).

One cache per solvent. The two solvent graphs share atoms and covalent bonds and
differ in ensemble geometry and 3D contact edges; `alt_cache` supplies the second
solvent and `solvent_token` appends iota_s (0 water, 1 hexane) to every atom.

Widths come from the cache's stored column lists, never from a constant, so the
model is built at whatever width the features have (Appendix A).

Standardization (Appendix E) is fitted on TRAIN ids only, separately for the node
block and each edge relation; zero-variance columns get sd 1. One-hot columns are
standardized too, which is harmless for a linear encoder.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import read_labels


def ensure_cache(cache_path):
    """Build a missing solvent cache in place (Sec. 3.2).

    The ensemble features are statistics over MD frames, so this needs the
    trajectories. Point CHEDDAR_TRAJ, CHEDDAR_ANNOT and CHEDDAR_CSV at them; the
    label CSV alone carries SMILES and PAMPA and cannot substitute.

    data/nal_fix.json beside the label CSV is picked up automatically. It holds
    the 95 naphthylalanine SMILES that the released table writes as 2-Nal while
    the deposited MD geometry is 1-Nal; without it those 95 molecules get atom
    features that do not match their own trajectory. CHEDDAR_SMILES_FIX
    overrides the location.
    """
    from .features import build_cache
    traj = os.environ.get("CHEDDAR_TRAJ")
    annot = os.environ.get("CHEDDAR_ANNOT")
    csv_path = os.environ.get("CHEDDAR_CSV")
    missing = [k for k, v in (("CHEDDAR_TRAJ", traj), ("CHEDDAR_ANNOT", annot),
                              ("CHEDDAR_CSV", csv_path)) if not v]
    if missing:
        raise FileNotFoundError(
            "%s is missing and cannot be built: set %s.\n"
            "The cache holds per-atom statistics over 100 MD frames (Sec. 3.2); "
            "it is built from the trajectories, not from the label table."
            % (cache_path, ", ".join(missing)))
    fix = os.environ.get("CHEDDAR_SMILES_FIX") or os.path.join(
        os.path.dirname(os.path.abspath(csv_path)), "nal_fix.json")
    if not os.path.exists(fix):
        raise FileNotFoundError(
            "the SMILES correction %s is missing. It ships in data/ and is needed "
            "for the 95 naphthylalanine peptides; set CHEDDAR_SMILES_FIX to point "
            "at it, or pass smiles_fix=None to build_cache to skip it knowingly."
            % fix)
    solvent = "Hexane" if "Hexane" in os.path.basename(cache_path) else "Water"
    build_cache(cache_path, annot, traj, csv_path, solvent=solvent, smiles_fix=fix)

class MechData:
    def __init__(self, cache_path, label_csv, label_col="PAMPA",
                 edge_block="both", solvent_token=True, alt_cache=None,
                 meanonly=False, drop_space_cols=()):
        if not os.path.exists(cache_path) or os.path.getsize(cache_path) == 0:
            ensure_cache(cache_path)
        with open(cache_path, "rb") as f:
            blob = pickle.load(f)
        self.meta, self.feat = blob["meta"], blob["feat"]
        self.labels = read_labels(label_csv, label_col)
        # Eq. (15): y_norm = 2*(clip(y,-8,-4)+8)/4 - 1, i.e. clip((y+6)/2, -1, 1).
        # Reversed before scoring by denorm() in train.py, Eq. (16).
        self.labels = {k: float(np.clip((v + 6.0) / 2.0, -1.0, 1.0))
                       for k, v in self.labels.items()}

        cols = self.meta["node_cols"]
        bc = self.meta["bond_cols"]
        sc = self.meta["space_cols"]
        self._nidx = list(range(len(cols)))
        self._bkeep = list(range(len(bc)))
        self._sidx = list(range(len(sc)))
        # meanonly removes the dispersion columns, leaving what a single mean
        # structure could supply. Used by the "w/o ensemble geometry" arm of
        # Table 2 together with a mean-distance-selected cache, which also takes
        # dispersion out of the edge membership. Unlike masking this changes the
        # feature width, and therefore the parameter count.
        self.meanonly = bool(meanonly)
        if self.meanonly:
            DROP_NODE = {"bl_nbr_sd", "bl_frame_sd", "ba_cos_nbr_sd",
                         "ba_cos_frame_sd", "tor_R", "tor_circsd", "tor_frame_sd"}
            DROP_SPACE = {"d_sd", "d_min", "d_max"}
            self._nidx = [i for i in self._nidx if cols[i] not in DROP_NODE]
            self._sidx = [i for i in self._sidx if sc[i] not in DROP_SPACE]

        # Name-based pruning on top of that. An unknown name is an error, not a
        # no-op: a typo would silently keep the column and invalidate the arm.
        def _prune(idx, names, drop, what):
            drop = [str(c) for c in (drop or ())]
            if not drop:
                return idx
            unknown = [c for c in drop if c not in names]
            if unknown:
                raise ValueError("drop_%s_cols: unknown column(s) %s" % (what, unknown))
            keep = [i for i in idx if names[i] not in set(drop)]
            if not keep:
                raise ValueError("drop_%s_cols would remove every column" % what)
            return keep

        self.dropped = list(drop_space_cols or ())
        self._sidx = _prune(self._sidx, sc, drop_space_cols, "space")
        # edge_block="bond" drops the whole 3D contact relation: the
        # "w/o 3D contact edges" ablation of Table 2.
        if edge_block not in ("both", "bond"):
            raise ValueError("edge_block must be 'both' or 'bond'")
        self.edge_block = edge_block


        # Second solvent. Molecules absent from either cache are dropped, so both
        # views always exist for every molecule the model sees.
        self.solvent_token = bool(solvent_token)
        self.alt = None
        if alt_cache:
            with open(alt_cache, "rb") as fh:
                self.alt = pickle.load(fh)["feat"]
            self.feat = {p: g for p, g in self.feat.items() if p in self.alt}

        self.node_dim = len(self._nidx) + (1 if self.solvent_token else 0)
        self.bond_dim = len(self._bkeep)
        self.space_dim = len(self._sidx)
        self._st = None

    def usable(self, ids):
        return [p for p in ids if p in self.feat and p in self.labels]

    def fit_scaler(self, train_ids):
        """Appendix E: column-wise mu and sigma over the training partition only."""
        N = np.concatenate([self.feat[p]["node"] for p in train_ids], 0)
        B = np.concatenate([self.feat[p]["bond_attr"] for p in train_ids], 0)
        S = np.concatenate([self.feat[p]["space_attr"] for p in train_ids], 0)
        st = {}
        for k, A in (("node", N), ("bond", B), ("space", S)):
            mu, sd = A.mean(0), A.std(0)
            sd = np.where(sd < 1e-6, 1.0, sd)
            st[k] = (mu.astype(np.float32), sd.astype(np.float32))
        self._st = st
        return {"node_zero_var": int((N.std(0) < 1e-6).sum()),
                "bond_zero_var": int((B.std(0) < 1e-6).sum()),
                "space_zero_var": int((S.std(0) < 1e-6).sum())}

    def get(self, pid, which="base"):
        g = self.feat[pid] if which == "base" else self.alt[pid]
        mu, sd = self._st["node"]
        x = ((g["node"] - mu) / sd)[:, self._nidx]
        if self.solvent_token:
            tok = np.full((x.shape[0], 1), 0.0 if which == "base" else 1.0, dtype=x.dtype)
            x = np.hstack([x, tok])
        mu, sd = self._st["bond"]
        ba = ((g["bond_attr"] - mu) / sd)[:, self._bkeep]
        mu, sd = self._st["space"]
        sa = ((g["space_attr"] - mu) / sd)[:, self._sidx]
        t = lambda a, dt=np.float32: torch.from_numpy(np.ascontiguousarray(a, dt))
        if self.edge_block == "bond":
            sa = sa[:0]
            g = dict(g)
            g["space_index"] = g["space_index"][:, :0]
        # m_ij of Eq. (7): flags contact edges selected by occupancy but NOT
        # by the mean-distance criterion. Not a column of e_ij; it is carried
        # alongside, outside the standardization path (Appendix A).
        _tm = g.get("trans_mask")
        if _tm is None:
            _tm = np.zeros(g["space_index"].shape[1], np.bool_)
        elif self.edge_block == "bond":
            _tm = _tm[:0]
        return {"pid": pid, "x": t(x),
                "trans_mask": torch.from_numpy(np.ascontiguousarray(_tm, np.bool_)),
                "bond_index": torch.from_numpy(g["bond_index"].astype(np.int64)),
                "bond_attr": t(ba),
                "space_index": torch.from_numpy(g["space_index"].astype(np.int64)),
                "space_attr": t(sa),
                "y": torch.tensor(self.labels[pid], dtype=torch.float32)}

    def view(self, ids, both=False):
        ids = self.usable(ids)
        if both and self.alt is not None:
            return _V2(self, ids)
        return _V(self, ids)


class _V(Dataset):
    def __init__(self, o, ids):
        self.o, self.ids = o, ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.o.get(self.ids[i])


class _V2(Dataset):
    """Each molecule twice: water graph first, then hexane graph (Section 3.3)."""

    def __init__(self, o, ids):
        self.o, self.ids = o, ids

    def __len__(self):
        return 2 * len(self.ids)

    def __getitem__(self, i):
        n = len(self.ids)
        return self.o.get(self.ids[i % n], "base" if i < n else "alt")


def mech_collate(samples):
    """Batch graphs, offsetting BOTH edge relations by the same node offset."""
    xs, bis, bas, sis, sas, bidx, ys, pids, tms = [], [], [], [], [], [], [], [], []
    off = 0
    for k, s in enumerate(samples):
        n = s["x"].shape[0]
        xs.append(s["x"])
        bis.append(s["bond_index"] + off)
        bas.append(s["bond_attr"])
        sis.append(s["space_index"] + off)
        sas.append(s["space_attr"])
        tms.append(s["trans_mask"])
        bidx.append(torch.full((n,), k, dtype=torch.long))
        ys.append(s["y"])
        pids.append(s["pid"])
        off += n
    return {"x": torch.cat(xs),
            "bond_index": torch.cat(bis, 1), "bond_attr": torch.cat(bas),
            "space_index": torch.cat(sis, 1), "space_attr": torch.cat(sas),
            "trans_mask": torch.cat(tms),
            "batch": torch.cat(bidx), "y": torch.stack(ys),
            "pids": pids, "n_graphs": len(samples)}
