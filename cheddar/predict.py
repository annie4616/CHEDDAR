"""Predict permeability for peptides that have MD ensembles (Section 3.3).

    python -m cheddar predict 5222 4705 4271      # a few peptides
    python -m cheddar predict --partition test    # a whole partition
    python -m cheddar predict --all --out pred.csv

Each prediction is the water and hexane views averaged, then averaged over the
five released seeds of one split; the spread over seeds is printed beside it.
Permeability is log P_app in cm/s, and training clips the label to [-8, -4], so a
prediction at either end of that window means "beyond the trained range".

A prediction needs the molecule's solvent ensembles, which is what the model
reads; SMILES alone is not enough. Any peptide in CycPeptMPDB-4D can be named
here. A new peptide needs its two MD trajectories first, placed in the
trajectory directory under the same naming as the public release.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .data import MechData, mech_collate
from .features import read_labels, read_split
from .model import MechGraphNet
from .train import SEEDS, SPLITS, denorm, resolve

ROOT = Path(__file__).resolve().parent.parent
CLIP = (-8.0, -4.0)

# Plain-language bands for readers who do not work in log units. They are
# percentiles of the 4,925 measured PAMPA values, not a clinical threshold:
# "high" means high for a cyclic peptide in this dataset, which is the only
# scale the model was ever fitted on.
BANDS = ((90, "high"), (65, "above average"), (35, "mid-range"),
         (10, "low"), (0, "very low"))


def measured_scale(label_csv, label_col="PAMPA"):
    """The measured log P_app values, sorted, for ranking a prediction."""
    return np.sort(np.fromiter(read_labels(label_csv, label_col).values(), float))


def rank(value, scale):
    """(percentile, word) for one prediction among the measured peptides."""
    pct = 100.0 * float(np.searchsorted(scale, value)) / len(scale)
    return pct, next(w for lo, w in BANDS if pct >= lo)


def models(split, seeds, cfg, data, device, ckpt_dir):
    out = []
    for s in seeds:
        p = Path(ckpt_dir) / ("cheddar-%s-s%d.pt" % (split, s))
        if not p.exists():
            print("  no checkpoint for seed %d (%s), skipping it" % (s, p), file=sys.stderr)
            continue
        m = MechGraphNet(data.node_dim, data.bond_dim, data.space_dim,
                         hidden=cfg.get("hidden", 128), layers=cfg.get("layers", 3),
                         dropout=cfg.get("dropout", 0.5),
                         head_hidden=cfg.get("head_hidden", 128),
                         trans_drop=cfg.get("trans_drop", 0.2)).to(device)
        blob = torch.load(p, map_location=device)
        m.load_state_dict(blob.get("model", blob))
        m.train(False)
        out.append((s, m))
    if not out:
        raise SystemExit("no checkpoints for split %s in %s" % (split, ckpt_dir))
    return out


def predict(ids, split="ID", seeds=SEEDS, device=None, ckpt_dir=None, batch=64):
    """{pid: (mean, sd over seeds)} on the log P_app scale. ids=None means all."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = resolve(dict(json.load(open(ROOT / "configs" / "cheddar.json")), split=split), str(ROOT))
    data = MechData(cfg["cache"], cfg["label_csv"], cfg.get("label_col", "PAMPA"),
                    edge_block=cfg.get("edge_block", "both"),
                    solvent_token=cfg.get("solvent_token", True),
                    alt_cache=cfg.get("alt_cache"))
    data.fit_scaler(data.usable(read_split(cfg["split_csv"], "train")))
    if ids is None:
        ids = sorted(data.feat)
    have = [p for p in ids if p in data.feat]
    missing = [p for p in ids if p not in data.feat]
    nets = models(split, seeds, cfg, data, device, ckpt_dir or (ROOT / "ckpts"))
    both = bool(cfg.get("both_solvents", True))
    P = {p: [] for p in have}
    with torch.no_grad():
        for i in range(0, len(have), batch):
            chunk = have[i:i + batch]
            views = ["base"] + (["alt"] if both and data.alt is not None else [])
            bb = mech_collate([data.get(p, w) for w in views for p in chunk])
            bb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in bb.items()}
            for s, m in nets:
                y = m(bb).cpu().numpy()
                y = y.reshape(len(views), len(chunk)).mean(0)
                for p, v in zip(chunk, denorm(y)):
                    P[p].append(float(v))
    out = {p: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
           for p, v in P.items()}
    return out, missing, data, cfg, [s for s, _ in nets]


def main():
    ap = argparse.ArgumentParser(prog="cheddar predict")
    ap.add_argument("ids", nargs="*", help="CycPeptMPDB ids")
    ap.add_argument("--split", default="ID", choices=SPLITS,
                    help="which released model to use (default ID)")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--partition", default=None, choices=("train", "val", "test"))
    ap.add_argument("--all", action="store_true", help="every peptide in the cache")
    ap.add_argument("--ckpt-dir", default=str(ROOT / "ckpts"))
    ap.add_argument("--out", default=None, help="write a csv here")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    cfg0 = resolve(dict(json.load(open(ROOT / "configs" / "cheddar.json")), split=a.split),
                   str(ROOT))
    part = {f: set(read_split(cfg0["split_csv"], f)) for f in ("train", "val", "test")}
    ids = [str(i) for i in a.ids]
    if a.partition:
        ids += sorted(part[a.partition])
    if not ids and not a.all:
        ap.error("name some ids, or use --partition test, or --all")

    if a.all:
        ids = None                      # every molecule the cache holds
    res, missing, data, cfg, seeds = predict(
        ids, a.split, [int(s) for s in a.seeds.split(",") if s], a.device, a.ckpt_dir)

    raw = read_labels(cfg["label_csv"], cfg.get("label_col", "PAMPA"))
    where = lambda p: next((f for f in ("train", "val", "test") if p in part[f]), "-")
    print("%-10s %8s %6s %10s %-8s" % ("id", "logPapp", "sd", "measured", "partition"))
    order = sorted(res) if ids is None else [i for i in ids if i in res]
    for p in order:
        m, sd = res[p]
        tag = "" if CLIP[0] + 0.05 < m < CLIP[1] - 0.05 else "   (at the clip bound)"
        print("%-10s %8.2f %6.2f %10s %-8s%s"
              % (p, m, sd, ("%.2f" % raw[p]) if p in raw else "-", where(p), tag))
    for p in missing:
        print("%-10s %8s %6s %10s %-8s   no MD ensemble in the cache" % (p, "-", "-", "-", "-"))

    # For a handful of peptides, say the same thing in words. Suppressed for a
    # whole partition, where 490 paragraphs would help nobody.
    if 0 < len(order) <= 10:
        scale = measured_scale(cfg["label_csv"], cfg.get("label_col", "PAMPA"))
        print()
        for p in order:
            m, sd = res[p]
            pct, word = rank(m, scale)
            print("%s: predicted log P_app %.2f, %s permeability for a cyclic peptide "
                  "-- more permeable than %.0f%% of the %d measured ones."
                  % (p, m, word, pct, len(scale)))
            note = "   the five released models agree to within %.2f log units." % sd
            if p in raw:
                note = ("   measured %.2f, so this prediction is %.2f log units off."
                        % (raw[p], abs(m - raw[p])))
            print(note)

    v = np.array([m for m, _ in res.values()])
    if len(v):
        print("\n%d peptide%s predicted with the %d-seed %s model, water and hexane views "
              "averaged: log P_app from %.2f to %.2f, mean %.2f."
              % (len(v), "" if len(v) == 1 else "s", len(seeds), a.split,
                 v.min(), v.max(), v.mean()))
        inpart = [p for p in res if where(p) == "train"]
        if inpart:
            print("%d of them are in this model's training partition, so those numbers are "
                  "in-sample; use --split OD_Murcko for a held-out view." % len(inpart))
    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["CycPeptMPDB_ID", "pred_logPapp", "sd_over_seeds", "measured", "partition"])
            for p, (m, sd) in res.items():
                w.writerow([p, "%.4f" % m, "%.4f" % sd,
                            "%.4f" % raw[p] if p in raw else "", where(p)])
        print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
