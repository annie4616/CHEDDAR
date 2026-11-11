"""Evaluate released CHEDDAR checkpoints (Table 1, Appendix F).

No training happens here. A checkpoint is loaded, the standardization of
Appendix E is refitted on the train partition of its own split, and the model is
scored on one partition with the two solvent views averaged, which is the
inference rule of Section 3.4.

  python -m cheddar.eval configs/cheddar.json --ckpt ckpts/cheddar-ID-s1.pt
  python -m cheddar.eval configs/cheddar.json --all        # 15 runs, mean +- sd

Paths inside a config may be relative; they are resolved against --data-root.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import MechData, mech_collate
from .features import read_split
from .model import MechGraphNet
from .train import SEEDS, SPLITS, denorm, metrics, resolve, run_epoch

CKPT = "cheddar-%s-s%d.pt"

# The CHEDDAR row of Table 1, so the summary can say whether this machine
# reproduced it rather than leaving the reader to compare by eye.
TABLE1 = {"ID": (0.145, 0.621), "OD_Murcko": (0.236, 0.660), "Cliff_ratio": (0.142, 0.683)}
NAMED = {"ID": "the random split", "OD_Murcko": "the scaffold split",
         "Cliff_ratio": "the activity-cliff split"}


def load(cfg, split, device):
    """Dataset, loaders and a fresh model for one split. The scaler is fitted on
    train ids only, exactly as in training, so the evaluation is unbiased."""
    data = MechData(cfg["cache"], cfg["label_csv"], cfg.get("label_col", "PAMPA"),
                    edge_block=cfg.get("edge_block", "both"),
                    solvent_token=cfg.get("solvent_token", True),
                    alt_cache=cfg.get("alt_cache"),
                    meanonly=cfg.get("meanonly", False),
                    drop_space_cols=cfg.get("drop_space_cols", ()))
    sp = lambda f: data.usable(read_split(cfg["split_csv"].format(split=split)
                                          if "{split}" in cfg["split_csv"] else cfg["split_csv"], f))
    ids = {f: sp(f) for f in ("train", "val", "test")}
    data.fit_scaler(ids["train"])
    both = bool(cfg.get("both_solvents", True))
    loaders = {f: DataLoader(data.view(v, both=both), batch_size=cfg["batch_size"],
                             shuffle=False, num_workers=0, collate_fn=mech_collate)
               for f, v in ids.items()}
    model = MechGraphNet(data.node_dim, data.bond_dim, data.space_dim,
                         hidden=cfg.get("hidden", 128), layers=cfg.get("layers", 3),
                         dropout=cfg.get("dropout", 0.5),
                         head_hidden=cfg.get("head_hidden", 128),
                         trans_drop=cfg.get("trans_drop", 0.2)).to(device)
    return data, ids, loaders, model, both


def score(model, ckpt_path, loader, device, both):
    blob = torch.load(ckpt_path, map_location=device)
    state = blob.get("model", blob)
    model.load_state_dict(state)
    y, p = run_epoch(model, loader, device, fold2=both)
    m = metrics(denorm(y), denorm(p))
    m["n_params"] = sum(t.numel() for t in model.parameters())
    m["epoch"] = blob.get("epoch", -1)
    return m, denorm(y), denorm(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--data-root", default=".", help="prefix for relative paths in the config")
    ap.add_argument("--ckpt", default=None, help="one checkpoint; split is read from its name")
    ap.add_argument("--ckpt-dir", default="ckpts")
    ap.add_argument("--split", default=None, choices=SPLITS)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--partition", default="test", choices=("train", "val", "test"))
    ap.add_argument("--all", action="store_true", help="every split and seed, mean +- sd")
    ap.add_argument("--brief", action="store_true",
                    help="one progress line per checkpoint instead of its metrics json")
    ap.add_argument("--out", default=None, help="write predictions.csv here")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg = json.load(open(a.config))

    if a.ckpt and not a.all:
        name = Path(a.ckpt).stem
        m = re.match(r".*-(%s)-s(\d+)$" % "|".join(SPLITS), name)
        if not m:
            raise SystemExit("cannot read split and seed from %s; pass --split and --seed" % name)
        jobs = [(a.split or m.group(1), a.seed if a.seed is not None else int(m.group(2)), a.ckpt)]
    else:
        splits = SPLITS if (a.all or not a.split) else (a.split,)
        seeds = SEEDS if (a.all or a.seed is None) else (a.seed,)
        root = Path(a.data_root) / a.ckpt_dir if not os.path.isabs(a.ckpt_dir) else Path(a.ckpt_dir)
        jobs = [(s, d, str(root / (CKPT % (s, d)))) for s in splits for d in seeds]

    rows, cached, npart = {}, {}, {}
    for split, seed, path in jobs:
        if not os.path.exists(path):
            print("MISSING %s" % path, flush=True)
            continue
        if split not in cached:    # the cache is ~2 GB; load it once per split
            c = resolve(dict(cfg, split=split), a.data_root)
            cached[split] = load(c, split, a.device)
        data, ids, loaders, model, both = cached[split]
        npart[split] = len(ids[a.partition])
        m, y, p = score(model, path, loaders[a.partition], a.device, both)
        rows.setdefault(split, []).append(m)
        if a.brief:
            print("  %-12s seed %-3d %4d %s peptides   mse %.4f  r2 %.4f"
                  % (split, seed, len(ids[a.partition]), a.partition, m["mse"], m["r2"]),
                  flush=True)
        else:
            print(json.dumps({"split": split, "seed": seed, "partition": a.partition,
                              "n": len(ids[a.partition]), **m}), flush=True)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            with open(a.out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["pid", "y", "pred"])
                pids = [q for b in loaders[a.partition] for q in b["pids"]]
                for q, yy, pp in zip(pids, y, p):
                    w.writerow([q, "%.6f" % yy, "%.6f" % pp])

    if len(jobs) > 1:
        print("\n%-12s %4s  %-16s %-16s" % ("split", "n", "MSE", "R2"))
        agg = {}
        for split in SPLITS:
            v = rows.get(split)
            if not v:
                continue
            f = lambda k: (float(np.mean([x[k] for x in v])),
                           float(np.std([x[k] for x in v], ddof=1)) if len(v) > 1 else 0.0)
            agg[split] = (len(v), f("mse"), f("r2"))
            print("%-12s %4d  %.3f +- %.3f    %.3f +- %.3f"
                  % (split, len(v), *agg[split][1], *agg[split][2]))
        print()
        ok = []
        word = {"test": "held-out", "val": "validation", "train": "training"}[a.partition]
        for split, (n, (mse, msd), (r2, rsd)) in agg.items():
            print("On %s, %d seed%s give a %s mean squared error of %.3f (sd %.3f) and an "
                  "R2 of %.3f (sd %.3f) over %d %s peptides."
                  % (NAMED.get(split, split), n, "" if n == 1 else "s", a.partition,
                     mse, msd, r2, rsd, npart[split], word))
            if split in TABLE1:
                ok.append(abs(mse - TABLE1[split][0]) <= 0.0015
                          and abs(r2 - TABLE1[split][1]) <= 0.0015)
        if ok and all(ok) and len(ok) == len(TABLE1):
            print("All three agree with Table 1 of the paper to the printed precision.")
        elif ok:
            bad = [s for s, (n, (m, _), (r, _)) in agg.items() if s in TABLE1 and
                   (abs(m - TABLE1[s][0]) > 0.0015 or abs(r - TABLE1[s][1]) > 0.0015)]
            if bad:
                print("Differs from Table 1 on %s; the paper reports %s."
                      % (", ".join(bad),
                         ", ".join("%.3f / %.3f" % TABLE1[s] for s in bad)))


if __name__ == "__main__":
    main()
