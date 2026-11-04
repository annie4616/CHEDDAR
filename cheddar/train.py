"""Train and evaluate CHEDDAR (Appendix F).

AdamW, weight decay 0.1; OneCycle at eta_max = 6e-4 with 20 % warmup; batch 16;
dropout 0.5; gradient-norm clipping 5.0; a fixed 120 epochs with no early
stopping. The reported model is the lowest-validation-MSE checkpoint and the
test partition is read once per run. Metrics are computed on the original
log P_app scale (Eq. 17).

Paths inside a config may be relative; they are resolved against --data-root.

  python -m cheddar.train configs/cheddar.json --data-root /path/to/data \
                         --split ID --seed 1 --out runs/cheddar-ID-s1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import MechData, mech_collate
from .features import read_split
from .model import MechGraphNet

SPLITS = ("ID", "OD_Murcko", "Cliff_ratio")
SEEDS = (1, 2, 7, 13, 42)


def denorm(a):
    """Eq. (16): reverse the linear rescaling, y = 2*y_norm - 6."""
    return a * 2.0 - 6.0


def metrics(y, p):
    e = (p - y) ** 2
    sst = float(np.sum((y - y.mean()) ** 2))
    return {"mse": float(e.mean()), "mae": float(np.abs(p - y).mean()),
            "r2": 1.0 - float(e.sum()) / sst if sst > 0 else float("nan"),
            "pearson": float(np.corrcoef(y, p)[0, 1]),
            "spearman": float(np.corrcoef(np.argsort(np.argsort(y)),
                                          np.argsort(np.argsort(p)))[0, 1]),
            "pred_sd": float(p.std()), "label_sd": float(y.std())}


def run_epoch(model, loader, dev, fold2=False):
    """Evaluation pass. fold2 averages the water and hexane predictions of the
    same molecule, which is the inference rule of Section 3.3."""
    model.train(False)
    Y, P = [], []
    with torch.no_grad():
        for bb in loader:
            bb = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in bb.items()}
            Y.append(bb["y"].cpu().numpy())
            P.append(model(bb).cpu().numpy())
    y, p = np.concatenate(Y), np.concatenate(P)
    if fold2 and len(y) % 2 == 0:
        h = len(y) // 2
        assert np.allclose(y[:h], y[h:]), "fold2 halves are not aligned"
        y, p = y[:h], 0.5 * (p[:h] + p[h:])
    return y, p


# The three split families, and the file each one lives in.
SPLIT_FILE = {"ID": "ID", "OD_Murcko": "OD", "Cliff_ratio": "Cliff"}


def resolve(cfg, root):
    """Make every path in the config absolute against --data-root, and point
    split_csv at the file for this run's split."""
    cfg["split_csv"] = cfg["split_csv"].format(split=SPLIT_FILE[cfg["split"]])
    for k in ("cache", "alt_cache", "label_csv", "split_csv"):
        v = cfg.get(k)
        if v and not os.path.isabs(v):
            cfg[k] = str(Path(root) / v)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--data-root", default=".", help="prefix for relative paths in the config")
    ap.add_argument("--split", default=None, choices=SPLITS)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None, help="output directory")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    if a.split:
        cfg["split"] = a.split
    if a.seed is not None:
        cfg["seed"] = a.seed
    cfg = resolve(cfg, a.data_root)          # needs cfg["split"] for split_csv
    out = Path(a.out or cfg.get("out_dir")
               or "runs/%s-%s-s%d" % (Path(a.config).stem, cfg["split"], cfg["seed"]))
    if (out / "test_metrics.json").exists():
        raise SystemExit("ABORT: %s already has results; refusing to overwrite" % out)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    if True:   # required by Appendix F, not optional
        # CUDA index_add_ uses atomics whose summation order varies between runs,
        # and float addition is not associative, so without this the same seed
        # does not reproduce (Appendix F).
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = MechData(cfg["cache"], cfg["label_csv"], cfg.get("label_col", "PAMPA"),
                    edge_block=cfg.get("edge_block", "both"),
                    solvent_token=cfg.get("solvent_token", True),
                    alt_cache=cfg.get("alt_cache"),
                    meanonly=cfg.get("meanonly", False),
                    drop_space_cols=cfg.get("drop_space_cols", ()))
    sp = lambda f: data.usable(read_split(cfg["split_csv"], f))
    tr, va, te = sp("train"), sp("val"), sp("test")
    zv = data.fit_scaler(tr)

    BOTH = bool(cfg.get("both_solvents", True))
    mk = lambda ids, sh, both=False: DataLoader(
        data.view(ids, both=both), batch_size=cfg["batch_size"],
        shuffle=sh, num_workers=0, collate_fn=mech_collate)
    ltr, lva, lte = mk(tr, True, BOTH), mk(va, False, BOTH), mk(te, False, BOTH)

    model = MechGraphNet(data.node_dim, data.bond_dim, data.space_dim,
                         hidden=cfg.get("hidden", 128),
                         layers=cfg.get("layers", 3),
                         dropout=cfg.get("dropout", 0.5),
                         head_hidden=cfg.get("head_hidden", 128),
                         trans_drop=cfg.get("trans_drop", 0.2)).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(json.dumps({"split": cfg["split"], "seed": cfg["seed"],
                      "node_dim": data.node_dim, "bond_dim": data.bond_dim,
                      "space_dim": data.space_dim, "n_params": nparam,
                      "edge_block": data.edge_block,
                      "meanonly": data.meanonly, "dropped": data.dropped,
                      "both_solvents": BOTH,
                      "train": len(tr), "val": len(va), "test": len(te), **zv}), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg.get("weight_decay", 0.1))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * max(1, len(ltr)),
        pct_start=cfg.get("warmup_frac", 0.2))

    best, best_ep, hist = float("inf"), -1, []
    for ep in range(cfg["epochs"]):
        t0 = time.time()
        model.train(True)
        for bb in ltr:
            bb = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in bb.items()}
            # Eq. (13): one supervised term per solvent view, both against the
            # same label; the two views are separate training examples.
            loss = torch.nn.functional.mse_loss(model(bb), bb["y"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
        yv, pv = run_epoch(model, lva, dev, fold2=BOTH)
        m = metrics(denorm(yv), denorm(pv))
        row = {"epoch": ep, "elapsed_s": time.time() - t0,
               **{"val/" + k: v for k, v in m.items()}}
        hist.append(row)
        print(json.dumps(row), flush=True)
        if m["mse"] < best:
            best, best_ep = m["mse"], ep
            torch.save({"model": model.state_dict(), "epoch": ep}, out / "best.pt")

    model.load_state_dict(torch.load(out / "best.pt", map_location=dev)["model"])
    yt, pt = run_epoch(model, lte, dev, fold2=BOTH)
    yt, pt = denorm(yt), denorm(pt)
    tm = metrics(yt, pt)
    tm.update({"best_epoch": best_ep, "best_val_mse": best, "n_params": nparam,
               "torch": torch.__version__})
    print("[test/%s] %s" % (cfg["split"], json.dumps(tm)), flush=True)
    json.dump({cfg["split"]: tm}, open(out / "test_metrics.json", "w"), indent=1)
    json.dump(hist, open(out / "history.json", "w"))
    with open(out / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pid", "y", "pred"])
        ids = [p for b in lte for p in b["pids"]]
        for p, yy, pp in zip(ids, yt, pt):
            w.writerow([p, "%.6f" % yy, "%.6f" % pp])


if __name__ == "__main__":
    main()
