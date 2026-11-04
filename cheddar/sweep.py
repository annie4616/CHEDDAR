"""Reproduce the CHEDDAR row of Table 1: 3 splits x 5 seeds = 15 runs.

    python -m cheddar.sweep configs/cheddar.json --data-root $DATA_ROOT
"""
import argparse
import json
import os
import subprocess
import sys

SPLITS = ("ID", "OD_Murcko", "Cliff_ratio")
SEEDS = (1, 2, 7, 13, 42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/cheddar.json")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()
    name = os.path.splitext(os.path.basename(a.config))[0]
    for split in SPLITS:
        for seed in SEEDS:
            d = os.path.join(a.out, "%s-%s-s%d" % (name, split, seed))
            done = os.path.join(d, "test_metrics.json")
            if os.path.exists(done) and os.path.getsize(done):
                print("skip %s" % d, flush=True)
                continue
            subprocess.run([sys.executable, "-m", "cheddar.train", a.config,
                            "--data-root", a.data_root, "--split", split,
                            "--seed", str(seed), "--out", d], check=True)
    rows = {}
    for split in SPLITS:
        vals = []
        for seed in SEEDS:
            f = os.path.join(a.out, "%s-%s-s%d" % (name, split, seed), "test_metrics.json")
            if os.path.exists(f):
                vals.append(json.load(open(f))[split]["mse"])
        if vals:
            rows[split] = sum(vals) / len(vals)
    print(json.dumps({"mean_test_mse": rows}, indent=2))


if __name__ == "__main__":
    main()
