"""Run the whole grid: 3 splits x 5 seeds, one worker per free GPU (Appendix E).

The sweep looks at the machine before it uses it. A GPU is taken only if nobody
else has a process on it and it is idle; CPU threads are capped at 80 % of the
cores this process is allowed to use, shared out between workers, so a shared
box stays usable. The two solvent caches are built once in this process before
any worker starts, because fifteen processes building the same cache would race.

    python -m cheddar sweep              # every unfinished run, all free GPUs
    python -m cheddar sweep --dry-run    # print the plan and exit
    python -m cheddar sweep --splits ID --seeds 1,2 --gpus 0,1

Runs land in runs/cheddar-<split>-s<seed>/ and a finished run is never redone,
so an interrupted sweep continues where it stopped.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "cheddar.json"
SPLITS = ("ID", "OD_Murcko", "Cliff_ratio")
SEEDS = (1, 2, 7, 13, 42)
CPU_CAP = 0.8                 # never ask for more than this share of the cores
THREADS_MAX = 4               # per worker; the model is small and GPU-bound
MIN_FREE_MB = 3000            # a run peaks at 0.72 GB, so this is generous
MAX_UTIL = 25                 # percent, averaged over the sampling instant


def _smi(query, flag="--query-gpu"):
    try:
        out = subprocess.run(["nvidia-smi", flag + "=" + query,
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return [[c.strip() for c in ln.split(",")] for ln in out.stdout.splitlines() if ln.strip()]


def gpus():
    """(usable, skipped). A GPU with anyone's process on it is left alone."""
    rows = _smi("index,uuid,memory.total,memory.used,utilization.gpu")
    if not rows:
        return [], [("-", "no nvidia-smi, running on CPU")]
    apps = _smi("gpu_uuid,pid,used_memory", "--query-compute-apps") or []
    busy = {}
    for a in apps:
        busy.setdefault(a[0], []).append(a[1])
    usable, skipped = [], []
    for idx, uuid, total, used, util in rows:
        i = int(idx)
        if uuid in busy:
            skipped.append((i, "in use by pid %s" % ",".join(busy[uuid])))
        elif float(total) - float(used) < MIN_FREE_MB:
            skipped.append((i, "only %d MB free" % (float(total) - float(used))))
        elif float(util) > MAX_UTIL:
            skipped.append((i, "%d%% utilised" % float(util)))
        else:
            usable.append(i)
    return usable, skipped


def cpu_plan(nworkers):
    """Cores per worker, honouring the 80 % cap and what the box is already doing."""
    cores = sorted(os.sched_getaffinity(0))
    try:
        busy = int(round(os.getloadavg()[0]))
    except OSError:
        busy = 0
    budget = max(1, int(CPU_CAP * len(cores)) - min(busy, len(cores) - 1))
    per = max(1, min(THREADS_MAX, budget // max(1, nworkers)))
    out, k = [], 0
    for _ in range(nworkers):
        out.append(cores[k:k + per] or cores[:per])
        k = (k + per) % max(1, len(cores) - per + 1)
    return out, per, len(cores), budget


def pending(splits, seeds, out_root, stem="cheddar"):
    jobs = []
    for s in splits:
        for d in seeds:
            out = Path(out_root) / ("%s-%s-s%d" % (stem, s, d))
            if not (out / "test_metrics.json").exists():
                jobs.append((s, d, out))
    return jobs


def _launch(job, gpu, cores, threads, config=CONFIG):
    split, seed, out = job
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        env[k] = str(threads)
    cmd = [sys.executable, "-m", "cheddar.train", str(config), "--data-root", str(ROOT),
           "--split", split, "--seed", str(seed), "--out", str(out)]
    log = open(out / "train.log", "w")
    p = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT,
                         preexec_fn=(lambda c=tuple(cores): os.sched_setaffinity(0, c)))
    p._log, p._job, p._t0 = log, job, time.time()
    return p


def report(jobs_done, out_root, stem="cheddar"):
    """Mean +- sd per split over whatever finished, plus one plain sentence."""
    import numpy as np
    rows = {}
    for s in SPLITS:
        for d in SEEDS:
            f = Path(out_root) / ("%s-%s-s%d" % (stem, s, d)) / "test_metrics.json"
            if f.exists():
                rows.setdefault(s, []).append(json.load(open(f))[s])
    print("\n%-12s %4s  %-16s %-16s" % ("split", "n", "MSE", "R2"))
    for s in SPLITS:
        v = rows.get(s)
        if not v:
            continue
        f = lambda k: (float(np.mean([x[k] for x in v])), float(np.std([x[k] for x in v], ddof=1)))
        print("%-12s %4d  %.3f +- %.3f    %.3f +- %.3f" % (s, len(v), *f("mse"), *f("r2")))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG), help="a config in configs/, default the released one")
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--gpus", default=None, help="comma list; default is every free GPU")
    ap.add_argument("--max-workers", type=int, default=0)
    ap.add_argument("--out-root", default=str(ROOT / "runs"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    splits = [s for s in a.splits.split(",") if s]
    seeds = [int(s) for s in a.seeds.split(",") if s]
    for s in splits:
        if s not in SPLITS:
            raise SystemExit("unknown split %s; choose from %s" % (s, ", ".join(SPLITS)))

    stem = Path(a.config).stem
    jobs = pending(splits, seeds, a.out_root, stem)
    total = len(splits) * len(seeds)
    if not jobs:
        print("all %d runs already have results in %s" % (total, a.out_root))
        report([], a.out_root, stem)
        return

    if a.gpus is not None:
        dev, skipped = [int(x) for x in a.gpus.split(",") if x != ""], []
    else:
        dev, skipped = gpus()
    for i, why in skipped:
        print("  gpu %s skipped: %s" % (i, why))
    nw = max(1, min(len(dev) or 1, len(jobs)))
    if a.max_workers:
        nw = min(nw, a.max_workers)
    plan, per, ncore, budget = cpu_plan(nw)
    dev = (dev or [None])[:nw]

    print("sweep: %d of %d runs to do, %d worker%s on %s, %d thread%s each "
          "(%d of %d cores allowed)"
          % (len(jobs), total, nw, "" if nw == 1 else "s",
             "cpu" if dev[0] is None else "gpu " + ",".join(str(d) for d in dev),
             per, "" if per == 1 else "s", min(nw * per, budget), ncore))
    if dev[0] is None:
        print("  no free GPU; one CPU worker is about 20x slower than a 3090")
    for s, d, out in jobs:
        print("  todo %-12s seed %-3d -> %s" % (s, d, out))
    if a.dry_run:
        print("dry run, nothing launched")
        return

    cfg = json.load(open(a.config))
    from .data import ensure_cache
    for key in ("cache", "alt_cache"):
        p = ROOT / cfg[key]
        if not p.exists() or p.stat().st_size == 0:
            print("building %s first, so the workers do not race for it" % cfg[key], flush=True)
            ensure_cache(str(p))

    queue, running, done, t0 = list(jobs), {}, [], time.time()
    while queue or running:
        for k, gpu in enumerate(dev):
            if gpu in running or not queue:
                continue
            job = queue.pop(0)
            running[gpu] = _launch(job, gpu, plan[k], per, a.config)
            print("[%5.1f min] gpu %s <- %s seed %d" %
                  ((time.time() - t0) / 60, gpu, job[0], job[1]), flush=True)
        time.sleep(5)
        for gpu, p in list(running.items()):
            if p.poll() is None:
                continue
            p._log.close()
            split, seed, out = p._job
            mse = ""
            f = out / "test_metrics.json"
            if f.exists():
                mse = "mse %.4f" % json.load(open(f))[split]["mse"]
            print("[%5.1f min] gpu %s -- %s seed %d %s %s (%d of %d, %.1f min)"
                  % ((time.time() - t0) / 60, gpu, split, seed,
                     "ok" if p.returncode == 0 else "FAILED rc=%d, see %s/train.log"
                     % (p.returncode, out), mse, len(done) + 1, len(jobs),
                     (time.time() - p._t0) / 60), flush=True)
            done.append((split, seed, p.returncode))
            del running[gpu]

    bad = [d for d in done if d[2] != 0]
    print("\nsweep finished in %.1f min, %d ok, %d failed"
          % ((time.time() - t0) / 60, len(done) - len(bad), len(bad)))
    rows = report(done, a.out_root, stem)
    json.dump({"runs": [{"split": s, "seed": d, "rc": r} for s, d, r in done],
               "minutes": (time.time() - t0) / 60,
               "n_by_split": {k: len(v) for k, v in rows.items()}},
              open(Path(a.out_root) / "sweep.json", "w"), indent=1)


if __name__ == "__main__":
    main()
