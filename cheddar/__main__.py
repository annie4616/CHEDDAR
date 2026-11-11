"""One entry point for the whole package.

    python -m cheddar               the default: score the 15 shipped checkpoints
    python -m cheddar check         what is installed, what data is present
    python -m cheddar data          find the MD trajectories, or download them
    python -m cheddar eval          score the checkpoints and say what that means
    python -m cheddar predict 5222  permeability for named peptides
    python -m cheddar sweep         retrain the grid, one worker per free GPU
    python -m cheddar train         one run (--split, --seed)

Nothing takes a path. The trajectories are found wherever they already are, and
downloaded from Zenodo if they are nowhere. Everything except `sweep` and
`train` runs happily on a CPU; scoring all 15 checkpoints there takes about 20 s.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = str(ROOT / "configs" / "cheddar.json")
SPLITS = ("ID", "OD_Murcko", "Cliff_ratio")
SEEDS = (1, 2, 7, 13, 42)


def _config(argv):
    """Pull an optional --config PATH out of argv; default is the released one."""
    if "--config" in argv:
        i = argv.index("--config")
        return argv[i + 1], argv[:i] + argv[i + 2:]
    return CONFIG, argv


def _run(module_main, argv):
    old = sys.argv
    sys.argv = argv
    try:
        module_main()
    finally:
        sys.argv = old


def check():
    """Say what is here and what is not, in plain terms."""
    import importlib
    import json
    print("CHEDDAR in %s\n" % ROOT)
    for mod in ("torch", "numpy", "rdkit", "joblib"):
        try:
            m = importlib.import_module(mod)
            print("  %-8s %s" % (mod, getattr(m, "__version__", "ok")))
        except ImportError:
            print("  %-8s MISSING   conda env create -f environment.yml" % mod)
    try:
        import torch
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print("  device   %s" % ("%d cuda gpu%s" % (n, "" if n == 1 else "s") if n
                                 else "cpu (fine, just slower)"))
    except ImportError:
        pass
    print()
    cfg = json.load(open(CONFIG))
    for label, rel in (("label table", cfg["label_csv"]), ("atom annotation", "data/atom_annot.pkl"),
                       ("SMILES fix", "data/nal_fix.json"), ("splits", "data/ID_split.csv"),
                       ("water cache", cfg["cache"]), ("hexane cache", cfg["alt_cache"])):
        p = ROOT / rel
        print("  %-16s %s %s" % (label, "found  " if p.exists() else "MISSING",
                                 rel if not p.exists() else "%s (%.0f MB)" % (rel, p.stat().st_size / 2**20)))
    n = len(list((ROOT / "ckpts").glob("cheddar-*.pt"))) if (ROOT / "ckpts").is_dir() else 0
    print("  %-16s %s %d of 15" % ("checkpoints", "found  " if n == 15 else "PARTIAL", n))
    from . import fetch
    traj = fetch.locate()
    print("  %-16s %s" % ("trajectories", traj or "not on this machine yet"))
    if not (ROOT / cfg["cache"]).exists():
        print("\nThe two solvent caches are built from the MD trajectories of CycPeptMPDB-4D.\n"
              + ("They are at %s and any command below builds the caches on first use,\n"
                 "about 72 s." % traj if traj else
                 "They are the public Zenodo record %s, 1.6 GB zip and %.1f GB unpacked;\n"
                 "`python -m cheddar data` downloads and unpacks it, or set CHEDDAR_TRAJ if\n"
                 "you have it already." % (fetch.RECORD, fetch.UNPACKED_GB)))


def evaluate(argv):
    from . import eval as ev
    _run(ev.main, ["cheddar.eval", CONFIG, "--data-root", str(ROOT), "--all", "--brief"] + argv)


def predict(argv):
    from . import predict as pr
    _run(pr.main, ["cheddar.predict"] + argv)


def data(argv):
    from . import fetch
    _run(fetch.main, ["cheddar.data"] + argv)


def sweep(argv):
    from . import sweep as sw
    cfg, rest = _config(argv)
    _run(sw.main, ["cheddar.sweep"] + (["--config", cfg] if cfg != CONFIG else []) + rest)


def train(argv):
    from . import train as tr
    CFG, argv = _config(argv)
    if "--all" in argv:
        rest = [a for a in argv if a != "--all"]
        for split in SPLITS:
            for seed in SEEDS:
                print("\n=== %s seed %d" % (split, seed), flush=True)
                _run(tr.main, ["cheddar.train", CFG, "--data-root", str(ROOT),
                               "--split", split, "--seed", str(seed),
                               "--out", str(ROOT / "runs" / ("cheddar-%s-s%d" % (split, seed)))]
                     + rest)
    else:
        _run(tr.main, ["cheddar.train", CFG, "--data-root", str(ROOT)] + argv)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    rest = sys.argv[2:]
    if cmd in ("-h", "--help", "help"):
        print(__doc__)
    elif cmd == "check":
        check()
    elif cmd in ("eval", "reproduce"):
        evaluate(rest)
    elif cmd == "predict":
        predict(rest)
    elif cmd == "data":
        data(rest)
    elif cmd == "sweep":
        sweep(rest)
    elif cmd == "train":
        train(rest)
    else:
        print(__doc__)
        raise SystemExit("unknown command: %s" % cmd)


if __name__ == "__main__":
    main()
