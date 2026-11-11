"""Find the CycPeptMPDB-4D trajectories, or download them (Appendix B).

The ensemble features of Section 3.2 are statistics over MD frames, so the
trajectories are required to build the two solvent caches. They are a public
Zenodo record, not part of this repository:

    https://zenodo.org/records/21274695     1.6 GB zip, 10.2 GB unpacked

`locate()` looks for an already-unpacked copy in the usual places, so a reviewer
who has the data anywhere sensible needs no environment variable. `ensure()`
falls back to downloading and unpacking it, resuming an interrupted transfer and
verifying the MD5 the record publishes. Set CHEDDAR_NO_DOWNLOAD=1 to forbid the
download and get an explanatory error instead.
"""
from __future__ import annotations

import glob
import hashlib
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

RECORD = "21274695"
URL = ("https://zenodo.org/api/records/%s/files/CycPeptMPDB-4D.zip/content" % RECORD)
ZIP_NAME = "CycPeptMPDB-4D.zip"
ZIP_BYTES = 1714929909
ZIP_MD5 = "f607ae79244a455c4d27bc334a5e3884"
UNPACKED_GB = 10.2            # Water 5.09 + Hexane 5.10, Trajectories/ is all of it
NEED_GB = 12.5                # zip plus unpacked, before the zip is removed
SOLVENTS = ("Water", "Hexane")
NAMES = ("CycPeptMPDB-4D", "CycPeptMPDB_4D", "cycpeptmpdb-4d")

ROOT = Path(__file__).resolve().parent.parent


def is_traj_dir(d) -> bool:
    """True if d is a trajectory root, in either layout features.find_traj reads."""
    d = Path(d)
    for s in SOLVENTS:
        if next(iglob_first(d / s / "Trajectories" / "*_Traj.pdb"), None):
            return True
        if next(iglob_first(d / ("*_%s_traj.npz" % s)), None):
            return True
    return False


def iglob_first(pattern):
    return glob.iglob(str(pattern))


def _bases():
    home = Path.home()
    cwd = Path.cwd()
    return [ROOT / "data", ROOT, ROOT.parent, cwd, cwd.parent, home, home / "data",
            home / "Downloads", home / "datasets", home / "scratch",
            Path("/data"), Path("/datasets"), Path("/scratch"), Path("/mnt/data")]


def locate(extra=()):
    """The first existing trajectory root, or None. $CHEDDAR_TRAJ wins."""
    seen, cands = set(), []
    env = os.environ.get("CHEDDAR_TRAJ")
    for c in ([Path(env)] if env else []) + [Path(e) for e in extra]:
        cands.append(c)
    for b in _bases():
        cands.append(b)
        cands.extend(b / n for n in NAMES)
    for c in cands:
        try:
            c = c.expanduser()
        except RuntimeError:
            continue
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        if c.is_dir() and is_traj_dir(c):
            return str(c)
    return None


def _human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return "%.1f %s" % (n, u)
        n /= 1024.0


def _free_gb(p):
    p = Path(p)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 2**30


def download_dir():
    """Where to put the download: $CHEDDAR_DATA_HOME, else the roomiest sensible place."""
    env = os.environ.get("CHEDDAR_DATA_HOME")
    if env:
        return Path(env).expanduser()
    here = ROOT / "data"
    if _free_gb(here) >= NEED_GB:
        return here
    alts = [Path.home() / "data", Path.home(), Path("/data"), Path("/scratch")]
    ok = [(round(_free_gb(a), 1), a) for a in alts if a.is_dir() and os.access(a, os.W_OK)]
    if ok:
        best = max(ok)
        if best[0] >= NEED_GB:
            return best[1] / "cheddar-data"
    return here


def download(dest_dir, quiet=False):
    """Fetch the zip with resume, verify its MD5, return the local path."""
    import urllib.request

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    final, part = dest_dir / ZIP_NAME, dest_dir / (ZIP_NAME + ".part")
    if final.exists() and final.stat().st_size == ZIP_BYTES:
        return final
    have = part.stat().st_size if part.exists() else 0
    if have > ZIP_BYTES:
        part.unlink()
        have = 0
    free = _free_gb(dest_dir)
    if free < NEED_GB - have / 2**30:
        raise SystemExit(
            "only %.1f GB free on %s; the data needs about %.1f GB (1.6 GB zip, %.1f GB\n"
            "unpacked, the zip is removed afterwards). Point CHEDDAR_DATA_HOME at a\n"
            "roomier disk, or unpack the record there by hand and set CHEDDAR_TRAJ."
            % (free, dest_dir, NEED_GB, UNPACKED_GB))
    if not quiet:
        print("CycPeptMPDB-4D trajectories: downloading %s to %s\n  %s"
              % (_human(ZIP_BYTES), dest_dir, URL), file=sys.stderr)
        if have:
            print("  resuming at %s" % _human(have), file=sys.stderr)
    req = urllib.request.Request(URL, headers={"User-Agent": "cheddar/1.0"})
    if have:
        req.add_header("Range", "bytes=%d-" % have)
    t0, done = time.time(), have
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:      # server ignored the range: start over
            have, done = 0, 0
        mode = "ab" if have else "wb"
        with open(part, mode) as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if not quiet and done % (32 << 20) < (1 << 20):
                    el = max(time.time() - t0, 1e-9)
                    rate = (done - have) / el
                    left = (ZIP_BYTES - done) / rate if rate > 0 else 0
                    print("\r  %5.1f%%  %s  %4.1f MB/s  %3.0f s left   "
                          % (100.0 * done / ZIP_BYTES, _human(done),
                             rate / 2**20, left), end="", file=sys.stderr)
    if not quiet:
        print("\r  download complete, %s in %.0f s          "
              % (_human(done), time.time() - t0), file=sys.stderr)
    if done != ZIP_BYTES:
        raise SystemExit("got %d bytes, expected %d; rerun to resume" % (done, ZIP_BYTES))
    if os.environ.get("CHEDDAR_SKIP_MD5") != "1":
        h = hashlib.md5()
        with open(part, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
        if h.hexdigest() != ZIP_MD5:
            raise SystemExit("MD5 mismatch (%s, expected %s); delete %s and retry"
                             % (h.hexdigest(), ZIP_MD5, part))
        if not quiet:
            print("  md5 ok", file=sys.stderr)
    part.rename(final)
    return final


def unpack(zip_path, dest, quiet=False):
    """Extract into its own directory, skipping any member with an unsafe name."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        bad = [n for n in names
               if n.startswith(("/", "\\")) or ".." in Path(n).parts or Path(n).is_absolute()]
        if bad:
            raise SystemExit("archive has %d member(s) with unsafe paths, e.g. %s"
                             % (len(bad), bad[0]))
        if not quiet:
            print("  unpacking %d files (%.1f GB) to %s" % (len(names), UNPACKED_GB, dest),
                  file=sys.stderr)
        for i, n in enumerate(names):
            z.extract(n, dest)
            if not quiet and i % 2000 == 0:
                print("\r    %5.1f%%   " % (100.0 * i / len(names)), end="", file=sys.stderr)
    if not quiet:
        print("\r  unpacked                 ", file=sys.stderr)
    return str(dest)


def ensure(quiet=False):
    """Return a trajectory root, downloading the record if there is none."""
    found = locate()
    if found:
        if not quiet:
            print("trajectories: %s" % found, file=sys.stderr)
        return found
    if os.environ.get("CHEDDAR_NO_DOWNLOAD") == "1":
        raise FileNotFoundError(
            "no CycPeptMPDB-4D trajectories found and CHEDDAR_NO_DOWNLOAD=1.\n"
            "Unpack https://zenodo.org/records/%s and either place it at %s\n"
            "or set CHEDDAR_TRAJ to it." % (RECORD, ROOT / "data" / NAMES[0]))
    where = download_dir()
    zp = download(where, quiet=quiet)
    out = Path(where) / NAMES[0]
    unpack(zp, out, quiet=quiet)
    root = locate(extra=[out]) or (str(out) if is_traj_dir(out) else None)
    if not root:
        raise SystemExit("unpacked %s but found no %s/Trajectories under %s"
                         % (zp, SOLVENTS[0], out))
    if zp.exists() and os.environ.get("CHEDDAR_KEEP_ZIP") != "1":
        zp.unlink()
        if not quiet:
            print("  removed the zip, %.1f GB now in place" % UNPACKED_GB, file=sys.stderr)
    return root


def main():
    """python -m cheddar data : report or fetch, then say where it is."""
    root = ensure()
    n = len(glob.glob(os.path.join(root, "Water", "Trajectories", "*_Traj.pdb"))) or \
        len(glob.glob(os.path.join(root, "*_Water_traj.npz")))
    print("%s\n%d water trajectories; the builder will use the %s with an atom annotation"
          % (root, n, "4,925" if n >= 4925 else "available ones"))


if __name__ == "__main__":
    main()
