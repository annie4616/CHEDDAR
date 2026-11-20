"""Build the atom annotation for peptides of your own (Section 3.1).

    python -m cheddar annotate --traj my_traj --csv my_peptides.csv \
                               --out data/my_annot.pkl
    CHEDDAR_ANNOT=data/my_annot.pkl ./predict.sh MYPEP1

The released `data/atom_annot.pkl` covers the 4,925 peptides of CycPeptMPDB-4D.
A peptide of your own needs its own entry, and this builds one from the same two
things the feature pass reads: the SMILES in the label csv, and the first frame
of the water trajectory.

What an entry has to supply is one permutation. The model's chemistry comes from
the SMILES through RDKit, and its geometry from the trajectory, so the only thing
that cannot be derived from either alone is which column of the coordinate array
belongs to which RDKit atom. Trajectory files do not store that: they are written
in whatever order the simulation used, hydrogens included, and in the released
data that order is not the SMILES order. So the annotation records, for RDKit
atom i, the absolute index npz_abs_idx of its heavy atom in the frame.

The matching is done on the molecular graph, not on atom names or residue
numbering, which differ between simulation packages. The first frame is read as a
molecule, bonds are inferred from distances, and RDKit is asked for the
isomorphism between that and the molecule the SMILES describes. Element and
degree have to agree atom by atom, so a mismatch is reported rather than
guessed at: a wrong permutation would still train and still predict, only on
features belonging to the wrong atoms.

One ambiguity is left, because it cannot be removed: atoms that the graph cannot
tell apart, such as the two methyls of a valine, may be assigned either way
round. Bond lengths are unaffected, and on this dataset the per-atom effect is
about 0.05 A on bonded distances and a few per cent of the 5 A contact set, so
predictions are not meaningfully changed -- but an annotation built here is not
guaranteed byte-identical to one built from the simulation's own atom order.

Stereochemistry is deliberately ignored while matching. A trajectory frame fixes
a configuration and the SMILES may name another, and for this purpose the two
need only agree as graphs -- the stereo channels on the node come from the
SMILES in either case.
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys

import numpy as np

HEAVY_FROM_PDB = ("ATOM", "HETATM")

# Enough matches to cover the symmetry a peptide actually has -- a handful of
# methyl and carboxyl pairs multiply out quickly -- while staying bounded on a
# molecule that turns out to be more symmetric than expected.
MAX_MATCHES = 20000


def _strain(ref, frame_mol, hit, topo):
    """Total excess, in Angstrom, over what the graph allows for each pair.

    This rejects a match that has moved an atom across the molecule, which is
    what matters when the frame's atom order is unrelated to the SMILES. It does
    NOT separate symmetry-equivalent matches: see `permutation`.
    """
    conf = frame_mol.GetConformer()
    total = 0.0
    for (i, j), lo, hi_ in topo:
        d = conf.GetAtomPosition(hit[i]).Distance(conf.GetAtomPosition(hit[j]))
        if d < lo:
            total += lo - d
        elif d > hi_:
            total += d - hi_
    return total


def _topology_windows(ref):
    """[((i, j), min, max)] for pairs two or three bonds apart in the SMILES.

    The windows are generous -- a 1-3 pair spans roughly 2.0 to 2.8 A and a 1-4
    pair 2.3 to 4.2 A as the torsion turns -- because they are only there to
    separate a correct match from one that has moved an atom across the molecule.
    """
    from rdkit import Chem

    d = Chem.GetDistanceMatrix(ref)
    out = []
    n = ref.GetNumAtoms()
    for i in range(n):
        for j in range(i + 1, n):
            if d[i, j] == 2:
                out.append(((i, j), 2.0, 2.8))
            elif d[i, j] == 3:
                out.append(((i, j), 2.3, 4.2))
    return out


def symmetry_classes(ref):
    """Atoms the molecular graph cannot tell apart, as a list of index groups."""
    from rdkit import Chem

    rank = list(Chem.CanonicalRankAtoms(ref, breakTies=False))
    byr = {}
    for i, r in enumerate(rank):
        byr.setdefault(r, []).append(i)
    return [v for v in byr.values() if len(v) > 1]


def read_first_frame(path):
    """(elements, coords) of the first model, every atom, in file order."""
    if path.endswith(".npz"):
        blob = np.load(path, allow_pickle=True)
        C = blob["coords"]
        el = blob["elements"] if "elements" in blob.files else None
        if el is None:
            raise SystemExit(
                "%s has no 'elements' array, so its atoms cannot be identified. "
                "Supply a pdb trajectory for annotation, or add elements to the npz."
                % path)
        return [str(e) for e in el], np.asarray(C[0], float)
    el, xyz = [], []
    with open(path) as fh:
        for ln in fh:
            if ln.startswith(HEAVY_FROM_PDB):
                # columns 77-78 hold the element; fall back to the atom name
                e = ln[76:78].strip() or ln[12:16].strip()
                el.append(e[:1].upper() + e[1:].lower() if len(e) > 1 else e.upper())
                xyz.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
            elif ln.startswith("ENDMDL") and el:
                break
    if not el:
        raise SystemExit("no ATOM/HETATM records in %s" % path)
    return el, np.asarray(xyz, float)


def frame_to_mol(elements, coords):
    """The frame as an RDKit molecule, heavy atoms only, bonds from geometry.

    Returns (mol, heavy_idx) where heavy_idx[k] is the absolute index in the
    frame of the molecule's atom k, so the two can be related afterwards.
    """
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    heavy = [i for i, e in enumerate(elements) if e != "H"]
    if not heavy:
        raise SystemExit("the frame has no heavy atoms")
    xyz = ["%d\n\n" % len(heavy)]
    for i in heavy:
        x, y, z = coords[i]
        xyz.append("%-3s %12.6f %12.6f %12.6f\n" % (elements[i], x, y, z))
    mol = Chem.MolFromXYZBlock("".join(xyz))
    if mol is None:
        raise SystemExit("RDKit could not read the frame as a molecule")
    # Charge 0 and no explicit valence check: the frame is a neutral peptide and
    # the bond orders do not matter here, only which pairs are bonded.
    try:
        rdDetermineBonds.DetermineConnectivity(mol, charge=0)
    except ValueError as e:
        raise SystemExit("bond perception failed on the frame: %s" % e)
    return mol, np.asarray(heavy, int)


def permutation(smiles, frame_mol):
    """For each RDKit atom of the SMILES, its atom index in frame_mol.

    Both molecules are compared as graphs with stereochemistry and bond orders
    stripped, since a frame fixes one configuration and carries no bond orders.

    Atoms the graph cannot distinguish are not resolved, and cannot be: the two
    methyls of a valine are the same vertex, so which 3D position is "the first"
    is arbitrary. Every bond length stays correct either way, and the residual is
    small -- on peptides of this dataset, bonded-neighbour distances move by
    about 0.05 A and the 5 A contact set by a few per cent -- but it is not zero.
    `symmetry_classes` lists the groups affected.
    """
    from rdkit import Chem

    ref = Chem.MolFromSmiles(smiles)
    if ref is None:
        return None, "the SMILES does not parse"
    if ref.GetNumAtoms() != frame_mol.GetNumAtoms():
        return None, ("%d heavy atoms in the SMILES, %d in the frame"
                      % (ref.GetNumAtoms(), frame_mol.GetNumAtoms()))

    def flat(m):
        q = Chem.RWMol(m)
        for a in q.GetAtoms():
            a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            a.SetNoImplicit(True)
            a.SetNumExplicitHs(0)
            a.SetFormalCharge(0)
            a.SetIsAromatic(False)
        for b in q.GetBonds():
            b.SetBondType(Chem.BondType.SINGLE)
            b.SetStereo(Chem.BondStereo.STEREONONE)
            b.SetIsAromatic(False)
        return q.GetMol()

    a, b = flat(ref), flat(frame_mol)
    # Substructure matching needs no sanitization of the flattened copies, and
    # sanitizing would reject the frame's perceived valences.
    #
    # A peptide has symmetry-equivalent atoms -- the two methyls of a valine are
    # the same vertex of the graph -- so there are many matches, and the graph
    # cannot say which is which. They are not interchangeable here: the ensemble
    # geometry of Eq. (1) is per atom, and swapping two methyls gives each the
    # other's neighbour distances. Pick the match whose bond lengths are physical
    # instead, since the frame's own bonded pairs are the ones at bonding distance.
    hits = b.GetSubstructMatches(a, useChirality=False, uniquify=False,
                                 maxMatches=MAX_MATCHES)
    if not hits:
        return None, "the frame and the SMILES are not the same molecular graph"
    topo = _topology_windows(ref)
    hit = min(hits, key=lambda h: _strain(ref, frame_mol, h, topo))
    if len(hits) >= MAX_MATCHES:
        print("  note: %d or more graph matches; the chosen one is geometrically "
              "consistent but may differ within a symmetry class" % MAX_MATCHES,
              file=sys.stderr)

    deg_a = [x.GetDegree() for x in ref.GetAtoms()]
    deg_b = [x.GetDegree() for x in frame_mol.GetAtoms()]
    for i, j in enumerate(hit):
        if ref.GetAtomWithIdx(i).GetAtomicNum() != frame_mol.GetAtomWithIdx(j).GetAtomicNum():
            return None, "element mismatch at SMILES atom %d" % i
        if deg_a[i] != deg_b[j]:
            return None, ("degree mismatch at SMILES atom %d: %d in the SMILES, "
                          "%d in the frame" % (i, deg_a[i], deg_b[j]))
    return list(hit), None


def annotate_one(pid, smiles, traj_path):
    """One annotation entry, or (None, reason)."""
    el, xyz = read_first_frame(traj_path)
    fmol, heavy_idx = frame_to_mol(el, xyz)
    perm, why = permutation(smiles, fmol)
    if perm is None:
        return None, why
    per_atom = [{"npz_heavy_idx": int(j), "npz_abs_idx": int(heavy_idx[j])}
                for j in perm]
    return {"per_atom": per_atom,
            "n_heavy": len(per_atom),
            "heavy_idx": heavy_idx,
            "smiles": smiles}, None


def main():
    from .features import find_traj

    ap = argparse.ArgumentParser(prog="cheddar annotate")
    ap.add_argument("--traj", required=True,
                    help="directory holding Water/ and Hexane/, as predict reads it")
    ap.add_argument("--csv", required=True,
                    help="label csv with CycPeptMPDB_ID and SMILES columns")
    ap.add_argument("--out", required=True, help="write the annotation pickle here")
    ap.add_argument("--merge", default=None,
                    help="start from this annotation and add to it, e.g. "
                         "data/atom_annot.pkl")
    ap.add_argument("--ids", default=None,
                    help="comma list; default is every row of the csv")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    for col in ("CycPeptMPDB_ID", "SMILES"):
        if rows and col not in rows[0]:
            raise SystemExit("%s has no %s column" % (a.csv, col))
    want = set(s.strip() for s in a.ids.split(",") if s.strip()) if a.ids else None

    ann = {}
    if a.merge:
        with open(a.merge, "rb") as f:
            ann = pickle.load(f)
        print("starting from %s, %d peptides" % (a.merge, len(ann)))

    done, kept, skipped = 0, 0, []
    for r in rows:
        pid = str(r["CycPeptMPDB_ID"]).strip()
        if want and pid not in want:
            continue
        f = find_traj(a.traj, pid, "Water")
        if f is None:
            # With --merge the label table holds the whole released set, whose
            # trajectories are elsewhere. Those already have an annotation, so
            # say nothing; only a peptide with neither is worth reporting.
            if pid not in ann:
                skipped.append((pid, "no water trajectory under %s" % a.traj))
            else:
                kept += 1
            continue
        try:
            entry, why = annotate_one(pid, r["SMILES"].strip(), f)
        except SystemExit as e:          # a bad frame should not stop the rest
            entry, why = None, str(e)
        if entry is None:
            skipped.append((pid, why))
            continue
        ann[pid] = entry
        done += 1
        print("  %-12s %d heavy atoms" % (pid, entry["n_heavy"]), flush=True)

    if not done and not a.merge:
        raise SystemExit("nothing could be annotated; see the reasons above")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(ann, f)
    print("\nwrote %s: %d peptide%s annotated now, %d in the file"
          % (a.out, done, "" if done == 1 else "s", len(ann)))
    if kept:
        print("%d already had an annotation and were left as they were." % kept)
    if skipped:
        print("\n%d skipped:" % len(skipped), file=sys.stderr)
        for pid, why in skipped:
            print("  %-12s %s" % (pid, why), file=sys.stderr)
    print("\nPoint the feature build at it with CHEDDAR_ANNOT=%s, and remember that "
          "the caches in cache/ were built with the old annotation: delete them so "
          "they are rebuilt." % a.out)


if __name__ == "__main__":
    main()
