"""Per-solvent ensemble features for one MD trajectory set (Section 3.1).

Builds, for every molecule and one solvent, the heterogeneous atom graph of
Table 3: chemical identity and ensemble-geometry statistics on the nodes, bond
edges, and 3D contact edges selected either by contact occupancy (Eq. 3) or by
mean distance (Eq. 10).

NODE, chemical block c_i in R^46
  element one-hot        C N O S F Cl Br I P other
  heavy-atom degree      one-hot 0-5
  total H count          one-hot 0-4
  formal charge          one-hot -2..+2
  hybridization          one-hot sp sp2 sp3 sp3d sp3d2 other
  aromaticity            aromatic, in-ring
  ring                   size one-hot 3,4,5,6,7, >7, in macrocycle (ring >= 8)
  CIP chirality          one-hot none/R/S
  H-bonding              RDKit donor, acceptor
NODE, ensemble geometry block g_i in R^12, Eq. (1)-(2)
  bond length            mean, sd over neighbours, sd over frames
  bond angle             the same three operations on cos(theta)
  torsion                pooled cos and sin, resultant length R, circular sd,
                         and the mean over neighbours of the per-frame circular sd
  degree                 number of bonded heavy-atom neighbours

BOND EDGE e^b in R^9
  type one-hot (single/double/triple/aromatic), conjugated, in ring, amide,
  both atoms in a macrocycle, and the ensemble-mean bond length.

3D CONTACT EDGE e^3D in R^40, Eq. (5)
  distance expanded in 24 Gaussians over [0,6] A, a seven-way topological
  indicator for 3..8 bonds and >= 9, an H-bond-candidate flag, the scaled
  separation min(tau,20)/20, the four distance statistics, and the three
  occupancy fractions p_5.0, p_3.5, p_4.5.

The space-by-topology pair is the point: a transannular hydrogen bond is "2 A
apart in space, 8 bonds apart in topology", and no internal coordinate can
express that conjunction. A single is-bond flag cannot either, which is why the
two relations carry separate feature blocks and separate message functions.

Element, bond, stereo and ring facts come from RDKit on the SMILES, which is the
single source of truth for topology; geometry comes from the trajectory. Atom
indexing is shared between the molecular graph and both solvent ensembles.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import pickle

import numpy as np
from joblib import Parallel, delayed

ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]
HYB = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]
RING_SIZES = [3, 4, 5, 6, 7]
RBF_N, RBF_LO, RBF_HI = 24, 0.0, 6.0
TOPO_CAP = 8
SPACE_CUTOFF = 5.0

NODE_CHEM = (
    ["elem_" + e for e in ELEMENTS] + ["elem_other"]
    + ["deg_%d" % d for d in range(6)]
    + ["nH_%d" % d for d in range(5)]
    + ["chg_%+d" % c for c in (-2, -1, 0, 1, 2)]
    + ["hyb_" + h for h in HYB] + ["hyb_other"]
    + ["aromatic", "in_ring"]
    + ["ring%d" % r for r in RING_SIZES] + ["ring_gt7", "in_macrocycle"]
    + ["cip_none", "cip_R", "cip_S"]
    + ["hbd", "hba"]
)
NODE_GEOM = ["bl_mean", "bl_nbr_sd", "bl_frame_sd",
             "ba_cos_mean", "ba_cos_nbr_sd", "ba_cos_frame_sd",
             "tor_cos_mean", "tor_sin_mean", "tor_R", "tor_circsd",
             "tor_frame_sd", "n_bonds"]
# E/Z bond-stereo channels are omitted (Appendix A): every bond in this
# dataset's SMILES is STEREONONE, so the channels would be constant.
# Atom-level CIP chirality is non-degenerate and is kept on the node.
BOND_COLS = (["b_single", "b_double", "b_triple", "b_aromatic",
              "b_conjugated", "b_in_ring", "b_amide", "b_backbone", "b_length"])
SPACE_COLS = (["rbf_%d" % i for i in range(RBF_N)]
              + ["topo_%d" % t for t in range(3, TOPO_CAP + 1)] + ["topo_far"]
              + ["hbond_cand", "seq_sep", "d_mean", "d_sd", "d_min", "d_max",
                 "p_contact", "p_35", "p_45"])


def _onehot(val, options):
    v = [0.0] * (len(options) + 1)
    v[options.index(val) if val in options else len(options)] = 1.0
    return v


def _dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.maximum(np.linalg.norm(b1, axis=-1, keepdims=True), 1e-9)
    v = b0 - (b0 * b1).sum(-1, keepdims=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdims=True) * b1
    return np.arctan2((np.cross(b1, v) * w).sum(-1), (v * w).sum(-1))


# ---------------------------------------------------------------- split / label IO
def read_split(path, partition):
    """One row per molecule, columns CycPeptMPDB_ID and partition."""
    return [str(r["CycPeptMPDB_ID"]) for r in csv.DictReader(open(path))
            if r["partition"] == partition]


def read_labels(csv_path: str | Path, col: str = "PAMPA") -> dict[str, float]:
    out = {}
    for r in csv.DictReader(open(csv_path)):
        v = r.get(col)
        if v not in (None, "", "NA", "nan"):
            try:
                out[str(r["CycPeptMPDB_ID"])] = float(v)
            except ValueError:
                pass
    return out


def _node_chem(mol, order):
    ri = mol.GetRingInfo()
    macro = set()
    for ring in ri.AtomRings():
        if len(ring) >= 8:
            macro.update(ring)
    cip = {}
    for a in mol.GetAtoms():
        cip[a.GetIdx()] = a.GetPropsAsDict().get("_CIPCode", "none")
    hbd = set()
    hba = set()
    for a in mol.GetAtoms():
        if a.GetSymbol() in ("N", "O") and a.GetTotalNumHs() > 0:
            hbd.add(a.GetIdx())
        if a.GetSymbol() in ("N", "O") and a.GetFormalCharge() <= 0:
            hba.add(a.GetIdx())
    rows = []
    for idx in order:
        a = mol.GetAtomWithIdx(int(idx))
        r = []
        r += _onehot(a.GetSymbol(), ELEMENTS)
        r += _onehot(min(a.GetDegree(), 5), list(range(6)))[:-1]
        r += _onehot(min(a.GetTotalNumHs(), 4), list(range(5)))[:-1]
        r += _onehot(int(np.clip(a.GetFormalCharge(), -2, 2)), [-2, -1, 0, 1, 2])[:-1]
        r += _onehot(str(a.GetHybridization()), HYB)
        r += [float(a.GetIsAromatic()), float(a.IsInRing())]
        rs = [1.0 if a.IsInRingSize(s) else 0.0 for s in RING_SIZES]
        r += rs + [1.0 if (a.IsInRing() and not any(rs)) else 0.0,
                   1.0 if int(idx) in macro else 0.0]
        r += _onehot(cip[int(idx)], ["none", "R", "S"])[:-1]
        r += [1.0 if int(idx) in hbd else 0.0, 1.0 if int(idx) in hba else 0.0]
        rows.append(r)
    return np.asarray(rows, dtype=np.float32)


def _topo_dist(mol, n, order):
    from rdkit import Chem
    full = Chem.GetDistanceMatrix(mol)
    sub = full[np.ix_(order, order)]
    return sub


_UNITS_OK = []


def _check_units(C, path):
    """Refuse nanometre input. The cutoff of Eq. (4) is an absolute 5 A, so a
    nanometre release of CycPeptMPDB-4D would select every pair, silently."""
    if _UNITS_OK:
        return
    P = np.asarray(C[0], float)
    d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(len(P), 1)
    nn = float(np.median(np.sort(d[iu])[:len(P)]))
    if nn < 0.6:
        raise ValueError(
            "%s looks like nanometres (median nearest-neighbour distance %.3f; "
            "a C-C bond is 1.5 A). Use the Angstrom release of CycPeptMPDB-4D, "
            "https://zenodo.org/records/21274695" % (path, nn))
    _UNITS_OK.append(nn)


PDB_TAG = {"Water": "H2O", "Hexane": "Hexane"}


def find_traj(traj_dir, pid, solvent):
    """Locate one peptide's trajectory in either accepted layout.

    The public CycPeptMPDB-4D release (https://zenodo.org/records/21274695) unpacks to
    <Solvent>/Trajectories/<Source>_<pid>_<H2O|Hexane>_Traj.pdb, 100 MODELs each. The
    internal layout is <pid>_<solvent>_traj.npz. Both are read; nothing else is.
    """
    npz = os.path.join(traj_dir, "%s_%s_traj.npz" % (pid, solvent))
    if os.path.exists(npz):
        return npz
    hit = glob.glob(os.path.join(traj_dir, solvent, "Trajectories",
                                 "*_%s_%s_Traj.pdb" % (pid, PDB_TAG[solvent])))
    return hit[0] if hit else None


def read_frames(path):
    """Frames of one trajectory as (n_frames, n_atoms, 3) in Angstrom."""
    if path.endswith(".npz"):
        return np.load(path, allow_pickle=True)["coords"]
    frames, cur = [], []
    with open(path) as fh:
        for ln in fh:
            if ln.startswith(("ATOM", "HETATM")):
                cur.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
            elif ln.startswith("ENDMDL") and cur:
                frames.append(cur)
                cur = []
    if cur:
        frames.append(cur)
    return np.asarray(frames, np.float32)


def _one(pid, ann, nconf, traj_dir, smiles, solvent="Water", select="occupancy", contact_p=0.05):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    per = ann["per_atom"]
    loc = np.array([a["npz_abs_idx"] for a in per])
    rdk = np.array([a.get("rdkit_idx", i) for i, a in enumerate(per)])
    n = len(per)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    heavy = [a.GetIdx() for a in mol.GetAtoms()]
    if len(heavy) != n:
        return None
    order = list(range(n))        # per_atom is already RDKit/SMILES order

    f = find_traj(traj_dir, pid, solvent)
    if f is None:
        return None
    C = read_frames(f)
    _check_units(C, f)
    if loc.max() >= C.shape[1]:
        return None
    sel = np.linspace(0, len(C) - 1, min(nconf, len(C))).astype(int)
    P = C[sel].astype(np.float64)[:, loc, :]
    D = np.sqrt(((P[:, :, None, :] - P[:, None, :, :]) ** 2).sum(-1))
    dmean = D.mean(0)

    # ---- bonds from RDKit (single source of truth), not a distance threshold
    bond_of = {}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond_of[(i, j)] = bond_of[(j, i)] = b
    nbrs = [[] for _ in range(n)]
    for (i, j) in bond_of:
        if j not in nbrs[i]:
            nbrs[i].append(j)

    # ---- node: chemistry + ensemble geometry
    chem = _node_chem(mol, order)
    geom = np.zeros((n, len(NODE_GEOM)), dtype=np.float64)
    for i in range(n):
        nb = np.array(nbrs[i], dtype=int)
        geom[i, 11] = len(nb)
        if len(nb) == 0:
            continue
        bl = D[:, i, nb]
        geom[i, 0] = bl.mean()
        geom[i, 1] = bl.mean(0).std() if len(nb) > 1 else 0.0
        geom[i, 2] = bl.std(0).mean()
        if len(nb) >= 2:
            cs = []
            for a_ in range(len(nb)):
                for b_ in range(a_ + 1, len(nb)):
                    u = P[:, nb[a_], :] - P[:, i, :]
                    w = P[:, nb[b_], :] - P[:, i, :]
                    u /= np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1e-9)
                    w /= np.maximum(np.linalg.norm(w, axis=-1, keepdims=True), 1e-9)
                    cs.append((u * w).sum(-1))
            cs = np.stack(cs, 1)
            geom[i, 3] = cs.mean()
            geom[i, 4] = cs.mean(0).std() if cs.shape[1] > 1 else 0.0
            geom[i, 5] = cs.std(0).mean()
        tc, ts = [], []
        for j in nbrs[i]:
            for k in nbrs[i]:
                if k == j:
                    continue
                for l in nbrs[j]:
                    if l in (i, k):
                        continue
                    ang = _dihedral(P[:, k, :], P[:, i, :], P[:, j, :], P[:, l, :])
                    tc.append(np.cos(ang)); ts.append(np.sin(ang))
                    break
                break
        if tc:
            TC = np.stack(tc, 1); TS = np.stack(ts, 1)
            cm, sm = TC.mean(), TS.mean()
            R = float(np.sqrt(cm * cm + sm * sm))
            geom[i, 6], geom[i, 7], geom[i, 8] = cm, sm, R
            geom[i, 9] = float(np.sqrt(max(-2.0 * np.log(max(R, 1e-12)), 0.0)))
            Rf = np.sqrt(TC.mean(0) ** 2 + TS.mean(0) ** 2)
            geom[i, 10] = float(np.sqrt(np.maximum(-2.0 * np.log(np.maximum(Rf, 1e-12)), 0.0)).mean())
    node = np.hstack([chem, geom.astype(np.float32)]).astype(np.float32)

    # ---- edges, two relations
    topo = _topo_dist(mol, n, order)
    ri = mol.GetRingInfo()
    macro_atoms = set()
    for ring in ri.AtomRings():
        if len(ring) >= 8:
            macro_atoms.update(ring)

    bi, bj, bfeat = [], [], []
    for (i, j), b in bond_of.items():
        if i >= j:
            continue
        bt = str(b.GetBondType())
        row = [float(bt == "SINGLE"), float(bt == "DOUBLE"), float(bt == "TRIPLE"),
               float(bt == "AROMATIC"), float(b.GetIsConjugated()), float(b.IsInRing())]
        a1, a2 = mol.GetAtomWithIdx(i), mol.GetAtomWithIdx(j)
        amide = 0.0
        for (x, y) in ((a1, a2), (a2, a1)):
            if x.GetSymbol() == "N" and y.GetSymbol() == "C":
                if any(nb.GetSymbol() == "O" and
                       str(mol.GetBondBetweenAtoms(y.GetIdx(), nb.GetIdx()).GetBondType()) == "DOUBLE"
                       for nb in y.GetNeighbors()):
                    amide = 1.0
        row += [amide, float(i in macro_atoms and j in macro_atoms)]
        row += [float(dmean[i, j])]
        for (x, y) in ((i, j), (j, i)):
            bi.append(x); bj.append(y); bfeat.append(row)

    # 3D contact edges: radius graph, 1-2 and 1-3 pairs excluded (tau >= 3)
    hbd = set(); hba = set()
    for a in mol.GetAtoms():
        if a.GetSymbol() in ("N", "O") and a.GetTotalNumHs() > 0:
            hbd.add(a.GetIdx())
        if a.GetSymbol() in ("N", "O") and a.GetFormalCharge() <= 0:
            hba.add(a.GetIdx())
    centers = np.linspace(RBF_LO, RBF_HI, RBF_N)
    width = (RBF_HI - RBF_LO) / RBF_N
    si, sj, sfeat = [], [], []
    # Edge selection. occupancy (Eq. 3) keeps a pair seen within 5 A in at
    # least CONTACT_P of frames; meandist (Eq. 10) keeps a pair whose MEAN
    # distance is under 5 A. The occupancy set is a superset in practice and
    # the difference is what the edge dropout of Eq. (7) acts on. Selecting on
    # occupancy rather than mean distance keeps transient approaches that a mean
    # would miss (a pair at 2 A in only 10 % of frames).
    pcon = (D < SPACE_CUTOFF).mean(0)          # omega_ij(5.0) of Eq. (3)
    if select == "occupancy":
        cand = np.argwhere((pcon >= contact_p) & (topo >= 3))     # Eq. (4)
    else:
        cand = np.argwhere((dmean < SPACE_CUTOFF) & (topo >= 3))  # Eq. (6)
    for i, j in cand:
        i, j = int(i), int(j)
        d = D[:, i, j]
        dm = float(d.mean())
        row = list(np.exp(-((dm - centers) ** 2) / (2 * width ** 2)))
        t = int(min(topo[i, j], TOPO_CAP + 1))
        row += [1.0 if t == k else 0.0 for k in range(3, TOPO_CAP + 1)]
        row += [1.0 if t > TOPO_CAP else 0.0]
        hb = 1.0 if (dm < 3.5 and ((i in hbd and j in hba) or (j in hbd and i in hba))) else 0.0
        row += [hb, float(min(topo[i, j], 20)) / 20.0,
                dm, float(d.std()), float(d.min()), float(d.max()),
                float(pcon[i, j]),
                float((d < 3.5).mean()), float((d < 4.5).mean())]
        si.append(i); sj.append(j); sfeat.append(row)

    if not si:
        return None
    return pid, {
        "node": node,
        "bond_index": np.asarray([bi, bj], dtype=np.int32),
        "bond_attr": np.asarray(bfeat, dtype=np.float32),
        "space_index": np.asarray([si, sj], dtype=np.int32),
        "space_attr": np.asarray(sfeat, dtype=np.float32),
        "n_atoms": n,
    }


def build_features(annot, traj, csv_path, solvent="Water", select="occupancy",
                   conformers=100, contact_p=0.05, jobs=32, smiles_fix=None, limit=0):
    """One feature pass. Returns {"meta": ..., "feat": ...}; see Eq. (4) and (6)."""
    import csv as _csv
    smi = {r["CycPeptMPDB_ID"]: r["SMILES"] for r in _csv.DictReader(open(csv_path))}
    if smiles_fix and os.path.exists(smiles_fix):
        for p_, v_ in json.load(open(smiles_fix))["fixed_smiles"].items():
            if p_ in smi:
                smi[p_] = v_
    with open(annot, "rb") as f:
        ann = pickle.load(f)
    pids = sorted(p for p in ann if str(p) in smi)
    if limit:
        pids = pids[:limit]
    missing = [p for p in pids if find_traj(traj, p, solvent) is None]
    if len(missing) == len(pids):
        raise FileNotFoundError(
            "no %s trajectories found in %s, in either accepted layout\n"
            "  <Solvent>/Trajectories/<Source>_<id>_<H2O|Hexane>_Traj.pdb   (the public release)\n"
            "  <id>_<solvent>_traj.npz\n"
            "The ensemble features of Sec. 3.2 are statistics over the MD frames; "
            "the label CSV supplies SMILES and PAMPA only and cannot substitute for them."
            % (solvent, traj))
    res = Parallel(n_jobs=jobs, verbose=1)(
        delayed(_one)(p, ann[p], conformers, traj, smi[str(p)], solvent, select, contact_p)
        for p in pids)
    feat = {r[0]: r[1] for r in res if r}
    one = next(iter(feat.values()))
    meta = {"node_cols": NODE_CHEM + NODE_GEOM,
            "bond_cols": BOND_COLS, "space_cols": SPACE_COLS,
            "node_dim": one["node"].shape[1],
            "bond_dim": one["bond_attr"].shape[1],
            "space_dim": one["space_attr"].shape[1],
            "conformers": conformers, "cutoff": SPACE_CUTOFF,
            "select": select, "contact_p": contact_p,
            "n_molecules": len(feat), "solvent": solvent}
    assert one["node"].shape[1] == len(NODE_CHEM) + len(NODE_GEOM)
    assert one["bond_attr"].shape[1] == len(BOND_COLS)
    assert one["space_attr"].shape[1] == len(SPACE_COLS)
    return {"meta": meta, "feat": feat}


def merge_edge_sets(occ, mean):
    """Flag the occupancy edges the mean-distance criterion would have missed (m_ij, Eq. 11)."""
    assert occ["meta"]["space_cols"] == mean["meta"]["space_cols"], "column mismatch"
    out = {}
    for p in sorted(set(occ["feat"]) & set(mean["feat"])):
        go, gm = occ["feat"][p], mean["feat"][p]
        keep = {tuple(x) for x in gm["space_index"].T.tolist()}
        g = dict(go)
        g["trans_mask"] = np.array(
            [tuple(x) not in keep for x in go["space_index"].T.tolist()], np.bool_)
        out[p] = g
    meta = dict(occ["meta"]); meta["n_molecules"] = len(out)
    return {"meta": meta, "feat": out}


def build_cache(out_path, annot, traj, csv_path, solvent="Water",
                conformers=100, contact_p=0.05, jobs=32, smiles_fix=None,
                select="occupancy"):
    """Build one solvent cache end to end and write it to out_path.

    select="occupancy" is the released model: contact edges chosen by Eq. (4),
    carrying m_ij of Eq. (7) so the edge dropout has something to act on.
    select="meandist" builds the mean-distance set of Eq. (6) on its own, which
    is the edge rule of the "w/o ensemble geometry" arm of Table 2.
    """
    if select == "meandist":
        blob = build_features(annot, traj, csv_path, solvent, "meandist",
                              conformers, contact_p, jobs, smiles_fix)
    else:
        occ = build_features(annot, traj, csv_path, solvent, "occupancy",
                             conformers, contact_p, jobs, smiles_fix)
        mean = build_features(annot, traj, csv_path, solvent, "meandist",
                              conformers, contact_p, jobs, smiles_fix)
        blob = merge_edge_sets(occ, mean)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(blob, f)
    print("built %s: %d molecules" % (out_path, len(blob["feat"])), flush=True)
    return blob
