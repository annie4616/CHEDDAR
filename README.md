# CHEDDAR: Cyclic-Peptide Heterogeneous-Edge Graphs with Dual-Solvent Ensembles for Permeability Prediction

Reference implementation of CHEDDAR. Training reproduces every number in the paper.

Each peptide becomes two heterogeneous atom graphs, one per explicit-solvent MD
ensemble (water and hexane). The graphs share atoms and covalent bonds and differ in
ensemble geometry and in their 3D contact edges. Features are statistics over 100
frames rather than per-frame coordinates; occupancy-selected contact edges carry a
separate message function from covalent bonds; one weight-shared encoder reads both
solvents under a solvent token, and the two predictions are averaged at inference.

## Layout

```
cheddar/
  data.py          Sec. 3.2 cache reader, App. E standardization, Eq. (15)
  features.py      Sec. 3.2 cache construction and split/label IO,
                   Eq. (3)-(5) and (6)-(7)
  model.py         Sec. 3.3, Eq. (8)-(12)
  train.py         Sec. 3.4, Eq. (13), App. F protocol, Eq. (16)
  eval.py          score a released checkpoint, no training
  sweep.py         3 splits x 5 seeds
configs/           the released model
ckpts/             the 15 checkpoints behind Table 1, cheddar-<split>-s<seed>.pt
data/              label table, atom annotation, and the three split files
LICENSE            MIT
```

## Install

```bash
conda env create -f environment.yml && conda activate cheddar
```

Reproducing the published weights bit-for-bit requires the pinned PyTorch version,
since deterministic kernels can change between releases.

## Data

The label table and the splits ship with this repository:

```
data/CycPeptMPDB-4D_new_imhb.csv       CycPeptMPDB_ID, SMILES, PAMPA
data/{ID,OD,Cliff}_split.csv           CycPeptMPDB_ID, partition (train/val/test)
data/nal_fix.json                      95 corrected naphthylalanine SMILES
```

`nal_fix.json` corrects 95 peptides whose table SMILES is 2-naphthylalanine while the
deposited MD geometry is 1-naphthylalanine. The builder applies it automatically when
it sits beside the label CSV and refuses to build without it; the atom features of
those 95 molecules otherwise disagree with their own trajectory.

The MD trajectories do not; they are 1.7 GB and are published separately as
CycPeptMPDB-4D, 100 frames per peptide in each of water and hexane:

```
https://zenodo.org/records/21274695          record 21274695, the Angstrom-unit release
https://github.com/Gobliu/CycPeptMPDB-4D
```

Use that record specifically. An earlier release of the same dataset stores
coordinates in nanometres, and the contact cutoff of Eq. (4) is an absolute 5 A, so
nanometre input would select every atom pair rather than the contacts. `features.py`
checks the units and refuses to build from the wrong release.

Point the builder at the unpacked trajectories and the per-molecule atom annotation:

```bash
export CHEDDAR_TRAJ=/path/to/traj CHEDDAR_ANNOT=atom_annot.pkl \
       CHEDDAR_CSV=data/CycPeptMPDB-4D_new_imhb.csv
```

A missing cache is then built on first use, or explicitly:

```python
from cheddar.features import build_cache
build_cache("cache/ensemble_Water_c100.pkl", annot, traj, csv_path, solvent="Water")
```

Two passes per solvent: `--select occupancy` builds the edge set the model uses,
Eq. (4); `--select meandist` builds the comparison set of Eq. (6). The occupancy set
is kept, with the edges the mean criterion would have missed flagged as `m_ij`, which
is what the edge dropout of Eq. (7) acts on.

## Running

```bash
python -m cheddar.train configs/cheddar.json --data-root $DATA_ROOT \
       --split ID --seed 1 --out runs/cheddar-ID-s1

python -m cheddar.sweep configs/cheddar.json --data-root $DATA_ROOT   # 15 runs
```

The released checkpoints are scored without retraining:

```bash
python -m cheddar.eval configs/cheddar.json --data-root $DATA_ROOT --all
python -m cheddar.eval configs/cheddar.json --data-root $DATA_ROOT \
       --ckpt ckpts/cheddar-ID-s1.pt --out preds.csv
```

`--all` prints one line per run and then mean and standard deviation per split,
which is the CHEDDAR row of Table 1: ID 0.145 +- 0.003, OOD 0.236 +- 0.018, cliff
0.142 +- 0.003. The standardization is refitted on the train partition of each
split, so evaluation never touches test statistics.

Each run writes `test_metrics.json`, `history.json`, `predictions.csv` and `best.pt`
into `runs/<config>-<split>-s<seed>/`. Report mean and standard deviation over seeds
1, 2, 7, 13, 42, on the original log P_app scale. Validation curves are in the
normalized space of Eq. (15); multiply by 4 to compare with test MSE.

