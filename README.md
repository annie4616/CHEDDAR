# CHEDDAR

Cyclic-peptide membrane permeability from solvent-dependent conformational ensembles.

Each peptide becomes two atom graphs, one per explicit-solvent MD ensemble (water and
hexane). Atom features are statistics over 100 frames rather than one structure, contact
edges are selected by how often a pair stays in contact, and the two solvent predictions are
averaged. The model has 554,561 parameters, trains in about 31 min on one GPU and predicts
in 2.3 ms per peptide.

## Cyclic peptide permeability prediction

A cyclic peptide is a short peptide whose backbone is closed into a ring. The class combines
broad interaction surfaces with conformational constraints that can improve resistance to
proteolysis, which makes it attractive for targets that conventional small molecules struggle
to address. Many cyclic peptides, however, have limited passive membrane permeability, which
restricts access to intracellular targets and contributes to poor oral absorption. Predicting
permeability is therefore a useful step in identifying promising candidates, particularly for
scaffolds beyond those already measured.

Permeability here depends on chemical composition and on whether polar groups are exposed or
shielded, which can change substantially with stereochemistry. Solvent-dependent
conformational preferences allow a molecule to expose polar groups in the aqueous environment
around the membrane while favouring intramolecular hydrogen bonding and polar-group burial in
its nonpolar core, a behaviour termed molecular chameleonicity. A representation of the
molecule therefore has to cover both environments and describe the conformational
distribution within each. CHEDDAR does this by reading independently sampled explicit-solvent
ensembles in water and in hexane, and predicts `log P_app` as measured by a PAMPA assay.

## Architecture

![The CHEDDAR architecture](docs/architecture.png)

A peptide enters as two graphs, one per solvent, sharing nodes and bonds and differing only
in the geometry read off the water and hexane ensembles and in the solvent token carried on
every node. Atom features combine a chemical block, an ensemble-geometry block and that
token; bonded and 3D-contact relations carry disjoint feature blocks and separate encoders
and message MLPs, so their aggregates reach the node update without being merged. Pooling
concatenates the sum, mean and maximum statistics and leaves their weighting to the head,
which is bounded to the normalized label range. The two views are never compared inside the
model -- each contributes its own loss term -- and are averaged only at inference.

## Table of Contents

1. [Setup](#setup)
2. [Datasets](#datasets)
3. [Inference](#inference)
4. [Training and Evaluation](#training-and-evaluation)
5. [Ablations](#ablations)
6. [Repository Layout](#repository-layout)
7. [License](#license)

* * *

## Setup

1. Install Miniconda or Anaconda. A GPU is needed only for training; prediction and
   evaluation run on a CPU.

2. Create the environment:

```bash
conda env create -f environment.yml
conda activate cheddar
```

3. Confirm what is in place:

```bash
python -m cheddar check
```

This reports the installed versions, the visible GPUs and every data file it located,
without computing anything.

```
CHEDDAR in /path/to/CHEDDAR

  torch    2.8.0+cu126
  numpy    1.26.4
  rdkit    2023.9.6
  joblib   1.5.1
  device   1 cuda gpu

  label table      found   data/CycPeptMPDB-4D_new_imhb.csv (2 MB)
  atom annotation  found   data/atom_annot.pkl (30 MB)
  SMILES fix       found   data/nal_fix.json (0 MB)
  water cache      MISSING cache/ensemble_Water_c100.pkl
  hexane cache     MISSING cache/ensemble_Hexane_c100.pkl
  splits           found   3 of 3 in data/
  checkpoints      found   15 of 15
  trajectories     not on this machine yet
```

A fresh clone looks like this: everything tracked by git is present, and the two caches and
the trajectories are not. `python -m cheddar data` supplies the trajectories and the first
command that reads them builds the caches.

> `environment.yml` pins `torch==2.8.0+cu126` and the other exact versions every number in
> the paper was produced with.

* * *

## Datasets

The label table (4,925 peptides with a measured PAMPA value), the atom annotation and the
three split files ship with the repository under `data/`. The MD trajectories do not.

```bash
python -m cheddar data
```

This searches the customary locations for an unpacked copy, downloads and unpacks the
Zenodo record if none is found, resumes an interrupted download and verifies the published
MD5. About 12.5 GB of free disk is needed while unpacking and 10.2 GB thereafter.

| variable | meaning |
| --- | --- |
| `CHEDDAR_TRAJ` | a copy in a non-standard location: the directory holding `Water/` and `Hexane/` |
| `CHEDDAR_NO_DOWNLOAD=1` | forbid all network access |

> Skipping this step does not report missing data. The first command that reads the
> trajectories starts the 1.6 GB download by itself, which is worth knowing before a command
> that appears to hang.

### CycPeptMPDB-4D (`https://zenodo.org/records/21274695`)

1.6 GB compressed, 10.2 GB unpacked. Read exactly as it unpacks, from
`Water/Trajectories/*_Traj.pdb` and the corresponding hexane path, so no conversion is
needed. The release holds 5,160 peptides; the model uses the 4,925 that have an atom
annotation here, and the remaining 235 are skipped, which is expected.

An earlier release of the same data stores coordinates in nanometres, under which the 5 Å
contact cutoff would make every atom pair a contact. The builder verifies the units and
declines to proceed rather than producing silently incorrect features.

### Feature caches (`cache/ensemble_{Water,Hexane}_c100.pkl`)

Built on first use from the trajectories, about 72 s each on 32 cores, together occupying
1.9 GB. Later commands reuse them and start in seconds. They are derived files and may be
deleted at any time; a rebuild reproduces every feature array bit for bit, so no result
depends on keeping a particular copy.

### The SMILES correction (`data/nal_fix.json`)

Corrects 95 peptides whose tabulated SMILES says 2-naphthylalanine while the deposited
geometry is 1-naphthylalanine. Applied automatically; without it those 95 receive atom
features inconsistent with their own trajectory.

* * *

## Inference

### Peptides in CycPeptMPDB-4D

```bash
./predict.sh 5222
```

`5222` is a CycPeptMPDB identifier, any number of which may be given. With no arguments a
worked example runs. Each prediction averages the two solvent views, then the five released
seeds of one split.

```
id          logPapp     sd   measured partition
5222          -5.43   0.05      -5.25 test

5222: predicted log P_app -5.43, above average permeability for a cyclic peptide
   -- more permeable than 68% of the 4925 measured ones.
   measured -5.25, so this prediction is 0.18 log units off.
```

`./predict.sh` wraps `python -m cheddar predict` and takes the same flags:

| flag | default | meaning |
| --- | --- | --- |
| `--split` | `ID` | which released model to use: `ID`, `OOD` or `Cliff` |
| `--seeds` | `1,2,7,13,42` | which of the five seeds to average |
| `--partition` | — | predict a whole partition: `train`, `val` or `test` |
| `--all` | — | every peptide in the cache |
| `--out` | — | write a csv here |
| `--ckpt-dir` | `ckpts` | where the checkpoints live |
| `--device` | cuda if present | `cpu` to force the CPU |

Reading the output:

- `logPapp` is the base-10 logarithm of the apparent permeability coefficient in cm/s, as
  measured by a PAMPA assay. It is negative throughout, and **less negative indicates greater
  permeability**: −5.0 crosses readily, −7.0 poorly. Each prediction is additionally ranked
  against the 4,925 measured peptides, that being the only scale the model was fitted on.
- `sd` is the spread across the five models. A small value means they agree; a large one
  means the peptide falls where they disagree, and the prediction should be weighted
  accordingly. For how accurate these averaged predictions are in general, see the
  `ensemble` row under [Training and Evaluation](#training-and-evaluation).
- `partition` says whether the peptide was in the training set of the model used. `train`
  means the figure is in-sample, so optimistic. `--split OOD` selects a model that has not
  encountered the scaffold:

```bash
./predict.sh 4705 --split OOD
```

### Peptides of your own

The model reads MD ensembles, not SMILES, so a peptide that is not in
CycPeptMPDB-4D needs **its own water and hexane trajectories first**. There is no
path from a SMILES string alone. Given those trajectories, four steps:

**1. Place the trajectories** where the feature pass looks for them, in either accepted
layout:

```
my_data/Water/Trajectories/MD_MYPEP1_H2O_Traj.pdb
my_data/Hexane/Trajectories/MD_MYPEP1_Hexane_Traj.pdb
my_data/MYPEP1_Water_traj.npz          # or this layout, holding a 'coords' array
```

The released peptides have 100 frames each, evenly spaced over the production run, and the
features are statistics over them. Fewer frames are accepted -- every frame present is used
-- but the dispersion columns are what the ensemble contributes, and they are only as
meaningful as the sampling behind them.

**2. Add the peptides to the label table.** Copy the released csv and append one row per
peptide, with an id of your own and its SMILES; leave `PAMPA` empty, as that is what is
being predicted. The released rows have to stay, because the standardization of Appendix E
is fitted on the training partition and those peptides must still be in the cache.

```bash
cp data/CycPeptMPDB-4D_new_imhb.csv my_data/labels.csv
# then append:  MYPEP1,CC(C)[C@H]1NC(=O)...,
```

**3. Build the atom annotation.** The released `data/atom_annot.pkl` covers the 4,925
peptides of the public release only, and the feature pass skips anything without an entry.
This derives one from the SMILES and the first frame:

```bash
python -m cheddar annotate --traj my_data \
                           --csv my_data/labels.csv \
                           --merge data/atom_annot.pkl \
                           --out my_data/annot.pkl \
                           --ids MYPEP1
```

`--merge` keeps the released entries and adds yours; `--ids` limits the work to the
peptides named, which saves walking the other 4,925 rows. What an entry supplies is the
permutation between RDKit's atom order and the trajectory's, which no trajectory file
records; the match is made on the molecular graph, and a peptide whose SMILES and frame
disagree is reported rather than annotated. An `npz` trajectory needs an `elements` array
beside its `coords` for this step, since a bare coordinate array does not say which atom is
which; a pdb carries that in the file.

**4. Rebuild the caches and predict**, pointing the environment at your files:

```bash
export CHEDDAR_TRAJ=my_data
export CHEDDAR_ANNOT=my_data/annot.pkl
export CHEDDAR_CSV=my_data/labels.csv
export CHEDDAR_SMILES_FIX=data/nal_fix.json
rm -f cache/ensemble_{Water,Hexane}_c100.pkl
./predict.sh MYPEP1
```

The first command rebuilds both caches from every peptide in the new label table, which
takes as long as the original build. `CHEDDAR_SMILES_FIX` has to be set explicitly because
the correction is otherwise looked for beside the label csv, which is now yours.

Two things to expect. A prediction outside the trained window is clipped: the label is
trained on [−8, −4], so a value at either end means "beyond the trained range" rather than a
precise figure. And atoms that the molecular graph cannot tell apart, such as the two methyls
of a valine, may be assigned to each other's position; bond lengths are unaffected and the
per-atom effect on this dataset is around 0.05 Å, but an annotation built here is not
guaranteed identical to one built from the simulation's own atom order.

* * *

## Training and Evaluation

The fifteen models behind the paper ship in `ckpts/`, so the paper's main result can be
reproduced without training:

```bash
python -m cheddar eval     # equivalently, just: python -m cheddar
```

```
split    scoring    nmol seeds  MSE             R2              MAE
ID       per seed    492     5  0.145 +- 0.003  0.621 +- 0.007  0.261
         ensemble    492     5  0.138           0.638           0.252
OOD      per seed    490     5  0.236 +- 0.018  0.660 +- 0.026  0.372
         ensemble    490     5  0.219           0.685           0.360
Cliff    per seed    493     5  0.142 +- 0.003  0.683 +- 0.006  0.261
         ensemble    493     5  0.136           0.697           0.252
```

Each split is reported twice, because the five seeds are aggregated at different stages and
the two figures answer to different uses:

- **`per seed`** scores each checkpoint on its own and reports the mean with a standard
  deviation. The spread separates the effect of the method from seed variance, which is why
  the paper reports this row, and these are the figures to compare against other work.
- **`ensemble`** averages the five predictions into one per peptide and scores that once.
  Whatever each model gets wrong on its own partly cancels, worth 4–7 % lower error here and
  costing nothing since all five checkpoints ship. This is what `predict.sh` returns, so the
  row says how accurate a prediction from it is.

Seeds 1, 2, 7, 13 and 42, taking about three minutes on a CPU, most of it spent reading the
two caches. The three splits are the random (`ID`), Bemis–Murcko scaffold (`OOD`) and
activity-cliff (`Cliff`) partitions of Appendix C. `nmol` counts the peptides scored and
`seeds` the checkpoints averaged over.

### Training

Rebuilding the checkpoints, one run at a time or the whole grid. **A GPU is required here**,
unlike everywhere else on this page.

```bash
python -m cheddar train --split ID --seed 1     # one model, about 31 min
python -m cheddar sweep                         # all fifteen
```

Useful flags:

| flag | default | meaning |
| --- | --- | --- |
| `--split` | — | `ID`, `OOD` or `Cliff` (`train`) |
| `--seed` | — | 1, 2, 7, 13 or 42 (`train`) |
| `--splits`, `--seeds` | all | comma lists restricting the grid (`sweep`) |
| `--gpus` | every free GPU | comma list of cards to use (`sweep`) |
| `--max-workers` | one per GPU | concurrent runs (`sweep`) |
| `--config` | `configs/cheddar.json` | a different configuration |
| `--out`, `--out-root` | `runs/` | where results are written |
| `--dry-run` | — | list what would run, and stop (`sweep`) |

Outputs written to `runs/cheddar-<split>-s<seed>/`:

```
test_metrics.json   MSE, R2, MAE and the rest, on the test partition
history.json        per-epoch training and validation loss
predictions.csv     per-peptide label and prediction
best.pt             the selected checkpoint
```

- `sweep` uses only GPUs that are not in use by others, limits itself to 80 % of the
  available cores, and skips runs whose results already exist, so an interrupted sweep
  resumes where it stopped.
- Deterministic kernels are enabled, so a given seed on the same GPU architecture and
  PyTorch build reproduces its result exactly, including a byte-identical `best.pt`.
- On eight RTX 3090s, one card per run puts the fifteen in two waves and finishes in 69 min.
  All fifteen at once, two per card, took 51 min: 20–26 s per epoch, plus about 8 min at the
  start while fifteen processes read the same two 1 GB caches from one disk. A run peaks at
  0.72 GB of GPU memory, so sharing a card costs speed alone.

### How training works

**Objective.** Stochastic gradient descent on the measured permeabilities of the training
peptides, minimising mean squared error on the scaled label. One pass over the training set
is an epoch and the budget is fixed at 120, with no early stopping. No conformational search
or docking happens at training time; the ensembles are inputs, computed in advance.

**Partitioning and model selection.** Each split divides the peptides into train, validation
and test. Parameters are fitted on train, the checkpoint is selected by validation MSE on the
original log P_app scale, and that checkpoint is scored once on test, which is never
consulted during fitting or selection. The test figures are what Table 1 reports.

**Replication.** Initialisation and batch order depend on the seed, so each configuration is
run with five seeds and reported as a mean with its standard deviation, which separates the
effect of the method from seed variance. Reproducing Table 1 by retraining is a deterministic
replay on the same GPU architecture and PyTorch build rather than an independent
reimplementation, and it covers the CHEDDAR row alone: the seven baselines of Table 1 are not
included here.

* * *

## Ablations

`configs/cheddar.json` holds every hyperparameter of the released model. Copy it, edit it and
pass it with `--config`; the `abl_*.json` files are the arms of Table 2, so each row of that
table is one sweep.

| Table 2 row | command |
| --- | --- |
| water only | `python -m cheddar sweep --config configs/abl_water_only.json` |
| hexane only | `python -m cheddar sweep --config configs/abl_hexane_only.json` |
| w/o 3D-contact edges | `python -m cheddar sweep --config configs/abl_no_contact_edges.json` |
| w/o ensemble geometry | `python -m cheddar sweep --config configs/abl_no_ensemble_geometry.json` |

Runs are named after the configuration, for example `runs/abl_water_only-ID-s1/`. A single
arm of one split, rather than the whole sweep:

```bash
python -m cheddar train --config configs/abl_no_contact_edges.json --split ID --seed 1
```

Configuration keys:

| key | meaning | released |
| --- | --- | --- |
| `hidden`, `head_hidden` | width of the message passing and of the head | 128, 128 |
| `layers` | message passing layers | 3 |
| `dropout` | dropout in the encoder and head | 0.5 |
| `trans_drop` | p_drop of Eq. (7), dropout on transient contact edges | 0.2 |
| `lr`, `weight_decay`, `warmup_frac` | OneCycle peak, AdamW decay, warmup share | 6e-4, 0.1, 0.2 |
| `batch_size`, `epochs` | batch and the fixed epoch budget | 16, 120 |
| `edge_block` | `both`, or `bond` to drop the 3D contact relation | both |
| `both_solvents`, `solvent_token` | use two solvent views, append iota_s | true, true |
| `meanonly` | remove the dispersion columns outright, which changes the feature width and the parameter count | absent |
| `drop_space_cols` | remove named columns of the 3D contact relation outright, on top of `meanonly`; an unknown name raises rather than being ignored | absent |
| `cache`, `alt_cache` | the two solvent caches; a name containing `meanedge` is built with the mean-distance rule of Eq. (6) instead of occupancy | ensemble_{Water,Hexane} |

> The "w/o ensemble geometry" row takes the ensemble out of the input in all three places it
> enters, which is why `abl_no_ensemble_geometry.json` changes four keys rather than one.
> `meanonly` removes the seven node dispersion columns and `d_sd`, `d_min`, `d_max`;
> `drop_space_cols` removes the three occupancy columns `p_contact`, `p_35`, `p_45`; and the
> `meanedge` cache decides contact membership by mean distance (Eq. 6) rather than by
> occupancy, so dispersion is gone from the edge set itself. `trans_drop` is 0.0 because
> Eq. (7) drops transient contacts by occupancy, and this arm has no occupancy to drop.
> Node width falls from 59 to 52 and the 3D contact relation from 40 to 34, so this arm's
> parameter count differs from the reference.

* * *

## Repository Layout

```
predict.sh          single-command prediction, the entry point most readers require
cheddar/
  __init__.py       marks the package, empty
  __main__.py       the command line; `python -m cheddar --help` lists every command
  fetch.py          find the trajectories, or download the Zenodo record
  features.py       cache construction from MD frames, split and label IO
  data.py           cache reader, standardization, label scaling
  model.py          the network
  train.py          training loop and evaluation
  sweep.py          the 3 x 5 grid across free GPUs
  eval.py           score checkpoints without training
  predict.py        permeability for named peptides
  annotate.py       atom annotation for peptides of your own
configs/
  cheddar.json      every hyperparameter of the released model
  abl_*.json        the arms of Table 2
ckpts/              the 15 checkpoints behind the paper's main table
data/               label table, atom annotation, the three splits, a SMILES correction
cache/              built on first use from the trajectories; not shipped, not tracked
environment.yml     the exact versions every number in the paper was produced with
```

* * *

## License

Released under the MIT License; see [LICENSE](LICENSE).
