# CHEDDAR

Cyclic-peptide membrane permeability from solvent-dependent conformational ensembles.

- Each peptide becomes two atom graphs, one per explicit-solvent MD ensemble (water and hexane).
- Atom features are statistics over 100 frames, not one structure.
- Contact edges are selected by how often a pair stays in contact.
- The two solvent predictions are averaged.
- 554,561 parameters; about 31 min to train on one GPU; 2.3 ms per peptide to predict.

[Setup](#setup) ·
[Obtaining a prediction](#obtaining-a-prediction) ·
[Evaluating the model](#evaluating-the-model) ·
[Training the model](#training-the-model) ·
[Tuning and ablations](#tuning-and-ablations) ·
[Cyclic peptide permeability prediction](#cyclic-peptide-permeability-prediction) ·
[Architecture](#architecture) ·
[Repository contents](#repository-contents) ·
[Data](#data) ·
[License](#license)

## Setup

Do this once. Predicting, evaluating and training all read the same two feature caches,
which are built from MD trajectories that a fresh clone does not contain.

```bash
conda env create -f environment.yml && conda activate cheddar   # 1. install, about 3 min
python -m cheddar check                                         # 2. report what is present
python -m cheddar data                                          # 3. obtain the MD data, 1.6 GB
```

- **Step 2** lists which files are present and which are missing. It computes nothing.
- **Step 3** downloads 1.6 GB, unpacks it to 10.2 GB, and needs about 12.5 GB of free disk
  while doing so. It exits immediately if the trajectories are already there.
- The first command that reads the data builds the two caches, about 72 s each, printing
  progress. Later commands reuse them and start in seconds.
- **Skipping step 3 does not report missing data** — the 1.6 GB download begins by itself.
  Worth knowing before a command that looks like it has hung.
- A GPU is needed only for training. Predicting and evaluating run on a CPU.

## Obtaining a prediction

```bash
./predict.sh 5222
```

```
id          logPapp     sd   measured partition
5222          -5.43   0.05      -5.25 test

5222: predicted log P_app -5.43, above average permeability for a cyclic peptide
   -- more permeable than 68% of the 4925 measured ones.
   measured -5.25, so this prediction is 0.18 log units off.
```

- `5222` is a CycPeptMPDB identifier. Any number of them may be given.
- No arguments runs a worked example.
- `./predict.sh` wraps `python -m cheddar predict` and takes the same arguments:
  `--partition test` for a whole partition, `--all` for every peptide, `--out results.csv`
  to write a file.
- `python -m cheddar check` reports versions, the visible GPU and the data files found,
  without computing anything.

### Interpreting the output

- `logPapp` — base-10 logarithm of the apparent permeability coefficient in cm/s, from a
  PAMPA assay. Negative throughout, and **less negative means more permeable**: −5.0 crosses
  readily, −7.0 poorly.
- Each prediction is also ranked against the 4,925 measured peptides, since an absolute log
  unit is hard to place. That set is the only scale the model was fitted on.
- `sd` — spread across the five released models. Small means they agree; large means the
  peptide falls where they disagree, and the prediction should be weighted accordingly.
- Each prediction is the five averaged into one, as under
  [Evaluating the model](#evaluating-the-model).
- `partition` — whether the peptide was in the training set of the model used. `train` means
  the figure is in-sample, so optimistic. For a model that has not seen the scaffold:

```bash
./predict.sh 4705 --split OOD
```

## Evaluating the model

Reproduces the paper's main result by scoring the fifteen checkpoints in `ckpts/`. No
training required.

```bash
python -m cheddar eval     # equivalently, just: python -m cheddar
```

```
split       n  MSE              R2
ID          5  0.145 +- 0.003    0.621 +- 0.007
OOD         5  0.236 +- 0.018    0.660 +- 0.026
Cliff       5  0.142 +- 0.003    0.683 +- 0.006
```

- The CHEDDAR row of Table 1: mean and standard deviation over seeds 1, 2, 7, 13 and 42,
  followed by the same result in prose.
- `eval` is the default command, so it may be omitted. No arguments or paths needed.
- About three minutes, most of it reading the two caches. Runs on a CPU.
- The three splits are random (`ID`), Bemis–Murcko scaffold (`OOD`) and activity-cliff
  (`Cliff`), described in Appendix C.

Adding `--ensemble` averages the five seeds of each split into one prediction per peptide
and scores that prediction once, which is what `predict.sh` returns:

```bash
python -m cheddar eval --ensemble
```

```
split       n  seeds  MSE      R2       MAE
ID        492      5  0.138    0.638    0.252
OOD       490      5  0.219    0.685    0.360
Cliff     493      5  0.136    0.697    0.252
```

Retraining from scratch yields the same figures:

- `python -m cheddar sweep` rebuilds all fifteen models from random initialisation and
  reproduces the first table in both mean and standard deviation, each `best.pt` being
  byte-identical to the checkpoint shipped here.
- Two limits on that check: it is a deterministic replay on the same GPU architecture and
  PyTorch build, not an independent reimplementation; and it covers the CHEDDAR row alone,
  as the seven baselines of Table 1 are not in this repository.

## Training the model

Not required in order to use CHEDDAR — the fifteen models behind the paper are already in
`ckpts/`, and both `predict.sh` and `eval` use them directly. **A GPU is needed here**,
unlike everywhere else on this page.

One model, about 31 min on one GPU:

```bash
python -m cheddar train --split ID --seed 1
```

All fifteen, which is how the paper's table was produced:

```bash
python -m cheddar sweep
```

- `--split` is `ID`, `OOD` or `Cliff`; `--seed` is 1, 2, 7, 13 or 42. The fifteen models are
  every combination, and `sweep` runs all of them.
- A run writes `test_metrics.json`, `history.json`, `predictions.csv` and `best.pt` to
  `runs/cheddar-<split>-s<seed>/`.
- `sweep` uses only GPUs not in use by others, limits itself to 80 % of the available cores,
  and skips runs that already have results, so an interrupted sweep resumes.
- Deterministic kernels are on, so a given seed on the same GPU architecture and PyTorch
  build reproduces its result exactly, `best.pt` included.
- On eight RTX 3090s: one card per run, fifteen runs in two waves, 69 min. All fifteen at
  once, two per card, took 51 min — 20–26 s per epoch, plus about 8 min at the start while
  fifteen processes read the same two 1 GB caches from one disk.
- A run peaks at 0.72 GB of GPU memory, so sharing a card costs speed alone.

### How training works

- **Objective.** Stochastic gradient descent on the measured permeabilities of the training
  peptides, minimising mean squared error on the scaled label. One pass over the training set
  is an epoch; the budget is fixed at 120. No conformational search or docking happens at
  training time — the ensembles are inputs, computed in advance.
- **Partitioning and model selection.** Each split divides the peptides into train,
  validation and test. Parameters are fitted on train, the checkpoint is selected by
  validation MSE on the original log P_app scale, and that checkpoint is scored once on test,
  which is never consulted during fitting or selection. The test figures are what Table 1
  reports.
- **Replication.** Initialisation and batch order depend on the seed, so each configuration
  runs with five seeds (1, 2, 7, 13, 42) and is reported as a mean with its standard
  deviation, separating the effect of the method from seed variance.

## Tuning and ablations

`configs/cheddar.json` holds every hyperparameter of the released model. Copy it, edit it,
and pass it with `--config`.

```bash
python -m cheddar sweep --config configs/abl_water_only.json
python -m cheddar train --config configs/abl_no_contact_edges.json --split ID --seed 1
```

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
| `mask_node_cols`, `mask_space_cols` | withhold named columns by setting them to 0.0 after standardization, leaving the feature width and the parameter count unchanged | absent |
| `cache`, `alt_cache` | the two solvent caches; a name containing `meanedge` is built with the mean-distance rule of Eq. (6) instead of occupancy | ensemble_{Water,Hexane} |

- The `abl_*.json` files are the arms of Table 2, so each row of that table is one sweep.
- Runs are named after the configuration, for example `runs/abl_water_only-ID-s1/`.
- The "w/o ensemble geometry" row came from masking, which `abl_no_ensemble_masked.json`
  reproduces: the seven dispersion columns on the node and six on the 3D contact relation are
  set to 0.0 after standardization, so the model keeps its 554,561 parameters and only the
  information is withheld.
- `abl_no_ensemble_geometry.json` instead removes those columns and reads a
  mean-distance-selected cache. That changes the feature width and does **not** reproduce the
  published figure.

## Cyclic peptide permeability prediction

- A cyclic peptide is a short peptide whose backbone is closed into a ring.
- The class combines broad interaction surfaces with conformational constraints that can
  improve resistance to proteolysis, which makes it attractive for targets conventional small
  molecules struggle to address.
- Many cyclic peptides have limited passive membrane permeability, restricting access to
  intracellular targets and contributing to poor oral absorption. Predicting permeability is
  therefore useful in identifying candidates, particularly for scaffolds beyond those already
  measured.
- Permeability depends on chemical composition and on whether polar groups are exposed or
  shielded, which can change substantially with stereochemistry.
- Solvent-dependent conformational preferences let a molecule expose polar groups in the
  aqueous environment around the membrane while favouring intramolecular hydrogen bonding and
  polar-group burial in its nonpolar core — molecular chameleonicity.
- A representation therefore has to cover both environments and describe the conformational
  distribution within each. CHEDDAR reads independently sampled explicit-solvent ensembles in
  water and in hexane, and predicts `log P_app` as measured by a PAMPA assay.

## Architecture

![The CHEDDAR architecture](docs/architecture.png)

- A peptide enters as two graphs, one per solvent, sharing nodes and bonds and differing only
  in the geometry read off the water and hexane ensembles and in the solvent token on every
  node.
- Atom features combine a chemical block, an ensemble-geometry block and that token.
- Bonded and 3D-contact relations carry disjoint feature blocks and separate encoders and
  message MLPs, so their aggregates reach the node update without being merged.
- Pooling concatenates the sum, mean and maximum statistics and leaves their weighting to the
  head, which is bounded to the normalized label range.
- The two views are never compared inside the model — each contributes its own loss term —
  and are averaged only at inference.

## Repository contents

```
predict.sh   single-command prediction, the entry point most readers require
cheddar/     __init__.py  marks the package, empty
             __main__.py  the command line; `python -m cheddar --help` lists every command
             fetch.py     find the trajectories, or download the Zenodo record
             features.py  cache construction from MD frames, split and label IO
             data.py      cache reader, standardization, label scaling
             model.py     the network
             train.py     training loop and evaluation
             sweep.py     the 3 x 5 grid across free GPUs
             eval.py      score checkpoints without training
             predict.py   permeability for named peptides
configs/     cheddar.json  every hyperparameter of the released model
             abl_*.json    the arms of Table 2
ckpts/       the 15 checkpoints behind the paper's main table
data/        label table, atom annotation, the three splits, a SMILES correction
cache/       built on first use from the trajectories; not shipped, not tracked
environment.yml  the exact versions every number in the paper was produced with
LICENSE      MIT
```

## Data

Included here: the label table, the atom annotation and the three split files. Not included:
the MD trajectories, which are the public CycPeptMPDB-4D release — 1.6 GB compressed, 10.2 GB
unpacked, at https://zenodo.org/records/21274695 .

**Obtaining them.** No configuration is necessary.

- `python -m cheddar data` searches the customary locations for an unpacked copy, downloads
  and unpacks the record if none is found, resumes an interrupted download, and verifies the
  published MD5.
- `CHEDDAR_TRAJ` points at a copy in a non-standard location: the directory containing
  `Water/` and `Hexane/`.
- `CHEDDAR_NO_DOWNLOAD=1` prevents any network access.
- About 12.5 GB of disk is needed during unpacking, 10.2 GB thereafter once the archive is
  removed.

**The caches.**

- The release is read exactly as it unpacks, from `Water/Trajectories/*_Traj.pdb` and the
  corresponding hexane path. No conversion needed.
- Both caches are built on first use, together occupying 1.9 GB. About 90 s each on 32 cores:
  some 20 s to compute features for the 4,925 peptides, the rest to write a gigabyte to disk.
- They are derived files and may be deleted at any time; the next command rebuilds them.
- A rebuild reproduces every feature array bit for bit, so no result depends on keeping a
  particular copy. Only the `.pkl` framing may shift by a few hundred bytes, which nothing
  reads.

**Two points worth attention.**

- The release holds 5,160 peptides; the model uses the 4,925 that have an atom annotation
  here. The remaining 235 are skipped, which is expected and not an error.
- An earlier release of the same data stores coordinates in nanometres, under which the 5 Å
  contact cutoff would make every atom pair a contact. The builder checks the units and
  declines to proceed rather than producing silently wrong features.

**The SMILES correction.** `data/nal_fix.json` corrects 95 peptides whose tabulated SMILES
says 2-naphthylalanine while the deposited geometry is 1-naphthylalanine. It is applied
automatically; without it those 95 get atom features inconsistent with their own trajectory.

## License

MIT, see LICENSE.
