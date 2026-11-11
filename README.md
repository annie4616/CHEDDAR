# CHEDDAR

Cyclic-peptide membrane permeability from solvent-dependent conformational ensembles.

Each peptide becomes two atom graphs, one per explicit-solvent MD ensemble (water and
hexane). Atom features are statistics over 100 frames rather than one structure, contact
edges are selected by how often a pair stays in contact, and the two solvent predictions are
averaged. The model has 554,561 parameters, trains in about 31 min on one GPU and predicts
in 2.3 ms per peptide.

## If you only want a prediction

Four steps, in this order. Step 3 downloads 1.6 GB, unpacks it to 10.2 GB and needs about
12.5 GB of free disk while it does so. It is the only slow step, it runs once, and
everything afterwards takes seconds.

```bash
conda env create -f environment.yml && conda activate cheddar   # 1. install, about 3 min
python -m cheddar check                                         # 2. what is already here
python -m cheddar data                                          # 3. get the MD data, 1.6 GB
./predict.sh 5222                                               # 4. predict
```

Step 2 prints what is present and what is missing, and step 3 is a no-op if you already have
the trajectories, so running both costs nothing when they are not needed.

`5222` is a CycPeptMPDB id; name as many as you like, or run `./predict.sh` with no
arguments for a worked example. The answer looks like this:

```
id          logPapp     sd   measured partition
5222          -5.43   0.05      -5.25 test

5222: predicted log P_app -5.43, above average permeability for a cyclic peptide
   -- more permeable than 68% of the 4925 measured ones.
   measured -5.25, so this prediction is 0.18 log units off.
```

### Reading that output

`log P_app` is the base-10 logarithm of the apparent permeability coefficient in cm/s, as
measured by a PAMPA assay. It is always negative, and **less negative means more
permeable**: −5.0 is a peptide that crosses well, −7.0 one that barely crosses. Because a
bare log unit is hard to place, each prediction is also ranked against the 4,925 peptides
that have a measured value, which is the only scale this model was ever fitted on.

`sd` is the spread across the five independently trained models that ship here. A small
`sd` means the five agree; a large one means the peptide sits somewhere they disagree
about, and the prediction deserves less weight.

`partition` says whether that peptide was in the training set of the model you used. If it
says `train`, the number is in-sample and flattering. Use `--split OD_Murcko` for a model
that has never seen the scaffold:

```bash
./predict.sh 4705 --split OD_Murcko
```

### One thing worth knowing before you start

CHEDDAR does not read a SMILES string and return a number. It reads **molecular dynamics
trajectories** — the peptide simulated in water and in hexane, 100 frames each — because
the whole point of the method is that permeability depends on how a peptide changes shape
between the two environments. Every peptide in CycPeptMPDB-4D already has these, so any of
those 4,925 can be predicted immediately. A *new* peptide needs its two MD simulations run
first, placed in the trajectory folder under the same naming as the public release.

The first command you run builds a summary of those trajectories and takes a couple of
minutes. Everything after that takes seconds.

## The six commands

Listed in the order you would run them. The first two are setup and only matter once; most
readers never need the last two.

```bash
python -m cheddar check                      # versions, device, which data files are present
python -m cheddar data                       # locate the trajectories, or download and unpack them
python -m cheddar predict 5222               # permeability for named peptides, about 8 s
python -m cheddar eval                       # score the shipped checkpoints (the default command)
python -m cheddar train --split ID --seed 1  # retrain one model, about 31 min on a GPU
python -m cheddar sweep                      # retrain 3 splits x 5 seeds, one worker per free GPU
```

`./predict.sh` is a thin wrapper over `python -m cheddar predict`, so anything you can pass
to one you can pass to the other: `--partition test` for a whole partition, `--all` for
every peptide, `--out results.csv` to write a spreadsheet.

If something looks wrong, `python -m cheddar check` reports what is installed, which GPU it
can see, and which data files it found, without running anything.

## Reproducing the paper

```bash
python -m cheddar          # scores the 15 shipped checkpoints, prints the paper's main row
```

No arguments, no paths. Scoring all fifteen checkpoints takes about three minutes, most of
it spent reading the two caches. It runs without a GPU, but several times slower. It prints

```
split           n  MSE              R2
ID              5  0.145 +- 0.003    0.621 +- 0.007
OD_Murcko       5  0.236 +- 0.018    0.660 +- 0.026
Cliff_ratio     5  0.142 +- 0.003    0.683 +- 0.006
```

which is the CHEDDAR row of Table 1, mean and standard deviation over seeds 1, 2, 7, 13, 42,
followed by the same result in words. The three splits are the random (`ID`), Bemis–Murcko
scaffold (`OD_Murcko`) and activity-cliff (`Cliff_ratio`) partitions of Appendix C.

Retraining from scratch gives the same table. `python -m cheddar sweep` rebuilds all fifteen
models from random initialisation and reproduces every row above in both mean and standard
deviation, each fresh `best.pt` coming out byte-identical to the checkpoint shipped here.
Two limits of that check are worth stating: it is a deterministic replay on the same GPU
architecture and PyTorch build rather than an independent reimplementation, and it covers
the CHEDDAR row only — the seven baselines of Table 1 are not in this repository.

## Training the model

You do not need to train anything to use CHEDDAR — the fifteen models behind the paper ship
in `ckpts/`, and `predict.sh` already uses them. This section is for rebuilding those fifteen
yourself. It assumes no background in deep learning.

**What training actually does.** The model starts from random numbers and repeatedly guesses
the permeability of peptides whose measured value we know. Each guess is compared with the
measurement, and the numbers inside the model are nudged so the next guess is a little
closer. One pass over the training peptides is an *epoch*; CHEDDAR does 120 of them. Nothing
is being searched or docked — the model is being fitted, the way you would fit a curve, just
with a great many more coefficients.

**How the data is divided.** The peptides are split three ways. The model is fitted on the
*training* part, watched on the *validation* part so we can tell when it has learned as much
as it usefully can, and judged once on the *test* part, which it never sees during fitting.
We keep the version of the model that did best on validation, then report its score on test.
That last number is the honest one, and it is what Table 1 reports.

**Why five seeds.** Training starts from random values, so two runs of the same recipe give
slightly different models. Running five and reporting the mean with its spread shows how
much of a result is the method and how much is luck.

One model, about 31 minutes on one GPU:

```bash
python -m cheddar train --split ID --seed 1
```

All fifteen, which is how the paper's table was produced:

```bash
python -m cheddar sweep
```

`sweep` takes only GPUs that nobody else is using, caps itself at 80 % of the cores it is
given, and skips runs that already have results, so an interrupted sweep continues where it
stopped. Runs land in `runs/cheddar-<split>-s<seed>/` as `test_metrics.json`, `history.json`,
`predictions.csv` and `best.pt`. Deterministic kernels are on, so the same seed on the same
GPU architecture and PyTorch build reproduces its result exactly, down to a byte-identical
`best.pt`.

`sweep` gives each run a card to itself, so on eight RTX 3090s the fifteen runs go in two
waves and take 69 min. Putting all fifteen on the eight cards at once instead, two per
card, took 51 min: 20–26 s an epoch, plus about 8 min at the start while fifteen processes
read the same two 1 GB caches off one disk. A run peaks at 0.72 GB of GPU memory, so
sharing a card costs speed and nothing else.

## Tuning and ablations

`configs/cheddar.json` holds every hyperparameter of the released model. Copy it, change what
you want and pass it with `--config`.

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
| `edge_block` | `both` or `bond` to drop the 3D contact relation | both |
| `both_solvents`, `solvent_token` | use two solvent views, append iota_s | true, true |
| `meanonly` | drop the dispersion columns, leaving ensemble means | absent |
| `cache`, `alt_cache` | the two solvent caches; a name containing `meanedge` is built with the mean-distance rule of Eq. (6) instead of occupancy | ensemble_{Water,Hexane} |

The four shipped `abl_*.json` are the four arms of Table 2, so each row of that table is one
sweep. Runs are named after the config, for example `runs/abl_water_only-ID-s1/`.

## What is here

```
predict.sh   one-click prediction, the only entry point most readers need
cheddar/     __init__.py  marks the package, empty
             __main__.py  the six commands above
             fetch.py     find the trajectories, or download the Zenodo record
             features.py  cache construction from MD frames, split and label IO
             data.py      cache reader, standardization, label scaling
             model.py     the network
             train.py     training loop and evaluation
             sweep.py     the 3 x 5 grid across free GPUs
             eval.py      score checkpoints without training
             predict.py   permeability for named peptides
configs/     cheddar.json  every hyperparameter of the released model
             abl_*.json    the four arms of Table 2
ckpts/       the 15 checkpoints behind the paper's main table
data/        label table, atom annotation, the three splits, a SMILES correction
cache/       built on first use from the trajectories; not shipped, not tracked
environment.yml  the exact versions every number in the paper was produced with
LICENSE      MIT
```

## Data

The label table, the atom annotation and the three split files ship here. The MD
trajectories do not; they are the public CycPeptMPDB-4D release, 1.6 GB compressed and
10.2 GB unpacked, at https://zenodo.org/records/21274695 .

Nothing needs to be set. `python -m cheddar data` looks for an unpacked copy in the usual
places, downloads and unpacks the record if there is none, resumes an interrupted download
and checks the published MD5. If your copy is somewhere unusual, point `CHEDDAR_TRAJ` at the
folder that holds `Water/` and `Hexane/`, and set `CHEDDAR_NO_DOWNLOAD=1` if the machine
should never reach the network. Budget about 12.5 GB of disk while unpacking and 10.2 GB
after, once the zip is removed.

The release is read exactly as it unpacks, `Water/Trajectories/*_Traj.pdb` and the same for
hexane, so nothing has to be converted. Both solvent caches are built on first use and
occupy 1.9 GB together, about 90 s each on 32 cores: some 20 s computing the features for
the 4,925 peptides and the rest writing a gigabyte to disk. They are derived files: delete
them whenever you like and the next command rebuilds them, and a rebuild reproduces every
feature array bit for bit, so the table above does not depend on keeping any particular
copy of them. Only the `.pkl` framing can shift by a few hundred bytes, which nothing reads.

Two things worth knowing. The release holds 5,160 peptides while the model uses the 4,925
that have an atom annotation here; the extra 235 are skipped, which is expected and not an
error. And an earlier release of the same data stores coordinates in nanometres, where the
5 A contact cutoff would make every atom pair a contact; the builder checks the units and
refuses rather than producing quiet nonsense.

`data/nal_fix.json` corrects 95 peptides whose table SMILES is 2-naphthylalanine while the
deposited geometry is 1-naphthylalanine. It is applied automatically; without it those 95
molecules get atom features that disagree with their own trajectory.

## License

MIT, see LICENSE.
