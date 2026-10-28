# experiments/

Everything behind `experiment.py`. One file per responsibility, each under ~200 lines.

```
config.py      RunSpec (one model's architecture: model_key, enc_channels, dec_channels)
               ExperimentConfig (MODEL_REGISTRY, sweep expansion, default-json -> --config
               file -> CLI-flags layering). ExperimentConfig.setup() is the only entry point.

data.py        DataManager -- seeding, dataloaders, the fixed seeded eval-image subset used
               for every checkpoint/model so comparisons are apples-to-apples.

eval.py        Evaluator -- given an already-built model and one image, runs it and computes
               bce/dice/iou. Does not build, train, persist, or plot anything.

train.py       CheckpointManager -- checkpoints/index.json, save/load/find_latest.
               TrainingManager -- builds on a model handed to it, hashes it (by actual
               state_dict signature, not config numbers -- see hash() docstring), trains it.
               "create and store the model."

plots.py       PredictGridPlotter -- per-checkpoint grid (image/gt/pred/prob, best+worst
               marked) and the training-curve snapshot for one model.
               ComparisonPlots -- the 6 cross-model comparison figures.
               Plots data only; never evaluates or trains.

dashboard.py   Status -- reads/writes experiment_output/status.json (global, cross-run) and
               each run's own <timestamp>/status.json (this run's pipeline progress).
               Dashboard -- builds the project-root dashboard.html from Status data.

report.py      ModelReport, Report -- the full-detail terminal/text report, built entirely
               from what's on disk for a run (so a past run's report can be reprinted
               without re-running anything).

runner.py      ModelRun, ModelResult, ExperimentRunner -- orchestrates everything above
               following the pipeline documented below.

replot.py      Replotter -- regenerates a past run's comparison figures + report entirely
               from disk (rebuilds each model, loads its latest checkpoint, re-plots), no
               training involved. For when a plotting change should apply to an old run too.
```

## Arguments

```
python3 experiment.py                                 # run everything with experiment_config.json's defaults
python3 experiment.py --models hourglass unet          # only these models (fastest-first order, not sorted)
python3 experiment.py --epochs-per-checkpoint 10 --checkpoints 3   # target epochs = [10, 20, 30]
python3 experiment.py --config myconfig.json            # load a different saved config (see below)
python3 experiment.py --config myconfig.json --lr 0.01  # flags still override whatever the file set
python3 experiment.py --report                          # reprint the latest run's report, no training
python3 experiment.py --report 261027T122740            # reprint one specific past run, by its <timestamp> dir name
python3 experiment.py --replot                          # regenerate the latest run's comparison figures, no training
python3 experiment.py --replot 261027T122740             # regenerate one specific past run's figures (e.g. after a plotting fix)
```

Every flag has a default pulled from `experiment_config.json` (project root) — that file *is* the
set of defaults, not just an example; edit it directly to change what a bare `python3
experiment.py` does. `--config some_other.json` swaps in a different file's values as the new
base instead (handy for saving named configs you want to re-run later); any flag you also pass
on the command line overrides whatever that file says.

| flag | meaning | default (from `experiment_config.json`) |
|---|---|---|
| `--models` | which model keys to train (space-separated) | `simpleconv hourglass unet` |
| `--epochs-per-checkpoint` | epochs trained per checkpoint step | `20` |
| `--checkpoints` | how many checkpoint steps = target epochs; e.g. `3` with the above → targets `[20, 40, 60]` | `1` |
| `--batch-size` | DataLoader batch size | `256` |
| `--lr` | Adam learning rate | `0.001` |
| `--num-eval-images` | size of the fixed seeded eval-image subset used for every checkpoint/model | `10` |
| `--seed` | seed for data split, eval-image subset, and each model's init/shuffling | `22` |
| `--output-root` | where everything gets written | `experiment_output` |
| `--config` | path to a JSON file of these same keys (plus optional `sweeps`) to use as the base instead of `experiment_config.json` | none |
| `--report` | print a report instead of running; bare flag = latest run, or pass a `<timestamp>` for a specific past run | none |
| `--replot` | regenerate a past run's comparison figures + report instead of running, entirely from what's already on disk (no training); bare flag = latest run, or pass a `<timestamp>` | none |

**There is no `--features`/`--depths` flag.** Varying feature count or depth is only done
through `sweeps` in a `--config` JSON file — not a flat flag, by design, since it's a list of
per-model entries rather than a single value. To sweep architectures, write a file and pass it:

```json
// hourglass_sweep.json
{
	"models": ["hourglass"],
	"sweeps": [
		{"model": "hourglass", "features": [8, 16, 32], "depths": [2, 3, 4]},
		{"model": "unet", "enc_channels": [3, 24, 48], "dec_channels": [48, 24, 1]}
	]
}
```
```
python3 experiment.py --config hourglass_sweep.json
```
Each sweep entry is either `{"model", "features", "depths"}` (expanded into every
features × depths combination, via the same symmetric doubling scheme each class's own
defaults follow) or `{"model", "enc_channels", "dec_channels"}` (used exactly as given, for a
custom shape the features/depths formula can't express). When `sweeps` is non-empty it replaces
`--models` entirely for that run — every architecture trained comes from the sweep list, and
`--models` on the command line is ignored.

## Pipeline

```
normal run:   config -> data -> train -> eval -> plot -> report
resumed run:  train (whatever's left) -> eval -> plot -> report
```

`train` and `eval` are interleaved *per target epoch*, not run as separate passes: for
each target epoch (`epochs_per_checkpoint * [1..checkpoints]`, e.g. `[20, 40, 60]`),
`ExperimentRunner` either reuses an existing checkpoint at that exact epoch, or trains just
the gap from wherever the model currently is, then immediately runs `eval` (the predict
grid) and checkpoints. This is also how resuming works: every model is identified by
`TrainingManager.hash()` (the actual parameter-shape signature, not the enc/dec numbers you
passed in -- so a checkpoint is never loaded into incompatible code), and
`CheckpointManager` looks up the latest checkpoint for that hash before training a single
epoch. Interrupting a run (lost SLURM connection, ctrl-C) and re-running the same command
picks up exactly where it left off, model by model.

## experiment_output/

```
experiment_output/
  status.json                          # global: every model hash ever trained (architecture,
                                        # best metrics seen, max epoch, which runs used it) and
                                        # every run (its stage, which hashes it touched)
  checkpoints/
    index.json                         # hash -> architecture + list of {epoch, path,
                                        # timestamp, training_time_seconds} -- the actual
                                        # resumability source of truth
    hourglass_<hash>_ep20_<timestamp>.pth
  SimpleConv_<hash>/                    # one directory per model hash, shared across every
    config.json                        # run that ever touches it (not nested under a
    history.json                       # timestamp -- that's what makes resuming across
    training_curve_ep20.png            # separate invocations possible)
    predict_grid_ep20.png
    predict_metrics_ep20.csv
  HourGlass_<hash>/ ...   UNet_<hash>/ ...
  <timestamp>/                          # one directory per experiment.py invocation
    config.json                        # the full effective config used (post CLI overrides)
    manifest.json                      # which model hashes this run compared
    status.json                        # this run's own pipeline progress (training ->
                                        # plotting -> reporting -> complete)
    training_curves.png  generalization_gap.png  final_metrics_bar.png
    loss_distribution_over_epochs.png  qualitative_grid.png
    summary_table.png  summary.csv
    report.txt
```

`status.json` (global) vs `checkpoints/index.json` vs `<timestamp>/status.json` -- three
different scopes, not duplicates of each other:
- `checkpoints/index.json`: physical truth about checkpoint *files* (what exists, where).
- `experiment_output/status.json`: cross-run *dashboard* data -- best metrics ever seen per
  hash, which runs used it -- info that exists nowhere else.
- `<timestamp>/status.json`: this *one run's* pipeline progress, so a crash between training
  and plotting can resume straight into plotting instead of re-deriving that from scratch.

`dashboard.html` (project root) is rebuilt from `experiment_output/status.json` at the end
of every run -- open it in a browser to click through runs and model hashes.
