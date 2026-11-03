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

train.py       CheckpointManager -- checkpoints/index.json, save/load/find_latest (per
               hash AND seed). TrainingManager -- builds on a model handed to it, hashes
               it (by actual state_dict signature + declared module tree, not config
               numbers -- see architecture_signature() docstring), trains it.
               "create and store the model."

plots.py       PredictGridPlotter -- per-checkpoint grid (image/gt/pred/prob, best+worst
               marked) and the training-curve snapshot for one model.
               ComparisonPlots -- the 7 cross-model comparison figures.
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
python3 experiment.py --models hourglass --features 8 16 32 --depths 2 3 4  # sweep all 9 combos
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
| `--features` | feature-count values to sweep, applied to every model in `--models` (grid with `--depths`); overrides any JSON `sweeps` | none |
| `--depths` | depth values to sweep, applied to every model in `--models` (grid with `--features`); overrides any JSON `sweeps` | none |
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

`--features`/`--depths` cover the common case directly from the CLI, e.g.
`--models hourglass --features 8 16 32 --depths 2 3 4` sweeps all 9 combinations for
`hourglass` (via the same symmetric doubling scheme each class's own defaults follow). They
apply the *same* grid to every model named in `--models`, in addition to (not instead of)
`--models` -- unlike the JSON form below, `--models` still picks which architectures train.

Anything more elaborate -- a different grid per model, or an exact custom `enc_channels`/
`dec_channels` shape the features/depths formula can't express -- still needs `sweeps` in a
`--config` JSON file:

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
features × depths combination) or `{"model", "enc_channels", "dec_channels"}` (used exactly
as given). When `sweeps` is non-empty -- whether from a JSON `--config` file or because
`--features`/`--depths` was passed -- it replaces `--models`-driven default architectures
entirely for that run: every architecture trained comes from the sweep list.

## Pipeline

```
normal run:   config -> data -> train -> eval -> plot -> report
resumed run:  train (whatever's left) -> eval -> plot -> report
```

`train` and `eval` are interleaved *per target epoch*, not run as separate passes: for
each target epoch (`epochs_per_checkpoint * [1..checkpoints]`, e.g. `[20, 40, 60]`),
`ExperimentRunner` either reuses an existing checkpoint at that exact epoch+seed, or trains
just the gap from wherever the model currently is, then immediately runs `eval` (the
predict grid) and checkpoints. This is also how resuming works: every model's
*architecture* is identified by `TrainingManager.architecture_signature()` -- both the
parameter name+shape signature AND the full declared module tree (`str(model)`), so a
checkpoint is never loaded into incompatible code, and a structural edit with no parameter
shape change (e.g. adding a ReLU inside `Block`) still invalidates it -- hashed into
`TrainingManager.hash_signature()`. `CheckpointManager` looks up the latest checkpoint for
that hash *and seed* before training a single epoch (different seeds of the same
architecture are different, non-resumable-into-each-other experiments, but do share the same
directory and checkpoint index entry). Interrupting a run (lost SLURM connection, ctrl-C)
and re-running the same command picks up exactly where it left off, model by model.

The architecture hash itself never appears in directory or file names, since it isn't
readable at a glance -- instead they use `arch_label` (e.g. `ft16_d3` for features=16,
depth=3), which is
human-readable but *not* guaranteed unique across genuinely different architectures. To
protect against that, `ExperimentRunner._prepare` fails loudly if a model directory's stored
`config.json` hash doesn't match what the current code just built for that same `arch_label`
-- that's your signal to move/delete the stale directory (and its `checkpoints/index.json`
entry) rather than silently mixing two incompatible models' checkpoints together.

## experiment_output/

```
experiment_output/
  status.json                          # global: every (model hash, seed) ever trained
                                        # (architecture, best metrics seen, max epoch, which
                                        # runs used it) and every run (its stage, which
                                        # hash+seed combos it touched)
  checkpoints/
    index.json                         # hash -> list of {epoch, seed, path, training_time_
                                        # seconds} -- the actual resumability source of truth
                                        # (filtered by seed, not just hash). architecture info
                                        # (enc/dec channels etc) is NOT duplicated here -- the
                                        # model's own config.json is the single source of truth
    simpleconv_ft16_d3_ep20_s22.pth
  SimpleConv_ft16_d3/                   # one directory per architecture (NOT per hash --
    config.json                        # see arch_label note above), shared across every run
    history_s22.json                   # and every seed that ever touches it
    history_s7.json
    training_curve_ep20_s22.png
    predict_grid_ep20_s22.png
    predict_metrics_ep20_s22.csv
  HourGlass_ft32_d4/ ...   UNet_ft24_d2/ ...
  <timestamp>/                          # one directory per experiment.py invocation
    config.json                        # the full effective config used (post CLI overrides)
    manifest.json                      # which model hashes this run compared
    status.json                        # this run's own pipeline progress (training ->
                                        # plotting -> reporting -> complete)
    training_curves.png  generalization_gap.png  final_metrics_bar.png
    loss_distribution_over_epochs.png  qualitative_grid.png  overlay_grid.png
    summary_table.png  summary.csv
    video_comparison_00011.gif         # every model's predicted-mask overlay, stacked as
                                        # rows (ground truth on top), one frame per video
                                        # frame -- always video 00011, slower than the
                                        # other plots above so it gets its own progress bar
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
