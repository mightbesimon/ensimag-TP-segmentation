# experiments/

Everything behind `experiment.py`. One file per responsibility, each under ~200 lines.

```
config.py      RunSpec (one model's architecture: model_key, enc_channels, dec_channels)
               ExperimentConfig (MODEL_REGISTRY, sweep expansion, default-json -> --config
               file -> CLI-flags layering). ExperimentConfig.load() is the only entry point.

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
               ComparisonPlots -- the 5 cross-model comparison figures.
               Plots data only; never evaluates or trains.

dashboard.py   Status -- reads/writes experiment_output/status.json (global, cross-run) and
               each run's own <timestamp>/status.json (this run's pipeline progress).
               Dashboard -- builds the project-root dashboard.html from Status data.

report.py      ModelReport, Report -- the full-detail terminal/text report, built entirely
               from what's on disk for a run (so a past run's report can be reprinted
               without re-running anything).

runner.py      ModelRun, ModelResult, ExperimentRunner -- orchestrates everything above
               following the pipeline documented below.
```

## Pipeline

```
normal run:   config -> data -> train -> eval -> plot -> report
resumed run:  train (whatever's left) -> eval -> plot -> report
```

`train` and `eval` are interleaved *per target epoch*, not run as separate passes: for
each target epoch (`epochs_per_checkpoint * [1..num_checkpoints]`, e.g. `[20, 40, 60]`),
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
    training_curves.png  final_metrics_bar.png  loss_distribution_over_epochs.png
    qualitative_grid.png  summary_table.png  summary.csv
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
