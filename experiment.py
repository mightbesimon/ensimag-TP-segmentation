# USAGE
# python experiment.py
# python experiment.py --config myconfig.json
# python experiment.py --models hourglass unet --epochs-per-checkpoint 10 --checkpoints 2
# python experiment.py --report                 # reprint the latest run's report
# python experiment.py --report 261027T122740    # reprint a specific past run's report
# python experiment.py --replot                  # regenerate comparison plots/report for the latest run
# python experiment.py --replot 261027T122740     # regenerate for one specific past run
#
# Thin entry point: all the actual logic lives in experiments/*.py (see
# experiments/README.md for the package layout and pipeline).

from experiments.config import ExperimentConfig
from experiments.replot import Replotter
from experiments.report import Report
from experiments.runner import ExperimentRunner


if __name__ == '__main__':
	cfg = ExperimentConfig.setup()

	if cfg.replot_timestamp:
		Replotter.replot(cfg.output_root, cfg.replot_timestamp)
	elif cfg.report_timestamp:
		Report.load(cfg.output_root, cfg.report_timestamp).print()
	else:
		ExperimentRunner(cfg).run()
