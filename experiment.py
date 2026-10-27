# USAGE
# python experiment.py
# python experiment.py --config myconfig.json
# python experiment.py --models hourglass unet --epochs-per-checkpoint 10 --num-checkpoints 2
# python experiment.py --report                 # reprint the latest run's report
# python experiment.py --report 261027T122740    # reprint a specific past run's report
#
# Thin entry point: all the actual logic lives in experiments/*.py (see
# experiments/README.md for the package layout and pipeline).

from experiments.config import ExperimentConfig
from experiments.report import Report
from experiments.runner import ExperimentRunner


if __name__ == '__main__':
	cfg = ExperimentConfig.setup()

	if cfg.report_timestamp:
		Report.load(cfg.output_root, cfg.report_timestamp).print()
	else:
		ExperimentRunner(cfg).run()
