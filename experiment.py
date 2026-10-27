# USAGE
# python experiment.py
# python experiment.py --config myconfig.json
# python experiment.py --models hourglass unet --epochs-per-checkpoint 10 --num-checkpoints 2
# python experiment.py --report                 # reprint the latest run's report
# python experiment.py --report 20261027_110156  # reprint a specific past run's report
#
# Thin entry point: all the actual logic lives in experiments/*.py.

from experiments.cli_config import ExperimentConfig
from experiments.report import Report
from experiments.runner import ExperimentRunner



if __name__ == '__main__':
	cfg, report_timestamp = ExperimentConfig.parse_args()

	if report_timestamp is not None:
		report = Report.load(cfg.output_root, report_timestamp)
		report.print()
	else:
		ExperimentRunner(cfg).run()
