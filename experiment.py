# USAGE
# python experiment.py
# python experiment.py --config myconfig.json
# python experiment.py --models hourglass unet --epochs-per-checkpoint 10 --checkpoints 2
# python experiment.py --report                 # reprint the latest run's report
# python experiment.py --report 261027T122740    # reprint a specific past run's report
# python experiment.py --resume                  # resume the latest run (training/plotting/
#                                                  # reporting, whichever is unfinished)
# python experiment.py --resume 261027T122740     # resume one specific past run
#
# Thin entry point: all the actual logic lives in experiments/*.py (see
# experiments/README.md for the package layout and pipeline).

import socket

from experiments.log import Log

Log.info(f'running on {socket.gethostname()}')

from experiments.config import ExperimentConfig
from experiments.report import Report
from experiments.resume import Resumer
from experiments.runner import ExperimentRunner


if __name__ == '__main__':
	cfg = ExperimentConfig.setup()

	if cfg.resume_timestamp:
		Resumer.resume(cfg.output_root, cfg.resume_timestamp)
	elif cfg.report_timestamp:
		Report.load(cfg.output_root, cfg.report_timestamp).print()
	else:
		ExperimentRunner(cfg).run()
