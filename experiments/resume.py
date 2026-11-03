# resumes a past run from wherever it left off -- training, plotting, or
# reporting, whichever is unfinished -- by re-entering ExperimentRunner.run()
# with that run's own saved config and run_specs, writing into its own
# <timestamp>/ directory instead of minting a new one.
#
# no separate reconstruction logic needed: every model already resumes at
# the checkpoint/epoch level on its own (CheckpointManager, keyed by hash +
# seed), so replaying the exact same pipeline does the right amount of work
# whether training was 0%, 50%, or 100% done -- a fully-trained model simply
# finds nothing left to train and falls straight through to plot + report.

import json
import os

from experiments.config import ExperimentConfig, RunSpec
from experiments.log import Log
from experiments.runner import ExperimentRunner


class Resumer:
	@staticmethod
	def _latest_resumable_timestamp(out_root):
		# config.json is written before anything else in run() -- the
		# earliest possible marker that a run directory exists at all,
		# unlike manifest.json (only written once every model finishes) or
		# status.json (only written once the first model finishes)
		if not os.path.isdir(out_root):
			return None
		candidates = [
			d for d in os.listdir(out_root)
			if os.path.isdir(os.path.join(out_root, d)) and os.path.exists(os.path.join(out_root, d, 'config.json'))
		]
		return sorted(candidates)[-1] if candidates else None

	@classmethod
	def resume(cls, out_root, timestamp='latest'):
		if timestamp == 'latest':
			timestamp = cls._latest_resumable_timestamp(out_root)
			if timestamp is None:
				raise FileNotFoundError(f'no experiment runs with a config.json found under {out_root}')

		run_dir = os.path.join(out_root, timestamp)
		with open(os.path.join(run_dir, 'config.json')) as f:
			saved_config = json.load(f)

		run_specs = [
			RunSpec(e['model_key'], tuple(e['enc_channels']), tuple(e['dec_channels']))
			for e in saved_config['run_specs']
		]
		cfg = ExperimentConfig(saved_config)
		Log.info(f'resuming {run_dir} (seed {cfg.seed})')
		return ExperimentRunner(cfg).run(run_specs=run_specs, timestamp=timestamp)
