# layered config: hard defaults -> optional --config json file -> explicit
# CLI flags (each layer overrides the previous). sweeps (varying num_features
# and depth per model) are JSON-only, since there's no clean flat-flag way
# to express a list of per-model sweep definitions.

import argparse
import json

from model import config as model_config
from experiments.model_config import MODEL_REGISTRY, default_run_specs, expand_sweep

DEFAULTS = {
	'models': sorted(MODEL_REGISTRY),
	'epochs_per_checkpoint': model_config.NUM_EPOCHS,
	'num_checkpoints': 1,
	'batch_size': model_config.BATCH_SIZE,
	'lr': model_config.INIT_LR,
	'num_eval_images': 10,
	'seed': 42,
	'output_root': 'experiment',
	'sweeps': [],
}


class ExperimentConfig:
	def __init__(self, values):
		self.values = values

	def __getattr__(self, name):
		try:
			return self.values[name]
		except KeyError:
			raise AttributeError(name)

	def to_dict(self):
		return dict(self.values)

	def run_specs(self):
		if self.values['sweeps']:
			return expand_sweep(self.values['sweeps'])
		return default_run_specs(self.values['models'])

	def target_epochs(self):
		# target total epoch counts, e.g. [20, 40, 60] -- every run spec is
		# driven to the same set of target epochs for a fair comparison
		step = self.values['epochs_per_checkpoint']
		n = self.values['num_checkpoints']
		return [step * i for i in range(1, n + 1)]

	@classmethod
	def parse_args(cls, argv=None):
		pre = argparse.ArgumentParser(add_help=False)
		pre.add_argument('--config', default=None)
		pre.add_argument('--report', nargs='?', const='latest', default=None)
		known, _ = pre.parse_known_args(argv)

		base = dict(DEFAULTS)
		if known.config:
			with open(known.config) as f:
				base.update(json.load(f))

		parser = argparse.ArgumentParser(description='run the full train/predict/compare pipeline for one or more models')
		parser.add_argument('--config', default=None, help='JSON file of defaults/sweeps; explicit flags below still override it')
		parser.add_argument('--report', nargs='?', const='latest', default=None, help='print a report instead of running (optionally a specific run timestamp)')
		parser.add_argument('--models', nargs='+', choices=sorted(MODEL_REGISTRY), default=base['models'])
		parser.add_argument('--epochs-per-checkpoint', type=int, default=base['epochs_per_checkpoint'])
		parser.add_argument('--num-checkpoints', type=int, default=base['num_checkpoints'])
		parser.add_argument('--batch-size', type=int, default=base['batch_size'])
		parser.add_argument('--lr', type=float, default=base['lr'])
		parser.add_argument('--num-eval-images', type=int, default=base['num_eval_images'])
		parser.add_argument('--seed', type=int, default=base['seed'])
		parser.add_argument('--output-root', default=base['output_root'])
		args = parser.parse_args(argv)

		values = dict(base)
		values.update({
			'models': args.models,
			'epochs_per_checkpoint': args.epochs_per_checkpoint,
			'num_checkpoints': args.num_checkpoints,
			'batch_size': args.batch_size,
			'lr': args.lr,
			'num_eval_images': args.num_eval_images,
			'seed': args.seed,
			'output_root': args.output_root,
		})

		return cls(values), args.report
