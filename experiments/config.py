# layered config: experiment_config.json (project root) -> optional --config
# file -> explicit CLI flags, each layer overriding the previous. sweeps
# (varying num_features and depth per model) are JSON-only -- there's no
# clean flat-flag way to express a list of per-model sweep definitions.

import argparse
import json
import os
from dataclasses import dataclass

from model import config as model_config
from model.network import HourGlass, SimpleConv, UNet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_CONFIG_PATH = os.path.join(ROOT, 'experiment_config.json')

MODEL_REGISTRY = {
	'simpleconv': SimpleConv,
	'hourglass': HourGlass,
	'unet': UNet,
}

DEFAULT_CHANNELS = {
	'simpleconv': ((3, 16), (16, 1)),
	'hourglass': ((3, 16, 32, 64), (64, 32, 16, 1)),
	'unet': ((3, 16, 32, 64), (64, 32, 16, 1)),
}


@dataclass(frozen=True)
class RunSpec:
	model_key: str
	enc_channels: tuple
	dec_channels: tuple

	@property
	def model_cls(self):
		return MODEL_REGISTRY[self.model_key]

	@property
	def class_name(self):
		return self.model_cls.__name__

	@property
	def short_label(self):
		depth = len(self.enc_channels) - 1
		features = self.enc_channels[1] if len(self.enc_channels) > 1 else '?'
		return f'{self.class_name}(features={features}, depth={depth})'

	def build_model(self):
		return self.model_cls(self.enc_channels, self.dec_channels)


class ExperimentConfig:
	HARD_DEFAULTS = {
		'models': MODEL_REGISTRY.keys(),
		'epochs_per_checkpoint': model_config.NUM_EPOCHS,
		'num_checkpoints': 1,
		'batch_size': model_config.BATCH_SIZE,
		'lr': model_config.INIT_LR,
		'num_eval_images': 10,
		'seed': 22,
		'output_root': 'experiment_output',
		'sweeps': [],
	}

	def __init__(self, values, report_timestamp=None):
		self.values = values
		self.report_timestamp = report_timestamp

	def __getattr__(self, name):
		try:
			return self.values[name]
		except KeyError:
			raise AttributeError(name)

	def to_dict(self):
		return dict(self.values)

	@staticmethod
	def make_channels(num_features, depth):
		# e.g. num_features=16, depth=3 -> enc=(3,16,32,64), dec=(64,32,16,1)
		enc_channels = (3,) + tuple(num_features * 2 ** i for i in range(depth))
		dec_channels = tuple(reversed(enc_channels[1:])) + (1,)
		return enc_channels, dec_channels

	@staticmethod
	def default_run_specs(model_keys):
		return [RunSpec(key, *DEFAULT_CHANNELS[key]) for key in model_keys]

	@staticmethod
	def expand_sweep(sweep_defs):
		# sweep_defs: list of {'model': key, 'features': [...], 'depths': [...]}
		specs = []
		for entry in sweep_defs:
			key = entry['model']
			features_list = entry.get('features', [DEFAULT_CHANNELS[key][0][1]])
			depths_list = entry.get('depths', [len(DEFAULT_CHANNELS[key][0]) - 1])
			for num_features in features_list:
				for depth in depths_list:
					enc, dec = ExperimentConfig.make_channels(num_features, depth)
					specs.append(RunSpec(key, enc, dec))
		return specs

	def run_specs(self):
		if self.values['sweeps']:
			return self.expand_sweep(self.values['sweeps'])
		return self.default_run_specs(self.values['models'])

	def target_epochs(self):
		# e.g. [20, 40, 60] -- every run spec is driven to the same set of
		# target epochs so the final comparison is a fair, same-epoch one
		step = self.values['epochs_per_checkpoint']
		n = self.values['num_checkpoints']
		return [step * i for i in range(1, n + 1)]

	@classmethod
	def setup(cls, argv=None):
		pre = argparse.ArgumentParser(add_help=False)
		pre.add_argument('--config', default=None)
		pre.add_argument('--report', nargs='?', const='latest', default=None)
		known, _ = pre.parse_known_args(argv)

		base = dict(cls.HARD_DEFAULTS)
		if os.path.exists(ROOT_CONFIG_PATH):
			with open(ROOT_CONFIG_PATH) as f:
				base.update(json.load(f))
		if known.config:
			with open(known.config) as f:
				base.update(json.load(f))

		parser = argparse.ArgumentParser(description='run the full train/eval/compare pipeline for one or more models')
		parser.add_argument('--config', default=None, help='JSON file of defaults/sweeps; explicit flags below still override it')
		parser.add_argument('--report', nargs='?', const='latest', default=None, help='print a report instead of running (optionally a specific run timestamp)')
		parser.add_argument('--models', nargs='+', choices=MODEL_REGISTRY.keys(), default=base['models'])
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

		return cls(values, args.report)
