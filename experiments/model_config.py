# model registry + translating a sweep definition (feature counts x depths)
# into a flat list of concrete (enc_channels, dec_channels) run specs.
# Replaces the old standalone bottleneck_experiment.py.

from dataclasses import dataclass

from model.network import HourGlass, SimpleConv, UNet

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


def make_channels(num_features, depth):
	# e.g. num_features=16, depth=3 -> enc=(3,16,32,64), dec=(64,32,16,1),
	# the same symmetric scheme used throughout the rest of the project
	enc_channels = (3,) + tuple(num_features * 2 ** i for i in range(depth))
	dec_channels = tuple(reversed(enc_channels[1:])) + (1,)
	return enc_channels, dec_channels


def default_run_specs(model_keys):
	specs = []
	for key in model_keys:
		enc, dec = DEFAULT_CHANNELS[key]
		specs.append(RunSpec(key, enc, dec))
	return specs


def expand_sweep(sweep_defs):
	# sweep_defs: list of {'model': key, 'features': [...], 'depths': [...]}
	# nested loops generate the grid; everything downstream only ever sees
	# the resulting flat list of RunSpecs
	specs = []
	for entry in sweep_defs:
		key = entry['model']
		features_list = entry.get('features', [DEFAULT_CHANNELS[key][0][1]])
		depths_list = entry.get('depths', [len(DEFAULT_CHANNELS[key][0]) - 1])
		for num_features in features_list:
			for depth in depths_list:
				enc, dec = make_channels(num_features, depth)
				specs.append(RunSpec(key, enc, dec))
	return specs
