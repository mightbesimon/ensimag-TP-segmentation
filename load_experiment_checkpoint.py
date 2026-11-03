# USAGE
# python load_experiment_checkpoint.py experiment_output/checkpoints/simpleconv_ft16_d3_ep20_s22.pth
#
# Bridges an experiment.py checkpoint (state_dict + architecture stored
# separately) into the full-object format train.py/predict.py expect, by
# rebuilding the right model class, loading the weights in, then re-saving
# it to output/best_model.pth -- so predict.py then runs completely unchanged.
# Also rebuilds output/test_data.csv (the same seeded train/test split used
# for this checkpoint's run), since predict.py reads that too.

import json
import os
import re
import sys

import torch

from model import config
from experiments.config import MODEL_REGISTRY
from experiments.data import DataManager

# filenames are <model_key>_<arch_label>_ep<epoch>_s<seed>.pth, e.g.
# simpleconv_ft16_d3_ep20_s22.pth -- arch_label itself contains underscores,
# so this is parsed by pattern rather than by splitting on '_'
FILENAME_RE = re.compile(r'^(?P<model_key>[a-z0-9]+)_(?P<arch_label>ft\d+_d\d+)_ep(?P<epoch>\d+)_s(?P<seed>\d+)\.pth$')

if __name__ == '__main__':
	checkpoint_path = sys.argv[1]

	ckpt = torch.load(checkpoint_path, weights_only=False)

	filename = os.path.basename(checkpoint_path)
	match = FILENAME_RE.match(filename)
	if not match:
		raise ValueError(f'unrecognized checkpoint filename: {filename}')
	model_key, arch_label, seed = match['model_key'], match['arch_label'], int(match['seed'])

	out_root = os.path.dirname(os.path.dirname(checkpoint_path))
	class_name = MODEL_REGISTRY[model_key].__name__
	arch_config_path = os.path.join(out_root, f'{class_name}_{arch_label}', 'config.json')
	with open(arch_config_path) as f:
		arch = json.load(f)

	model = MODEL_REGISTRY[model_key](tuple(arch['enc_channels']), tuple(arch['dec_channels']))
	model.load_state_dict(ckpt['model_state_dict'])
	model = model.to(config.DEVICE)

	os.makedirs(config.BASE_OUTPUT, exist_ok=True)
	torch.save(model, config.BEST_MODEL_PATH)
	print(f'[INFO] wrote full model object to {config.BEST_MODEL_PATH} (epoch {ckpt["epoch"]})')

	data = DataManager(batch_size=1, seed=seed, num_eval_images=1).build()
	with open(config.TEST_PATH, 'w') as f:
		f.write('\n'.join(data.test_images))
	print(f'[INFO] wrote {config.TEST_PATH} ({len(data.test_images)} images, seed {seed})')

	print('[INFO] you can now run: python3 predict.py')
