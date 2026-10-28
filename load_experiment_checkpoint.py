# USAGE
# python load_experiment_checkpoint.py experiment_output/checkpoints/hourglass_847b376e_ep20_<timestamp>.pth
#
# Bridges an experiment.py checkpoint (state_dict + architecture stored
# separately) into the full-object format train.py/predict.py expect, by
# rebuilding the right model class, loading the weights in, then re-saving
# it to output/best_model.pth -- so predict.py then runs completely unchanged.
# Also rebuilds output/test_data.csv (the same seeded train/test split used
# for this checkpoint's most recent run), since predict.py reads that too.

import json
import os
import sys

import torch

from model import config
from experiments.config import MODEL_REGISTRY
from experiments.data import DataManager


def _seed_used_for(out_root, model_hash):
	with open(os.path.join(out_root, 'status.json')) as f:
		status = json.load(f)
	run_timestamps = status['models'][model_hash]['runs']
	latest_run = sorted(run_timestamps)[-1]
	with open(os.path.join(out_root, latest_run, 'config.json')) as f:
		return json.load(f)['seed']


def main(checkpoint_path):
	ckpt = torch.load(checkpoint_path, weights_only=False)

	# filenames are <model_key>_<hash>_ep<epoch>_<timestamp>.pth
	filename = os.path.basename(checkpoint_path)
	model_key, model_hash = filename.split('_')[:2]

	out_root = os.path.dirname(os.path.dirname(checkpoint_path))
	class_name = MODEL_REGISTRY[model_key].__name__
	arch_config_path = os.path.join(out_root, f'{class_name}_{model_hash}', 'config.json')
	with open(arch_config_path) as f:
		arch = json.load(f)

	model = MODEL_REGISTRY[model_key](tuple(arch['enc_channels']), tuple(arch['dec_channels']))
	model.load_state_dict(ckpt['model_state_dict'])
	model = model.to(config.DEVICE)

	os.makedirs(config.BASE_OUTPUT, exist_ok=True)
	torch.save(model, config.BEST_MODEL_PATH)
	print(f'[INFO] wrote full model object to {config.BEST_MODEL_PATH} (epoch {ckpt["epoch"]})')

	seed = _seed_used_for(out_root, model_hash)
	data = DataManager(batch_size=1, seed=seed, num_eval_images=1).build()
	with open(config.TEST_PATH, 'w') as f:
		f.write('\n'.join(data.test_images))
	print(f'[INFO] wrote {config.TEST_PATH} ({len(data.test_images)} images, seed {seed})')

	print('[INFO] you can now run: python3 predict.py')


if __name__ == '__main__':
	main(sys.argv[1])
