# regenerates comparison figures (and the report) for a past run entirely
# from what's on disk -- no training involved. Useful after a plotting
# change (new figure, fixed table width, ...) without re-running anything.

import glob
import json
import os
import re

from torch.nn import BCEWithLogitsLoss

from model import config as model_config
from experiments.config import RunSpec
from experiments.data import DataManager
from experiments.report import Report
from experiments.runner import ExperimentRunner, ModelResult
from experiments.train import CheckpointManager

METRIC_CSV_RE = re.compile(r'predict_metrics_ep(\d+)\.csv$')


class Replotter:
	@staticmethod
	def _rebuild_model_result(out_root, entry, device, final_target):
		# rebuilds a ModelResult entirely from what a previous run saved to
		# disk, scoped to the epoch THAT run reached -- a model's directory
		# is shared across every run that ever trained that hash further, so
		# naively reading "the latest checkpoint" or "every predict_metrics
		# csv" would include epochs trained by later runs too
		model_dir = os.path.join(out_root, entry['output_dir'])
		checkpoint_mgr = CheckpointManager(out_root)
		ckpt_entry = checkpoint_mgr.find_at_or_before(entry['hash'], final_target)

		checkpoints = []
		for path in sorted(glob.glob(os.path.join(model_dir, 'predict_metrics_ep*.csv'))):
			epoch = int(METRIC_CSV_RE.search(path).group(1))
			if epoch <= final_target:
				checkpoints.append({'epoch': epoch, 'results': Report.read_csv_rows(path)})
		checkpoints.sort(key=lambda c: c['epoch'])

		run_spec = RunSpec(entry['model_key'], tuple(entry['enc_channels']), tuple(entry['dec_channels']))
		model = run_spec.build_model().to(device)
		ckpt = checkpoint_mgr.load_into(model, ckpt_entry)
		num_params = sum(p.numel() for p in model.parameters())

		result = ModelResult(entry['label'], model, run_spec, entry['hash'], ckpt['history'], num_params, ckpt_entry['training_time_seconds'])
		result.checkpoints = checkpoints
		return result

	@classmethod
	def replot(cls, out_root, timestamp='latest'):
		if timestamp == 'latest':
			timestamp = Report.latest_timestamp(out_root)
			if timestamp is None:
				raise FileNotFoundError(f'no experiment runs with a manifest.json found under {out_root}')

		run_dir = os.path.join(out_root, timestamp)
		with open(os.path.join(run_dir, 'manifest.json')) as f:
			manifest = json.load(f)
		with open(os.path.join(run_dir, 'config.json')) as f:
			run_config = json.load(f)
		final_target = run_config['epochs_per_checkpoint'] * run_config['checkpoints']

		device = model_config.DEVICE
		all_results = [cls._rebuild_model_result(out_root, entry, device, final_target) for entry in manifest['models']]
		if len(all_results) <= 1:
			print(f'[WARN] {run_dir} has only {len(all_results)} model(s) -- comparison plots need at least 2, nothing to replot')
			return run_dir

		data = DataManager(run_config['batch_size'], run_config['seed'], run_config['num_eval_images']).build()
		loss_fn = BCEWithLogitsLoss()
		ExperimentRunner._plot_comparison(run_dir, all_results, data, device, loss_fn)

		report = Report.load(out_root, timestamp)
		report.save()
		print(f'[INFO] replotted {run_dir}')
		return run_dir
