# top-level orchestration: for every configured run spec, resume-or-train up
# to the target epochs, predict at each new target, then build the
# cross-model comparison and report.

import json
import os
from datetime import datetime

from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm

from model import config as model_config
from experiments.checkpoints import CheckpointManager
from experiments.data import DataManager, set_seed
from experiments.hashing import hash_model
from experiments.plots import ComparisonPlots, PredictGridPlotter
from experiments.report import Report
from experiments.trainer import Trainer


class ModelResult:
	def __init__(self, label, model, run_spec, model_hash, history, num_params, training_time_seconds):
		self.label = label
		self.model = model
		self.run_spec = run_spec
		self.model_hash = model_hash
		self.history = history
		self.num_params = num_params
		self.training_time_seconds = training_time_seconds
		self.checkpoints = []  # [{'epoch': int, 'results': [...]}]


class _PreparedRun:
	def __init__(self, run_spec, model, model_hash, model_dir, current_epoch, history, cumulative_time):
		self.run_spec = run_spec
		self.model = model
		self.model_hash = model_hash
		self.model_dir = model_dir
		self.current_epoch = current_epoch
		self.history = history
		self.cumulative_time = cumulative_time


class ExperimentRunner:
	def __init__(self, cfg):
		self.cfg = cfg
		self.device = model_config.DEVICE
		self.out_root = cfg.output_root
		self.checkpoint_mgr = CheckpointManager(self.out_root)

	def _model_dir(self, run_spec, model_hash):
		return os.path.join(self.out_root, f'{run_spec.class_name}_{model_hash}')

	def _prepare(self, run_spec):
		model = run_spec.build_model().to(self.device)
		model_hash = hash_model(model)
		model_dir = self._model_dir(run_spec, model_hash)
		os.makedirs(model_dir, exist_ok=True)

		arch_config_path = os.path.join(model_dir, 'config.json')
		if not os.path.exists(arch_config_path):
			with open(arch_config_path, 'w') as f:
				json.dump({
					'model_key': run_spec.model_key,
					'class_name': run_spec.class_name,
					'enc_channels': list(run_spec.enc_channels),
					'dec_channels': list(run_spec.dec_channels),
					'hash': model_hash,
				}, f, indent='\t')

		latest = self.checkpoint_mgr.find_latest(model_hash)
		if latest is not None:
			ckpt = self.checkpoint_mgr.load_into(model, latest)
			current_epoch = latest['epoch']
			history = ckpt['history']
			cumulative_time = latest['training_time_seconds']
			print(f'[INFO] {run_spec.short_label}: resuming from epoch {current_epoch} (hash {model_hash})')
		else:
			current_epoch = 0
			history = {'train_loss': [], 'test_loss': []}
			cumulative_time = 0.0

		return _PreparedRun(run_spec, model, model_hash, model_dir, current_epoch, history, cumulative_time)

	def run(self):
		set_seed(self.cfg.seed)
		data = DataManager(self.cfg.batch_size, self.cfg.seed, self.cfg.num_eval_images).build()
		loss_fn = BCEWithLogitsLoss()
		grid_plotter = PredictGridPlotter(self.device, loss_fn)

		run_specs = self.cfg.run_specs()
		target_epochs = self.cfg.target_epochs()
		final_target = target_epochs[-1]

		prepared = [self._prepare(spec) for spec in run_specs]

		total_remaining = sum(max(0, final_target - p.current_epoch) for p in prepared)
		pbar = tqdm(total=total_remaining, desc='experiment')

		all_results = []
		manifest_entries = []

		for p in prepared:
			print(f'[INFO] {p.run_spec.short_label}: targeting epochs {target_epochs}')
			existing = {c['epoch']: c for c in self.checkpoint_mgr.list_checkpoints(p.model_hash)}
			trainer = None
			checkpoints_out = []

			for target in target_epochs:
				if target in existing:
					if p.current_epoch != target:
						self.checkpoint_mgr.load_into(p.model, existing[target])
						p.current_epoch = target
						p.history = existing[target]['history']
						p.cumulative_time = existing[target]['training_time_seconds']
					# else: already sitting at exactly this epoch in memory
				elif p.current_epoch < target:
					need = target - p.current_epoch
					if trainer is None:
						trainer = Trainer(p.model, data.train_loader, data.test_loader, self.device, self.cfg.lr)
					elapsed = trainer.train_epochs(need, pbar, p.history)
					p.cumulative_time += elapsed
					p.current_epoch = target
					self.checkpoint_mgr.save(p.model, p.run_spec, p.model_hash, target, p.cumulative_time, p.history)
				else:
					# current_epoch already past this target (e.g. a previous
					# run used a different epochs-per-checkpoint); can't go
					# back, so this particular target is simply unreachable
					continue

				results = grid_plotter.render(p.model, data.eval_images, p.model_dir, p.current_epoch, f'{p.run_spec.short_label} - epoch {p.current_epoch}')
				checkpoints_out.append({'epoch': p.current_epoch, 'results': results})
				self._save_training_curve(p)
				self._save_history(p)

			if not checkpoints_out:
				print(f'[WARN] {p.run_spec.short_label}: no checkpoints reached (hash {p.model_hash} already past every target), skipping from comparison')
				continue

			result = ModelResult(p.run_spec.short_label, p.model, p.run_spec, p.model_hash, p.history, p.history.get('num_params', 0), p.cumulative_time)
			result.checkpoints = checkpoints_out
			all_results.append(result)

			manifest_entries.append({
				'label': p.run_spec.short_label,
				'model_key': p.run_spec.model_key,
				'hash': p.model_hash,
				'enc_channels': list(p.run_spec.enc_channels),
				'dec_channels': list(p.run_spec.dec_channels),
				'output_dir': os.path.relpath(p.model_dir, self.out_root),
			})

		pbar.close()

		timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
		run_dir = os.path.join(self.out_root, timestamp)
		os.makedirs(run_dir, exist_ok=True)

		with open(os.path.join(run_dir, 'config.json'), 'w') as f:
			json.dump(self.cfg.to_dict(), f, indent='\t')

		with open(os.path.join(run_dir, 'manifest.json'), 'w') as f:
			json.dump({'config': self.cfg.to_dict(), 'models': manifest_entries}, f, indent='\t')

		if len(all_results) > 1:
			plots = ComparisonPlots(run_dir)
			plots.training_curves(all_results)
			plots.final_metrics_bar(all_results)
			plots.loss_distribution_over_epochs(all_results)
			plots.qualitative_grid(all_results, data.eval_images, self.device, loss_fn, max_images=5)
			plots.summary_table(all_results)
			print(f'[INFO] comparison written to {run_dir}')

		report = Report.load(self.out_root, timestamp)
		report.save()
		report.print()

		print(f'[INFO] experiment outputs written to {run_dir}')
		return run_dir

	def _save_history(self, p):
		bottleneck = getattr(p.model, 'bottleneck_shape', None)
		num_params = sum(param.numel() for param in p.model.parameters())
		p.history['num_params'] = num_params
		p.history['training_time_seconds'] = p.cumulative_time
		p.history['bottleneck_shape'] = bottleneck
		with open(os.path.join(p.model_dir, 'history.json'), 'w') as f:
			json.dump(p.history, f, indent='\t')

	def _save_training_curve(self, p):
		import matplotlib.pyplot as plt
		plt.style.use('ggplot')
		fig = plt.figure()
		plt.plot(p.history['train_loss'], 'o', ls='-', ms=4, linewidth=2.0, label='train loss')
		plt.plot(p.history['test_loss'], 'o', ls='-', ms=4, linewidth=2.0, label='test loss')
		plt.title(f'{p.run_spec.short_label} training loss')
		plt.xlabel('epoch #')
		plt.ylabel('loss')
		plt.legend()
		fig.savefig(os.path.join(p.model_dir, f'training_curve_ep{p.current_epoch}.png'))
		plt.close(fig)
