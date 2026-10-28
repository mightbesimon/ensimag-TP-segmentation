# orchestration, following the pipeline:
#   normal:  config -> data -> train -> eval -> plot -> report
#   resume:  train (whatever's left) -> eval -> plot -> report
# (train/eval are interleaved per target epoch: see _train_and_eval below)

import json
import os
from datetime import datetime

from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm

from model import config as model_config
from experiments.dashboard import Dashboard, Status, TIMESTAMP_FORMAT
from experiments.data import DataManager
from experiments.plots import ComparisonPlots, PredictGridPlotter
from experiments.report import Report
from experiments.train import CheckpointManager, TrainingManager


class ModelRun:
	def __init__(self, run_spec, model, model_hash, model_dir, current_epoch, history, cumulative_time):
		self.run_spec = run_spec
		self.model = model
		self.model_hash = model_hash
		self.model_dir = model_dir
		self.current_epoch = current_epoch
		self.history = history
		self.cumulative_time = cumulative_time


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
		model_hash = TrainingManager.hash(model)
		model_dir = self._model_dir(run_spec, model_hash)
		os.makedirs(model_dir, exist_ok=True)

		arch_config_path = os.path.join(model_dir, 'config.json')
		if not os.path.exists(arch_config_path):
			with open(arch_config_path, 'w') as f:
				json.dump({
					'model_key': run_spec.model_key, 'class_name': run_spec.class_name,
					'enc_channels': list(run_spec.enc_channels), 'dec_channels': list(run_spec.dec_channels),
					'hash': model_hash,
				}, f, indent='\t')

		latest = self.checkpoint_mgr.find_latest(model_hash)
		if latest is not None:
			ckpt = self.checkpoint_mgr.load_into(model, latest)
			print(f'[INFO] {run_spec.short_label}: resuming from epoch {latest["epoch"]} (hash {model_hash})')
			return ModelRun(run_spec, model, model_hash, model_dir, latest['epoch'], ckpt['history'], latest['training_time_seconds'])
		return ModelRun(run_spec, model, model_hash, model_dir, 0, {'train_loss': [], 'test_loss': []}, 0.0)

	def _train_and_eval(self, run, target_epochs, data, grid_plotter, pbar):
		# train (whatever's left to reach each target) -> eval, per target
		existing = {c['epoch']: c for c in self.checkpoint_mgr.list_checkpoints(run.model_hash)}
		trainer = None
		checkpoints_out = []

		for target in target_epochs:
			if target in existing:
				if run.current_epoch != target:
					self.checkpoint_mgr.load_into(run.model, existing[target])
					run.current_epoch = target
					run.history = existing[target]['history']
					run.cumulative_time = existing[target]['training_time_seconds']
			elif run.current_epoch < target:
				if trainer is None:
					trainer = TrainingManager(run.model, data.train_loader, data.test_loader, self.device, self.cfg.lr)
				run.cumulative_time += trainer.train(target - run.current_epoch, pbar, run.history)
				run.current_epoch = target
				self.checkpoint_mgr.save(run.model, run.run_spec, run.model_hash, target, run.cumulative_time, run.history)
			else:
				# already past this target (e.g. a previous run used a
				# different epochs-per-checkpoint); can't go back, so this
				# particular target is simply unreachable
				continue

			results = grid_plotter.render_grid(run.model, data.eval_images, run.model_dir, run.current_epoch, f'{run.run_spec.short_label} - epoch {run.current_epoch}')
			checkpoints_out.append({'epoch': run.current_epoch, 'results': results})
			grid_plotter.render_training_curve(run.history, run.model_dir, run.current_epoch, run.run_spec.short_label)
			self._save_model_history(run)

		return checkpoints_out

	def _save_model_history(self, run):
		run.history['num_params'] = sum(p.numel() for p in run.model.parameters())
		run.history['training_time_seconds'] = run.cumulative_time
		run.history['bottleneck_shape'] = getattr(run.model, 'bottleneck_shape', None)
		with open(os.path.join(run.model_dir, 'history.json'), 'w') as f:
			json.dump(run.history, f, indent='\t')

	def run(self):
		target_epochs = self.cfg.target_epochs()
		run_specs = self.cfg.run_specs()

		print(f'[INFO] config: {json.dumps(self.cfg.to_dict())}')
		print(f'[INFO] target epochs: {target_epochs}')
		for spec in run_specs:
			print(f'[INFO] will train: {spec.short_label}  enc={spec.enc_channels} dec={spec.dec_channels}')

		DataManager.set_seed(self.cfg.seed)
		data = DataManager(self.cfg.batch_size, self.cfg.seed, self.cfg.num_eval_images).build()
		loss_fn = BCEWithLogitsLoss()
		grid_plotter = PredictGridPlotter(self.device, loss_fn)

		timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
		run_dir = os.path.join(self.out_root, timestamp)
		os.makedirs(run_dir, exist_ok=True)
		config_to_save = self.cfg.to_dict()
		config_to_save['run_specs'] = [
			{'model_key': s.model_key, 'enc_channels': list(s.enc_channels), 'dec_channels': list(s.dec_channels)}
			for s in run_specs
		]
		with open(os.path.join(run_dir, 'config.json'), 'w') as f:
			json.dump(config_to_save, f, indent='\t')

		# build+train one model fully before touching the next: nn.Module
		# init and DataLoader shuffling both draw from the global RNG, so
		# building every model upfront would make each model's result
		# depend on which/how-many sibling models run alongside it in this
		# same invocation. reseeding right before each model keeps every
		# model's result reproducible and independent of its siblings.
		pbar = tqdm(total=0, desc='experiment')
		runs = []
		all_results, manifest_entries = [], []
		for spec in run_specs:
			DataManager.set_seed(self.cfg.seed)
			run = self._prepare(spec)
			runs.append(run)
			pbar.total += max(0, target_epochs[-1] - run.current_epoch)
			pbar.refresh()

			print(f'[INFO] {run.run_spec.short_label}: targeting epochs {target_epochs}')
			checkpoints_out = self._train_and_eval(run, target_epochs, data, grid_plotter, pbar)
			Status.note_model(self.out_root, timestamp, run.run_spec, run.model_hash, checkpoints_out)
			Status.write_run_status(run_dir, 'training', runs)

			if not checkpoints_out:
				print(f'[WARN] {run.run_spec.short_label}: no checkpoints reached (hash {run.model_hash} already past every target), skipping from comparison')
				continue

			result = ModelResult(run.run_spec.short_label, run.model, run.run_spec, run.model_hash, run.history, run.history.get('num_params', 0), run.cumulative_time)
			result.checkpoints = checkpoints_out
			all_results.append(result)
			manifest_entries.append({
				'label': run.run_spec.short_label, 'model_key': run.run_spec.model_key, 'hash': run.model_hash,
				'enc_channels': list(run.run_spec.enc_channels), 'dec_channels': list(run.run_spec.dec_channels),
				'output_dir': os.path.relpath(run.model_dir, self.out_root),
			})
		pbar.close()

		with open(os.path.join(run_dir, 'manifest.json'), 'w') as f:
			json.dump({'config': self.cfg.to_dict(), 'models': manifest_entries}, f, indent='\t')

		Status.set_run_stage(self.out_root, timestamp, 'plotting')
		Status.write_run_status(run_dir, 'plotting', runs)
		if len(all_results) > 1:
			plots = ComparisonPlots(run_dir)
			plot_steps = [
				('training curves', lambda: plots.training_curves(all_results)),
				('final metrics bar', lambda: plots.final_metrics_bar(all_results)),
				('loss distribution', lambda: plots.loss_distribution_over_epochs(all_results)),
				('qualitative grid', lambda: plots.qualitative_grid(all_results, data.eval_images, self.device, loss_fn, max_images=5)),
				('summary table', lambda: plots.summary_table(all_results)),
			]
			for _, step in tqdm(plot_steps, desc='plot'):
				step()
			print(f'[INFO] comparison written to {run_dir}')

		Status.set_run_stage(self.out_root, timestamp, 'reporting')
		Status.write_run_status(run_dir, 'reporting', runs)
		report = Report.load(self.out_root, timestamp)
		report.save()
		report.print()

		Status.set_run_stage(self.out_root, timestamp, 'complete')
		Status.write_run_status(run_dir, 'complete', runs)
		Dashboard.build(self.out_root)

		print(f'[INFO] experiment outputs written to {run_dir}')
		return run_dir
