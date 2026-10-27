# everything related to reporting: builds a full-detail report purely from
# what's on disk for a given run, so a past run's report can be reprinted
# without re-running anything.

import csv
import glob
import json
import os
import re

import numpy as np

METRIC_CSV_RE = re.compile(r'predict_metrics_ep(\d+)\.csv$')


class ModelReport:
	def __init__(self, manifest_entry, history, checkpoint_metrics):
		self.manifest_entry = manifest_entry
		self.history = history
		# checkpoint_metrics: list of (epoch, [{name, bce, dice, iou}, ...])
		self.checkpoint_metrics = checkpoint_metrics

	@property
	def label(self):
		return self.manifest_entry['label']

	def lowest_test_loss_epoch(self):
		test_loss = self.history['test_loss']
		if not test_loss:
			return None, None
		idx = int(np.argmin(test_loss))
		return idx + 1, test_loss[idx]

	def final_checkpoint(self):
		return self.checkpoint_metrics[-1] if self.checkpoint_metrics else (None, [])


class Report:
	def __init__(self, out_root, timestamp, manifest, model_reports):
		self.out_root = out_root
		self.timestamp = timestamp
		self.manifest = manifest
		self.model_reports = model_reports

	@staticmethod
	def _read_csv_rows(path):
		with open(path) as f:
			rows = list(csv.DictReader(f))
		for r in rows:
			r['bce'], r['dice'], r['iou'] = float(r['bce']), float(r['dice']), float(r['iou'])
		return rows

	@classmethod
	def latest_timestamp(cls, out_root):
		if not os.path.isdir(out_root):
			return None
		candidates = [
			d for d in os.listdir(out_root)
			if os.path.isdir(os.path.join(out_root, d)) and os.path.exists(os.path.join(out_root, d, 'manifest.json'))
		]
		return sorted(candidates)[-1] if candidates else None

	@classmethod
	def load(cls, out_root, timestamp='latest'):
		if timestamp == 'latest':
			timestamp = cls.latest_timestamp(out_root)
			if timestamp is None:
				raise FileNotFoundError(f'no experiment runs with a manifest.json found under {out_root}')

		run_dir = os.path.join(out_root, timestamp)
		with open(os.path.join(run_dir, 'manifest.json')) as f:
			manifest = json.load(f)

		model_reports = []
		for entry in manifest['models']:
			model_dir = os.path.join(out_root, entry['output_dir'])

			with open(os.path.join(model_dir, 'history.json')) as f:
				history = json.load(f)

			checkpoint_metrics = []
			for path in sorted(glob.glob(os.path.join(model_dir, 'predict_metrics_ep*.csv'))):
				epoch = int(METRIC_CSV_RE.search(path).group(1))
				checkpoint_metrics.append((epoch, cls._read_csv_rows(path)))
			checkpoint_metrics.sort(key=lambda t: t[0])

			model_reports.append(ModelReport(entry, history, checkpoint_metrics))

		return cls(out_root, timestamp, manifest, model_reports)

	def _lines(self):
		lines = [f'experiment report - run {self.timestamp}', f'config: {json.dumps(self.manifest["config"])}', '']

		lines.append('=== per-model checkpoint progression ===')
		for mr in self.model_reports:
			lines.append(f'-- {mr.label} (hash {mr.manifest_entry["hash"]}, {mr.history["num_params"]} params) --')
			for epoch, rows in mr.checkpoint_metrics:
				bce = np.mean([r['bce'] for r in rows])
				dice = np.mean([r['dice'] for r in rows])
				iou = np.mean([r['iou'] for r in rows])
				lines.append(f'  epoch {epoch:>4}: mean bce={bce:.4f} dice={dice:.4f} iou={iou:.4f}')
			best_epoch, best_loss = mr.lowest_test_loss_epoch()
			if best_epoch is not None:
				lines.append(f'  lowest test loss at epoch {best_epoch}: {best_loss:.4f}')
			lines.append(f'  cumulative training time: {mr.history["training_time_seconds"]:.2f}s')
			lines.append('')

		lines.append('=== final checkpoint summary ===')
		final_rows = []
		for mr in self.model_reports:
			epoch, rows = mr.final_checkpoint()
			if not rows:
				continue
			final_rows.append({
				'label': mr.label, 'epoch': epoch,
				'bce': float(np.mean([r['bce'] for r in rows])),
				'dice': float(np.mean([r['dice'] for r in rows])),
				'iou': float(np.mean([r['iou'] for r in rows])),
				'params': mr.history['num_params'],
				'training_time_seconds': mr.history['training_time_seconds'],
			})

		lines.append(f'{"model":<30} {"epoch":>6} {"params":>10} {"time(s)":>10} {"bce":>8} {"dice":>8} {"iou":>8}')
		for r in final_rows:
			lines.append(f'{r["label"]:<30} {r["epoch"]:>6} {r["params"]:>10} {r["training_time_seconds"]:>10.2f} {r["bce"]:>8.4f} {r["dice"]:>8.4f} {r["iou"]:>8.4f}')
		lines.append('')

		if final_rows:
			best_bce = min(final_rows, key=lambda r: r['bce'])
			best_dice = max(final_rows, key=lambda r: r['dice'])
			best_iou = max(final_rows, key=lambda r: r['iou'])
			lines.append('=== callouts ===')
			lines.append(f'best final bce:  {best_bce["label"]} ({best_bce["bce"]:.4f})')
			lines.append(f'best final dice: {best_dice["label"]} ({best_dice["dice"]:.4f})')
			lines.append(f'best final iou:  {best_iou["label"]} ({best_iou["iou"]:.4f})')

			best_img = worst_img = None
			for mr in self.model_reports:
				for epoch, rows in mr.checkpoint_metrics:
					for r in rows:
						tagged = {**r, 'model': mr.label, 'epoch': epoch}
						if best_img is None or tagged['bce'] < best_img['bce']:
							best_img = tagged
						if worst_img is None or tagged['bce'] > worst_img['bce']:
							worst_img = tagged
			if best_img is not None:
				lines.append(f'best single prediction overall:  {best_img["name"]} ({best_img["model"]}, epoch {best_img["epoch"]}, bce={best_img["bce"]:.4f})')
				lines.append(f'worst single prediction overall: {worst_img["name"]} ({worst_img["model"]}, epoch {worst_img["epoch"]}, bce={worst_img["bce"]:.4f})')

		return lines

	def print(self):
		for line in self._lines():
			print(line)

	def save(self):
		path = os.path.join(self.out_root, self.timestamp, 'report.txt')
		with open(path, 'w') as f:
			f.write('\n'.join(self._lines()) + '\n')
		return path
