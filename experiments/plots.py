# this file just plots data -- it does not evaluate or train anything.

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from experiments.eval import Evaluator


class PredictGridPlotter:
	def __init__(self, device, loss_fn):
		self.device = device
		self.loss_fn = loss_fn

	def render_grid(self, model, eval_images, out_dir, epoch, label):
		results = [Evaluator.evaluate_image(model, path, self.device, self.loss_fn) for path in eval_images]
		best_idx = min(range(len(results)), key=lambda i: results[i]['bce'])
		worst_idx = max(range(len(results)), key=lambda i: results[i]['bce'])

		n = len(results)
		fig, axs = plt.subplots(nrows=n, ncols=4, figsize=(10, 2.5 * n))
		if n == 1:
			axs = axs.reshape(1, 4)
		for col, title in enumerate(('image', 'ground truth', 'predicted mask', 'predicted probability')):
			axs[0, col].set_title(title)

		for row, r in enumerate(results):
			axs[row, 0].imshow(r['orig'])
			axs[row, 1].imshow(r['gt_mask'], cmap='gray')
			axs[row, 2].imshow(r['pred_mask'], cmap='gray')
			axs[row, 3].imshow(r['pred_prob'], cmap='gray')

			row_label = f'{r["name"]}\nbce={r["bce"]:.3f} dice={r["dice"]:.2f} iou={r["iou"]:.2f}'
			edge_color = None
			if row == best_idx:
				row_label, edge_color = 'BEST\n' + row_label, 'tab:green'
			elif row == worst_idx:
				row_label, edge_color = 'WORST\n' + row_label, 'tab:red'

			axs[row, 0].set_ylabel(row_label, fontsize=7, rotation=0, ha='right', va='center')
			for col in range(4):
				if edge_color is not None:
					for spine in axs[row, col].spines.values():
						spine.set_edgecolor(edge_color)
						spine.set_linewidth(2.5)
				axs[row, col].set_xticks([])
				axs[row, col].set_yticks([])

		fig.suptitle(label)
		fig.tight_layout()
		os.makedirs(out_dir, exist_ok=True)
		fig.savefig(os.path.join(out_dir, f'predict_grid_ep{epoch}.png'))
		plt.close(fig)

		metrics_path = os.path.join(out_dir, f'predict_metrics_ep{epoch}.csv')
		with open(metrics_path, 'w', newline='') as f:
			writer = csv.DictWriter(f, fieldnames=['name', 'bce', 'dice', 'iou'])
			writer.writeheader()
			for r in results:
				writer.writerow({k: r[k] for k in ('name', 'bce', 'dice', 'iou')})

		return results

	def render_training_curve(self, history, out_dir, epoch, label):
		plt.style.use('ggplot')
		fig = plt.figure()
		plt.plot(history['train_loss'], 'o', ls='-', ms=4, linewidth=2.0, label='train loss')
		plt.plot(history['test_loss'], 'o', ls='-', ms=4, linewidth=2.0, label='test loss')
		plt.title(f'{label} training loss')
		plt.xlabel('epoch #')
		plt.ylabel('loss')
		plt.legend()
		os.makedirs(out_dir, exist_ok=True)
		fig.savefig(os.path.join(out_dir, f'training_curve_ep{epoch}.png'))
		plt.close(fig)


class ComparisonPlots:
	def __init__(self, out_dir):
		self.out_dir = out_dir
		os.makedirs(out_dir, exist_ok=True)

	def _path(self, filename):
		return os.path.join(self.out_dir, filename)

	def training_curves(self, model_results):
		fig = plt.figure()
		colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
		for color, res in zip(colors, model_results):
			epochs = range(1, len(res.history['train_loss']) + 1)
			plt.plot(epochs, res.history['train_loss'], color=color, linestyle='-', label=f'{res.label} train')
			plt.plot(epochs, res.history['test_loss'], color=color, linestyle='--', label=f'{res.label} test')
		plt.title('training loss comparison')
		plt.xlabel('epoch #')
		plt.ylabel('loss')
		plt.legend(fontsize=8)
		fig.savefig(self._path('training_curves.png'))
		plt.close(fig)

	def final_metrics_bar(self, model_results):
		metrics = ('bce', 'dice', 'iou')
		fig, axs = plt.subplots(ncols=len(metrics), figsize=(4 * len(metrics), 4))
		labels = [res.label for res in model_results]
		for ax, metric in zip(axs, metrics):
			final_results = [res.checkpoints[-1]['results'] for res in model_results]
			means = [np.mean([r[metric] for r in results]) for results in final_results]
			stds = [np.std([r[metric] for r in results]) for results in final_results]
			ax.bar(labels, means, yerr=stds, capsize=4)
			ax.set_title(metric)
			ax.tick_params(axis='x', labelrotation=20)
		fig.suptitle('final checkpoint: mean +/- std over eval images')
		fig.tight_layout()
		fig.savefig(self._path('final_metrics_bar.png'))
		plt.close(fig)

	def loss_distribution_over_epochs(self, model_results):
		all_epochs = sorted({c['epoch'] for res in model_results for c in res.checkpoints})
		n_models = len(model_results)
		width = 0.8 / n_models
		colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

		fig, ax = plt.subplots(figsize=(2.5 * len(all_epochs) + 2, 5))
		for m, res in enumerate(model_results):
			by_epoch = {c['epoch']: c['results'] for c in res.checkpoints}
			data = [[r['bce'] for r in by_epoch[e]] for e in all_epochs if e in by_epoch]
			positions = [i + (m - (n_models - 1) / 2) * width for i, e in enumerate(all_epochs) if e in by_epoch]
			bp = ax.boxplot(data, positions=positions, widths=width * 0.9, patch_artist=True)
			for box in bp['boxes']:
				box.set_facecolor(colors[m % len(colors)])
				box.set_label(res.label)
			for box in bp['boxes'][1:]:
				box.set_label('_nolegend_')

		ax.set_xticks(range(len(all_epochs)))
		ax.set_xticklabels(all_epochs)
		ax.set_xlabel('epoch #')
		ax.set_ylabel('per-image BCE loss')
		ax.set_title('loss distribution over epochs')
		ax.legend(fontsize=8)
		fig.tight_layout()
		fig.savefig(self._path('loss_distribution_over_epochs.png'))
		plt.close(fig)

	def qualitative_grid(self, model_results, eval_images, device, loss_fn, max_images):
		shown = eval_images[:max_images]
		n, n_models = len(shown), len(model_results)

		fig, axs = plt.subplots(nrows=n, ncols=2 + n_models, figsize=(2.2 * (2 + n_models), 2.5 * n))
		if n == 1:
			axs = axs.reshape(1, -1)
		axs[0, 0].set_title('image')
		axs[0, 1].set_title('ground truth')
		for m, res in enumerate(model_results):
			axs[0, 2 + m].set_title(res.label, fontsize=8)

		for row, path in enumerate(shown):
			r0 = Evaluator.evaluate_image(model_results[0].model, path, device, loss_fn)
			axs[row, 0].imshow(r0['orig'])
			axs[row, 1].imshow(r0['gt_mask'], cmap='gray')
			axs[row, 0].set_ylabel(r0['name'], fontsize=7, rotation=0, ha='right', va='center')

			for m, res in enumerate(model_results):
				r = Evaluator.evaluate_image(res.model, path, device, loss_fn)
				axs[row, 2 + m].imshow(r['pred_mask'], cmap='gray')
				axs[row, 2 + m].set_xlabel(f'bce={r["bce"]:.3f}', fontsize=7)

			for col in range(axs.shape[1]):
				axs[row, col].set_xticks([])
				axs[row, col].set_yticks([])

		fig.suptitle('qualitative comparison across models (final checkpoint)')
		fig.tight_layout()
		fig.savefig(self._path('qualitative_grid.png'))
		plt.close(fig)

	def summary_table(self, model_results):
		rows = []
		for res in model_results:
			final_results = res.checkpoints[-1]['results']
			rows.append({
				'model': res.label,
				'hash': res.model_hash,
				'num_params': res.num_params,
				'training_time_seconds': round(res.training_time_seconds, 2),
				'final_epoch': res.checkpoints[-1]['epoch'],
				'final_bce': round(float(np.mean([r['bce'] for r in final_results])), 4),
				'final_dice': round(float(np.mean([r['dice'] for r in final_results])), 4),
				'final_iou': round(float(np.mean([r['iou'] for r in final_results])), 4),
			})

		with open(self._path('summary.csv'), 'w', newline='') as f:
			writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
			writer.writeheader()
			writer.writerows(rows)

		fig, ax = plt.subplots(figsize=(11, 1 + 0.5 * len(rows)))
		ax.axis('off')
		columns = list(rows[0].keys())
		cell_text = [[str(row[c]) for c in columns] for row in rows]
		table = ax.table(cellText=cell_text, colLabels=columns, loc='center', cellLoc='center')
		table.auto_set_font_size(False)
		table.set_fontsize(8)
		table.scale(1, 1.6)
		fig.tight_layout()
		fig.savefig(self._path('summary_table.png'))
		plt.close(fig)
		return rows
