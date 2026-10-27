# USAGE
# python experiment.py
# python experiment.py --models hourglass unet --epochs-per-checkpoint 10 --num-checkpoints 2
#
# Runs the whole train -> checkpoint -> predict -> compare pipeline for one or
# more models in a single process, so you don't have to hop between train.py
# and predict.py and manually swap the model in train.py each time.
#
# The dataloader/epoch-loop logic is copied from train.py and the per-image
# inference logic is copied from predict.py (adapted into reusable functions)
# rather than imported, so train.py/predict.py stay untouched.

import argparse
import csv
import json
import os
import random
import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from imutils import paths
from sklearn.model_selection import train_test_split
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from model import config
from model.dataset import SegmentationDataset
from model.network import HourGlass, SimpleConv, UNet

MODEL_REGISTRY = {
	'simpleconv': (SimpleConv, (3, 16), (16, 1)),
	'hourglass': (HourGlass, (3, 16, 32, 64), (64, 32, 16, 1)),
	'unet': (UNet, (3, 16, 32, 64), (64, 32, 16, 1)),
}


def parse_args():
	parser = argparse.ArgumentParser(description='run the full train/predict/compare pipeline for one or more models')
	parser.add_argument('--models', nargs='+', choices=sorted(MODEL_REGISTRY), default=sorted(MODEL_REGISTRY))
	parser.add_argument('--epochs-per-checkpoint', type=int, default=config.NUM_EPOCHS)
	parser.add_argument('--num-checkpoints', type=int, default=3)
	parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE)
	parser.add_argument('--lr', type=float, default=config.INIT_LR)
	parser.add_argument('--num-eval-images', type=int, default=10)
	parser.add_argument('--seed', type=int, default=22)
	parser.add_argument('--output-root', default='experiment')
	return parser.parse_args()


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)


def build_dataloaders(batch_size, seed):
	# copied from train.py: load image/mask paths, split, wrap in datasets/loaders
	imagePaths = sorted(list(paths.list_images(config.IMAGES_PATH)))
	maskPaths = sorted(list(paths.list_images(config.MASKS_PATH)))

	split = train_test_split(imagePaths, maskPaths, test_size=config.TEST_SPLIT, random_state=seed)
	(trainImages, testImages) = split[:2]
	(trainMasks, testMasks) = split[2:]

	tfms = transforms.Compose([
		transforms.ToPILImage(),
		transforms.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)),
		transforms.ToTensor(),
	])

	trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks, transforms=tfms)
	testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks, transforms=tfms)
	print(f'[INFO] found {len(trainDS)} examples in the training set...')
	print(f'[INFO] found {len(testDS)} examples in the test set...')

	trainLoader = DataLoader(trainDS, shuffle=True, batch_size=batch_size, pin_memory=config.PIN_MEMORY, num_workers=os.cpu_count())
	testLoader = DataLoader(testDS, shuffle=False, batch_size=batch_size, pin_memory=config.PIN_MEMORY, num_workers=os.cpu_count())

	return trainLoader, testLoader, testImages


def pick_eval_subset(testImages, num_eval_images, seed):
	# same images get evaluated at every checkpoint and for every model, so
	# the grids/metrics/distributions are actually comparable to each other
	rng = random.Random(seed)
	n = min(num_eval_images, len(testImages))
	return rng.sample(testImages, n)


def train_model(cnnet, trainLoader, testLoader, device, lr, epochs, pbar, history):
	# copied from train.py's epoch loop, parameterised so it can be called
	# repeatedly on the same model/optimizer to 'continue' training between
	# checkpoints, and reports into a shared global progress bar
	lossFunc = BCEWithLogitsLoss()
	opt = Adam(cnnet.parameters(), lr=lr)

	trainSteps = max(1, len(trainLoader.dataset) // trainLoader.batch_size)
	testSteps = max(1, len(testLoader.dataset) // testLoader.batch_size)

	startTime = time.time()
	for _ in range(epochs):
		cnnet.train()
		totalTrainLoss = 0
		totalTestLoss = 0

		for (x, y) in trainLoader:
			(x, y) = (x.to(device), y.to(device))

			pred = cnnet(x)
			loss = lossFunc(pred, y)

			opt.zero_grad()
			loss.backward()
			opt.step()

			totalTrainLoss += loss

		with torch.no_grad():
			cnnet.eval()
			for (x, y) in testLoader:
				(x, y) = (x.to(device), y.to(device))
				pred = cnnet(x)
				totalTestLoss += lossFunc(pred, y)

		avgTrainLoss = totalTrainLoss / trainSteps
		avgTestLoss = totalTestLoss / testSteps
		history['train_loss'].append(float(avgTrainLoss.cpu().detach().numpy()))
		history['test_loss'].append(float(avgTestLoss.cpu().detach().numpy()))

		pbar.update(1)

	return time.time() - startTime


def dice_iou(pred_bin, gt_bin):
	pred_bin = pred_bin.astype(bool)
	gt_bin = gt_bin.astype(bool)
	intersection = np.logical_and(pred_bin, gt_bin).sum()
	union = np.logical_or(pred_bin, gt_bin).sum()
	pred_sum = pred_bin.sum()
	gt_sum = gt_bin.sum()

	if union == 0:
		# both masks empty: perfect agreement, not a failure case
		return 1.0, 1.0

	dice = 2 * intersection / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 1.0
	iou = intersection / union
	return float(dice), float(iou)


def evaluate_image(model, imagePath, device, lossFunc):
	# adapted from predict.py's make_predictions: returns arrays + metrics
	# instead of saving its own standalone plot, so the caller can assemble
	# a combined grid and collect per-image metrics
	model.eval()
	with torch.no_grad():
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.astype('float32') / 255.0
		image = cv2.resize(image, (config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT))
		orig = image.copy()

		filename = imagePath.split(os.path.sep)[-1]
		groundTruthPath = os.path.join(config.MASKS_PATH, filename)
		gtMask = cv2.imread(groundTruthPath, 0)
		gtMask = cv2.resize(gtMask, (config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT))
		gtMaskNorm = (gtMask.astype('float32') / 255.0)

		imageTensor = np.transpose(image, (2, 0, 1))
		imageTensor = np.expand_dims(imageTensor, 0)
		imageTensor = torch.from_numpy(imageTensor).to(device)

		gtTensor = torch.from_numpy(gtMaskNorm).unsqueeze(0).unsqueeze(0).to(device)

		predLogits = model(imageTensor)
		bceLoss = float(lossFunc(predLogits, gtTensor).cpu().item())

		predMask = torch.sigmoid(predLogits).squeeze().cpu().numpy()
		predProb = predMask * 255
		predBinary = (predMask > config.THRESHOLD)
		predMaskImg = (predBinary * 255).astype(np.uint8)

		dice, iou = dice_iou(predBinary, gtMask > 127)

	return {
		'name': os.path.splitext(filename)[0],
		'orig': orig,
		'gt_mask': gtMask,
		'pred_mask': predMaskImg,
		'pred_prob': predProb,
		'bce': bceLoss,
		'dice': dice,
		'iou': iou,
	}


def run_predict_grid(model, eval_images, device, out_dir, label):
	os.makedirs(out_dir, exist_ok=True)
	lossFunc = BCEWithLogitsLoss()

	results = [evaluate_image(model, path, device, lossFunc) for path in eval_images]

	best_idx = min(range(len(results)), key=lambda i: results[i]['bce'])
	worst_idx = max(range(len(results)), key=lambda i: results[i]['bce'])

	n = len(results)
	fig, axs = plt.subplots(nrows=n, ncols=4, figsize=(10, 2.5 * n))
	if n == 1:
		axs = axs.reshape(1, 4)

	col_titles = ('image', 'ground truth', 'predicted mask', 'predicted probability')
	for col, title in enumerate(col_titles):
		axs[0, col].set_title(title)

	for row, r in enumerate(results):
		axs[row, 0].imshow(r['orig'])
		axs[row, 1].imshow(r['gt_mask'], cmap='gray')
		axs[row, 2].imshow(r['pred_mask'], cmap='gray')
		axs[row, 3].imshow(r['pred_prob'], cmap='gray')

		rowLabel = f'{r["name"]}\nbce={r["bce"]:.3f} dice={r["dice"]:.2f} iou={r["iou"]:.2f}'
		if row == best_idx:
			rowLabel = 'BEST\n' + rowLabel
			edgeColor = 'tab:green'
		elif row == worst_idx:
			rowLabel = 'WORST\n' + rowLabel
			edgeColor = 'tab:red'
		else:
			edgeColor = None

		axs[row, 0].set_ylabel(rowLabel, fontsize=7, rotation=0, ha='right', va='center')
		if edgeColor is not None:
			for col in range(4):
				for spine in axs[row, col].spines.values():
					spine.set_edgecolor(edgeColor)
					spine.set_linewidth(2.5)

		for col in range(4):
			axs[row, col].set_xticks([])
			axs[row, col].set_yticks([])

	fig.suptitle(label)
	fig.tight_layout()
	fig.savefig(os.path.join(out_dir, 'grid.png'))
	plt.close(fig)

	with open(os.path.join(out_dir, 'metrics.csv'), 'w', newline='') as f:
		writer = csv.DictWriter(f, fieldnames=['name', 'bce', 'dice', 'iou'])
		writer.writeheader()
		for r in results:
			writer.writerow({k: r[k] for k in ('name', 'bce', 'dice', 'iou')})

	return results


def save_checkpoint(cnnet, path, epoch, history):
	torch.save({
		'model_state_dict': cnnet.state_dict(),
		'epoch': epoch,
		'history': history,
	}, path)


def run_model_experiment(model_key, args, trainLoader, testLoader, eval_images, device, out_root, pbar):
	model_cls, default_enc, default_dec = MODEL_REGISTRY[model_key]
	model_name = model_cls.__name__
	model = model_cls(default_enc, default_dec).to(device)

	out_dir = os.path.join(out_root, f'output_{model_name}')
	os.makedirs(out_dir, exist_ok=True)

	history = {'train_loss': [], 'test_loss': []}
	num_params = sum(p.numel() for p in model.parameters())
	total_time = 0.0
	checkpoints = []

	print(f'[INFO] training {model_name}...')
	for ckpt_idx in range(args.num_checkpoints):
		elapsed = train_model(model, trainLoader, testLoader, device, args.lr, args.epochs_per_checkpoint, pbar, history)
		total_time += elapsed
		epoch = (ckpt_idx + 1) * args.epochs_per_checkpoint

		print(f'[INFO] {model_name}: epoch {epoch} done, +{elapsed:.2f}s (total {total_time:.2f}s)')

		ckpt_path = os.path.join(out_dir, f'checkpoint_epoch{epoch:04d}.pth')
		save_checkpoint(model, ckpt_path, epoch, history)

		predict_dir = os.path.join(out_dir, 'predict', f'epoch{epoch:04d}')
		results = run_predict_grid(model, eval_images, device, predict_dir, f'{model_name} - epoch {epoch}')
		checkpoints.append({'epoch': epoch, 'results': results})

		bottleneck = getattr(model, 'bottleneck_shape', None)
		if bottleneck is not None:
			print(f'[INFO] {model_name}: bottleneck shape (N, C, H, W) = {bottleneck}')

		history_path = os.path.join(out_dir, 'history.json')
		with open(history_path, 'w') as f:
			json.dump({
				'train_loss': history['train_loss'],
				'test_loss': history['test_loss'],
				'training_time_seconds': total_time,
				'num_params': num_params,
				'bottleneck_shape': bottleneck,
			}, f, indent='\t')

	plt.style.use('ggplot')
	fig = plt.figure()
	plt.plot(history['train_loss'], 'o', ls='-', ms=4, linewidth=2.0, label='train loss')
	plt.plot(history['test_loss'], 'o', ls='-', ms=4, linewidth=2.0, label='test loss')
	plt.title(f'{model_name} training loss')
	plt.xlabel('epoch #')
	plt.ylabel('loss')
	plt.legend()
	fig.savefig(os.path.join(out_dir, 'training_curve.png'))
	plt.close(fig)

	return {
		'name': model_name,
		'model': model,
		'history': history,
		'num_params': num_params,
		'training_time_seconds': total_time,
		'checkpoints': checkpoints,
		'bottleneck_shape': getattr(model, 'bottleneck_shape', None),
	}


def plot_training_curves_comparison(all_results, out_dir):
	fig = plt.figure()
	colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
	for color, res in zip(colors, all_results):
		epochs = range(1, len(res['history']['train_loss']) + 1)
		plt.plot(epochs, res['history']['train_loss'], color=color, linestyle='-', label=f'{res["name"]} train')
		plt.plot(epochs, res['history']['test_loss'], color=color, linestyle='--', label=f'{res["name"]} test')
	plt.title('training loss comparison')
	plt.xlabel('epoch #')
	plt.ylabel('loss')
	plt.legend(fontsize=8)
	fig.savefig(os.path.join(out_dir, 'training_curves.png'))
	plt.close(fig)


def plot_final_metrics_bar(all_results, out_dir):
	metrics = ('bce', 'dice', 'iou')
	fig, axs = plt.subplots(ncols=len(metrics), figsize=(4 * len(metrics), 4))

	names = [res['name'] for res in all_results]
	for ax, metric in zip(axs, metrics):
		final_results = [res['checkpoints'][-1]['results'] for res in all_results]
		means = [np.mean([r[metric] for r in results]) for results in final_results]
		stds = [np.std([r[metric] for r in results]) for results in final_results]
		ax.bar(names, means, yerr=stds, capsize=4)
		ax.set_title(metric)
		ax.tick_params(axis='x', labelrotation=20)

	fig.suptitle('final checkpoint: mean +/- std over eval images')
	fig.tight_layout()
	fig.savefig(os.path.join(out_dir, 'final_metrics_bar.png'))
	plt.close(fig)


def plot_loss_distribution_over_epochs(all_results, out_dir):
	num_checkpoints = len(all_results[0]['checkpoints'])
	epochs = [c['epoch'] for c in all_results[0]['checkpoints']]
	n_models = len(all_results)

	fig, ax = plt.subplots(figsize=(2.5 * num_checkpoints + 2, 5))
	width = 0.8 / n_models
	colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

	for m, res in enumerate(all_results):
		color = colors[m % len(colors)]
		data = [[r['bce'] for r in c['results']] for c in res['checkpoints']]
		positions = [i + (m - (n_models - 1) / 2) * width for i in range(num_checkpoints)]
		bp = ax.boxplot(data, positions=positions, widths=width * 0.9, patch_artist=True)
		for box in bp['boxes']:
			box.set_facecolor(color)
			box.set_label(res['name'])
		# avoid duplicate legend entries per box
		for box in bp['boxes'][1:]:
			box.set_label('_nolegend_')

	ax.set_xticks(range(num_checkpoints))
	ax.set_xticklabels(epochs)
	ax.set_xlabel('epoch #')
	ax.set_ylabel('per-image BCE loss')
	ax.set_title('loss distribution over epochs')
	ax.legend()
	fig.tight_layout()
	fig.savefig(os.path.join(out_dir, 'loss_distribution_over_epochs.png'))
	plt.close(fig)


def plot_qualitative_grid(all_results, eval_images, device, out_dir, max_images):
	lossFunc = BCEWithLogitsLoss()
	shown = eval_images[:max_images]
	n = len(shown)
	n_models = len(all_results)

	fig, axs = plt.subplots(nrows=n, ncols=1 + 1 + n_models, figsize=(2.2 * (2 + n_models), 2.5 * n))
	if n == 1:
		axs = axs.reshape(1, -1)

	axs[0, 0].set_title('image')
	axs[0, 1].set_title('ground truth')
	for m, res in enumerate(all_results):
		axs[0, 2 + m].set_title(res['name'])

	for row, path in enumerate(shown):
		r0 = evaluate_image(all_results[0]['model'], path, device, lossFunc)
		axs[row, 0].imshow(r0['orig'])
		axs[row, 1].imshow(r0['gt_mask'], cmap='gray')
		axs[row, 0].set_ylabel(r0['name'], fontsize=7, rotation=0, ha='right', va='center')

		for m, res in enumerate(all_results):
			r = evaluate_image(res['model'], path, device, lossFunc)
			axs[row, 2 + m].imshow(r['pred_mask'], cmap='gray')
			axs[row, 2 + m].set_xlabel(f'bce={r["bce"]:.3f}', fontsize=7)

		for col in range(axs.shape[1]):
			axs[row, col].set_xticks([])
			axs[row, col].set_yticks([])

	fig.suptitle('qualitative comparison across models (final checkpoint)')
	fig.tight_layout()
	fig.savefig(os.path.join(out_dir, 'qualitative_grid.png'))
	plt.close(fig)


def save_summary_table(all_results, out_dir):
	rows = []
	for res in all_results:
		final_results = res['checkpoints'][-1]['results']
		rows.append({
			'model': res['name'],
			'num_params': res['num_params'],
			'training_time_seconds': round(res['training_time_seconds'], 2),
			'bottleneck_shape': res['bottleneck_shape'],
			'final_bce': round(float(np.mean([r['bce'] for r in final_results])), 4),
			'final_dice': round(float(np.mean([r['dice'] for r in final_results])), 4),
			'final_iou': round(float(np.mean([r['iou'] for r in final_results])), 4),
		})

	with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
		writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
		writer.writeheader()
		writer.writerows(rows)

	fig, ax = plt.subplots(figsize=(10, 1 + 0.5 * len(rows)))
	ax.axis('off')
	columns = list(rows[0].keys())
	cellText = [[str(row[c]) for c in columns] for row in rows]
	table = ax.table(cellText=cellText, colLabels=columns, loc='center', cellLoc='center')
	table.auto_set_font_size(False)
	table.set_fontsize(8)
	table.scale(1, 1.6)
	fig.tight_layout()
	fig.savefig(os.path.join(out_dir, 'summary_table.png'))
	plt.close(fig)


if __name__ == '__main__':
	args = parse_args()
	set_seed(args.seed)

	timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
	out_root = os.path.join(args.output_root, timestamp)
	os.makedirs(out_root, exist_ok=True)

	with open(os.path.join(out_root, 'config.json'), 'w') as f:
		json.dump(vars(args), f, indent='\t')

	trainLoader, testLoader, testImages = build_dataloaders(args.batch_size, args.seed)
	eval_images = pick_eval_subset(testImages, args.num_eval_images, args.seed)

	total_epochs = len(args.models) * args.epochs_per_checkpoint * args.num_checkpoints
	pbar = tqdm(total=total_epochs, desc='experiment')

	all_results = []
	for model_key in args.models:
		result = run_model_experiment(model_key, args, trainLoader, testLoader, eval_images, config.DEVICE, out_root, pbar)
		all_results.append(result)
	pbar.close()

	if len(all_results) > 1:
		comparison_dir = os.path.join(out_root, 'comparison')
		os.makedirs(comparison_dir, exist_ok=True)
		plot_training_curves_comparison(all_results, comparison_dir)
		plot_final_metrics_bar(all_results, comparison_dir)
		plot_loss_distribution_over_epochs(all_results, comparison_dir)
		plot_qualitative_grid(all_results, eval_images, config.DEVICE, comparison_dir, max_images=5)
		save_summary_table(all_results, comparison_dir)
		print(f'[INFO] comparison written to {comparison_dir}')

	print(f'[INFO] experiment outputs written to {out_root}')
