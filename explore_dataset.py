# USAGE
# python explore_dataset.py
#
# One-off look at the segmentation dataset (images/masks under
# model/config.py's IMAGES_PATH/MASKS_PATH) -- not part of the experiment
# system, just a quick way to actually see what's in the data instead of
# guessing: pairing integrity, image/mask sizes, class balance (foreground
# pixel ratio -- directly relevant to why a weak model like SimpleConv can
# collapse to "always predict background"), and sample grids with the mask
# overlaid on the image, including the smallest/largest-foreground examples.

import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths

from model import config

OUT_DIR = 'dataset_exploration'
SEED = 42


def pair_image_mask_paths():
	image_paths = sorted(list(paths.list_images(config.IMAGES_PATH)))
	mask_by_name = {os.path.basename(p): p for p in paths.list_images(config.MASKS_PATH)}

	pairs, missing = [], []
	for img_path in image_paths:
		name = os.path.basename(img_path)
		mask_path = mask_by_name.get(name)
		if mask_path is None:
			missing.append(name)
		else:
			pairs.append((img_path, mask_path))
	return pairs, missing


def report_sizes(sample):
	image_sizes = {cv2.imread(img).shape[:2] for img, _ in sample}
	mask_sizes = {cv2.imread(mask, 0).shape for _, mask in sample}
	print(f'[INFO] image sizes in this sample: {image_sizes}')
	print(f'[INFO] mask sizes in this sample: {mask_sizes}')


def compute_ratios(sample):
	# rows: (image_path, mask_path, foreground_pixel_ratio)
	rows, unique_vals = [], set()
	for img_path, mask_path in sample:
		mask = cv2.imread(mask_path, 0)
		rows.append((img_path, mask_path, float((mask > 127).mean())))
		unique_vals.update(np.unique(mask).tolist())
	return rows, unique_vals


def save_histogram(rows):
	ratios = np.array([r for _, _, r in rows])
	print(f'[INFO] foreground pixel ratio over {len(rows)} masks: '
		f'mean={ratios.mean():.3f} median={np.median(ratios):.3f} '
		f'min={ratios.min():.3f} max={ratios.max():.3f} std={ratios.std():.3f}')

	fig = plt.figure()
	plt.hist(ratios, bins=30)
	plt.title('foreground pixel ratio per mask (class balance)')
	plt.xlabel('fraction of pixels that are foreground')
	plt.ylabel('number of masks')
	fig.savefig(os.path.join(OUT_DIR, 'foreground_ratio_histogram.png'))
	plt.close(fig)
	print(f'[INFO] wrote {OUT_DIR}/foreground_ratio_histogram.png')


def save_grid(rows, filename, title):
	fig, axs = plt.subplots(nrows=len(rows), ncols=3, figsize=(7, 2.3 * len(rows)))
	if len(rows) == 1:
		axs = axs.reshape(1, 3)
	for col, col_title in enumerate(('image', 'mask', 'overlay')):
		axs[0, col].set_title(col_title)

	for row, (img_path, mask_path, ratio) in enumerate(rows):
		image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype('float32') / 255.0
		mask = cv2.imread(mask_path, 0)

		overlay = image.copy()
		fg = mask > 127
		overlay[fg] = overlay[fg] * 0.55 + np.array([1.0, 0.0, 0.0]) * 0.45

		axs[row, 0].imshow(image)
		axs[row, 1].imshow(mask, cmap='gray')
		axs[row, 2].imshow(overlay)
		axs[row, 0].set_ylabel(f'{os.path.basename(img_path)}\nfg={ratio:.2%}', fontsize=6, rotation=0, ha='right', va='center')
		for col in range(3):
			axs[row, col].set_xticks([])
			axs[row, col].set_yticks([])

	fig.suptitle(title)
	fig.tight_layout()
	fig.savefig(os.path.join(OUT_DIR, filename))
	plt.close(fig)
	print(f'[INFO] wrote {OUT_DIR}/{filename}')


if __name__ == '__main__':
	os.makedirs(OUT_DIR, exist_ok=True)
	rng = random.Random(SEED)

	pairs, missing = pair_image_mask_paths()
	print(f'[INFO] {len(pairs)} image/mask pairs found')
	if missing:
		print(f'[WARN] {len(missing)} images have no matching mask, e.g. {missing[:5]}')

	sample = rng.sample(pairs, min(300, len(pairs)))
	report_sizes(sample[:50])

	rows, unique_vals = compute_ratios(sample)
	shown_vals = sorted(unique_vals)[:20]
	print(f'[INFO] distinct pixel values seen in sampled masks: {shown_vals}{" ..." if len(unique_vals) > 20 else ""}')
	save_histogram(rows)

	rows_by_ratio = sorted(rows, key=lambda r: r[2])
	save_grid(rows_by_ratio[:4], 'smallest_foreground.png', 'smallest foreground (hardest-looking) examples')
	save_grid(rows_by_ratio[-4:], 'largest_foreground.png', 'largest foreground (easiest-looking) examples')
	save_grid(rng.sample(rows, min(8, len(rows))), 'random_sample.png', 'random sample of images + masks')

	print(f'[INFO] done -- see {OUT_DIR}/')
