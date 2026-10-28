# USAGE
# python explore_dataset.py
#
# One-off look at the segmentation dataset (images/masks under
# model/config.py's IMAGES_PATH/MASKS_PATH) -- not part of the experiment
# system, just a quick way to actually see what's in the data instead of
# guessing: pairing integrity, image/mask sizes, class balance (foreground
# pixel ratio -- directly relevant to why a weak model like SimpleConv can
# collapse to "always predict background"), sample grids (image / mask /
# overlay / cropped-to-mask), and a collage of every video's first frame.
# Re-running skips any output file that already exists.

import math
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

from dataset_tools import crop_to_mask, group_by_video, overlay_mask, pair_image_mask_paths

OUT_DIR = 'dataset_exploration'
SEED = 42


def skipping(path):
	exists = os.path.exists(path)
	if exists:
		print(f'[INFO] {path} already exists, skipping')
	return exists


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


def save_histogram(rows, path):
	ratios = np.array([r for _, _, r in rows])
	print(f'[INFO] foreground pixel ratio over {len(rows)} masks: '
		f'mean={ratios.mean():.3f} median={np.median(ratios):.3f} '
		f'min={ratios.min():.3f} max={ratios.max():.3f} std={ratios.std():.3f}')

	fig = plt.figure()
	plt.hist(ratios, bins=30)
	plt.title('foreground pixel ratio per mask (class balance)')
	plt.xlabel('fraction of pixels that are foreground')
	plt.ylabel('number of masks')
	fig.savefig(path)
	plt.close(fig)
	print(f'[INFO] wrote {path}')


def save_grid(rows, path, title):
	fig, axs = plt.subplots(nrows=len(rows), ncols=4, figsize=(9, 2.3 * len(rows)))
	if len(rows) == 1:
		axs = axs.reshape(1, 4)
	for col, col_title in enumerate(('image', 'mask', 'overlay', 'cropped to mask')):
		axs[0, col].set_title(col_title)

	for row, (img_path, mask_path, ratio) in enumerate(rows):
		image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype('float32') / 255.0
		mask = cv2.imread(mask_path, 0)
		fg = mask > 127

		axs[row, 0].imshow(image)
		axs[row, 1].imshow(mask, cmap='gray')
		axs[row, 2].imshow(overlay_mask(image, fg, color=(1.0, 0.0, 0.0)))
		axs[row, 3].imshow(crop_to_mask(image, fg))
		axs[row, 0].set_ylabel(f'{os.path.basename(img_path)}\nfg={ratio:.2%}', fontsize=6, rotation=0, ha='right', va='center')
		for col in range(4):
			axs[row, col].set_xticks([])
			axs[row, col].set_yticks([])

	fig.suptitle(title)
	fig.tight_layout()
	fig.savefig(path)
	plt.close(fig)
	print(f'[INFO] wrote {path}')


def save_video_first_frames_collage(pairs, path):
	by_video = group_by_video(pairs)
	first_frames = [(video, frames[0][0]) for video, frames in sorted(by_video.items())]
	n = len(first_frames)
	cols = math.ceil(math.sqrt(n))
	rows = math.ceil(n / cols)

	fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 1.4, rows * 1.4))
	axs = np.array(axs).reshape(-1)
	for ax, (video, img_path) in zip(axs, first_frames):
		image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
		ax.imshow(image)
		ax.set_title(video, fontsize=5)
	for ax in axs:
		ax.set_xticks([])
		ax.set_yticks([])

	fig.suptitle(f'first frame of each of {n} videos')
	fig.tight_layout()
	fig.savefig(path, dpi=150)
	plt.close(fig)
	print(f'[INFO] wrote {path} ({n} videos)')


if __name__ == '__main__':
	os.makedirs(OUT_DIR, exist_ok=True)
	rng = random.Random(SEED)

	pairs, missing = pair_image_mask_paths()
	print(f'[INFO] {len(pairs)} image/mask pairs found')
	if missing:
		print(f'[WARN] {len(missing)} images have no matching mask, e.g. {missing[:5]}')

	histogram_path = os.path.join(OUT_DIR, 'foreground_ratio_histogram.png')
	smallest_path = os.path.join(OUT_DIR, 'smallest_foreground.png')
	largest_path = os.path.join(OUT_DIR, 'largest_foreground.png')
	random_path = os.path.join(OUT_DIR, 'random_sample.png')
	collage_path = os.path.join(OUT_DIR, 'video_first_frames.png')

	need_histogram = not os.path.exists(histogram_path)
	need_smallest = not os.path.exists(smallest_path)
	need_largest = not os.path.exists(largest_path)

	if not (need_histogram or need_smallest or need_largest):
		print(f'[INFO] {histogram_path}, {smallest_path}, {largest_path} all exist, skipping')
	else:
		sample = rng.sample(pairs, min(300, len(pairs)))
		report_sizes(sample[:50])
		rows, unique_vals = compute_ratios(sample)
		shown_vals = sorted(unique_vals)[:20]
		print(f'[INFO] distinct pixel values seen in sampled masks: {shown_vals}{" ..." if len(unique_vals) > 20 else ""}')

		if need_histogram:
			save_histogram(rows, histogram_path)
		else:
			print(f'[INFO] {histogram_path} already exists, skipping')

		rows_by_ratio = sorted(rows, key=lambda r: r[2])
		if need_smallest:
			save_grid(rows_by_ratio[:4], smallest_path, 'smallest foreground (hardest-looking) examples')
		else:
			print(f'[INFO] {smallest_path} already exists, skipping')
		if need_largest:
			save_grid(rows_by_ratio[-4:], largest_path, 'largest foreground (easiest-looking) examples')
		else:
			print(f'[INFO] {largest_path} already exists, skipping')

	if not skipping(random_path):
		random_sample = rng.sample(pairs, min(8, len(pairs)))
		random_rows = [(img, mask, float((cv2.imread(mask, 0) > 127).mean())) for img, mask in random_sample]
		save_grid(random_rows, random_path, 'random sample of images + masks')

	if not skipping(collage_path):
		save_video_first_frames_collage(pairs, collage_path)

	print(f'[INFO] done -- see {OUT_DIR}/')
