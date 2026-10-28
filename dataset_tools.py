# shared helpers for the one-off dataset-exploration scripts (explore_dataset.py,
# dataset_videos.py) -- not part of the experiments/ training system.

import os

import numpy as np
from imutils import paths

from model import config


def parse_video_frame(path):
	# filenames are <video>_<frame>.png, e.g. 00037_0015.png
	name = os.path.splitext(os.path.basename(path))[0]
	video, frame = name.split('_', 1)
	return video, int(frame)


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


def group_by_video(pairs):
	# video number -> [(image_path, mask_path, frame_number), ...], sorted by frame
	by_video = {}
	for img_path, mask_path in pairs:
		video, frame = parse_video_frame(img_path)
		by_video.setdefault(video, []).append((img_path, mask_path, frame))
	for frames in by_video.values():
		frames.sort(key=lambda t: t[2])
	return by_video


def overlay_mask(image, mask_binary, color, alpha=0.45):
	# color must be in the same range as image (0..1 floats or 0..255 ints)
	orig_dtype = image.dtype
	result = image.astype('float32').copy()
	result[mask_binary] = result[mask_binary] * (1 - alpha) + np.array(color, dtype='float32') * alpha
	if orig_dtype.kind == 'u':
		result = result.clip(0, 255)
	return result.astype(orig_dtype)


def crop_to_mask(image, mask_binary):
	# literally removes every pixel not in the mask (sets it to 0)
	cropped = image.copy()
	cropped[~mask_binary] = 0
	return cropped
