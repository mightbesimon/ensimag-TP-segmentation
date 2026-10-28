# USAGE
# python dataset_videos.py
#
# One-off script: for each hardcoded video number below, stitches every
# frame of that video into an animated GIF with 4 panels per frame --
# image / mask / overlay / cropped-to-mask -- so you can watch the
# segmentation across an entire video instead of single frames.
# Re-running skips any GIF that already exists.

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw

from dataset_tools import crop_to_mask, group_by_video, overlay_mask, pair_image_mask_paths

# edit this list with whichever video numbers (the <video> in <video>_<frame>.png) you want to see
VIDEO_NUMBERS = [
	'00002',
	'00011',
	'00089',
	'00113',
]

OUT_DIR = 'dataset_exploration/videos'
FPS = 6
LABEL_HEIGHT = 16


def label_panel(array, text):
	img = Image.fromarray(array)
	canvas = Image.new('RGB', (img.width, img.height + LABEL_HEIGHT), 'white')
	canvas.paste(img, (0, LABEL_HEIGHT))
	ImageDraw.Draw(canvas).text((4, 2), text, fill='black')
	return canvas


def build_frame(image_path, mask_path):
	image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
	mask = cv2.imread(mask_path, 0)
	fg = mask > 127

	mask_rgb = np.stack([mask] * 3, axis=-1)
	overlay = overlay_mask(image, fg, color=(255, 0, 0))
	cropped = crop_to_mask(image, fg)

	panels = [
		label_panel(image, 'image'),
		label_panel(mask_rgb, 'mask'),
		label_panel(overlay, 'overlay'),
		label_panel(cropped, 'cropped'),
	]
	composite = Image.new('RGB', (sum(p.width for p in panels), panels[0].height), 'white')
	x = 0
	for p in panels:
		composite.paste(p, (x, 0))
		x += p.width
	return composite


def make_gif(video_number, frames_info, out_path):
	frames = [build_frame(img, mask) for img, mask, _ in frames_info]
	frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=int(1000 / FPS), loop=0)
	print(f'[INFO] wrote {out_path} ({len(frames)} frames)')


if __name__ == '__main__':
	os.makedirs(OUT_DIR, exist_ok=True)

	pairs, missing = pair_image_mask_paths()
	by_video = group_by_video(pairs)

	for video_number in VIDEO_NUMBERS:
		frames_info = by_video.get(video_number)
		if not frames_info:
			print(f'[WARN] no frames found for video {video_number}')
			continue

		out_path = os.path.join(OUT_DIR, f'{video_number}.gif')
		if os.path.exists(out_path):
			print(f'[INFO] {out_path} already exists, skipping')
			continue

		make_gif(video_number, frames_info, out_path)

	print(f'[INFO] done -- see {OUT_DIR}/')
