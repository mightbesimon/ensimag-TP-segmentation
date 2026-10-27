# Evaluator: given an already-built model, evaluate it on one image.
# does not build, train, or persist models, and does not plot anything --
# that's train.py's and plots.py's job respectively.

import os

import cv2
import numpy as np
import torch

from model import config


class Evaluator:
	@staticmethod
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

	@staticmethod
	def evaluate_image(model, image_path, device, loss_fn):
		model.eval()
		with torch.no_grad():
			image = cv2.imread(image_path)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			image = image.astype('float32') / 255.0
			image = cv2.resize(image, (config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT))
			orig = image.copy()

			filename = image_path.split(os.path.sep)[-1]
			ground_truth_path = os.path.join(config.MASKS_PATH, filename)
			gt_mask = cv2.imread(ground_truth_path, 0)
			gt_mask = cv2.resize(gt_mask, (config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT))
			gt_mask_norm = gt_mask.astype('float32') / 255.0

			image_tensor = np.transpose(image, (2, 0, 1))
			image_tensor = np.expand_dims(image_tensor, 0)
			image_tensor = torch.from_numpy(image_tensor).to(device)

			gt_tensor = torch.from_numpy(gt_mask_norm).unsqueeze(0).unsqueeze(0).to(device)

			pred_logits = model(image_tensor)
			bce_loss = float(loss_fn(pred_logits, gt_tensor).cpu().item())

			pred_mask = torch.sigmoid(pred_logits).squeeze().cpu().numpy()
			pred_prob = pred_mask * 255
			pred_binary = pred_mask > config.THRESHOLD
			pred_mask_img = (pred_binary * 255).astype(np.uint8)

			dice, iou = Evaluator.dice_iou(pred_binary, gt_mask > 127)

		return {
			'name': os.path.splitext(filename)[0],
			'orig': orig,
			'gt_mask': gt_mask,
			'pred_mask': pred_mask_img,
			'pred_prob': pred_prob,
			'bce': bce_loss,
			'dice': dice,
			'iou': iou,
		}
