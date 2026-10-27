# dataloader/seed setup, copied and adapted from train.py's loading logic.

import os
import random

import numpy as np
import torch
from imutils import paths
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from model import config
from model.dataset import SegmentationDataset


class DataManager:
	def __init__(self, batch_size, seed, num_eval_images):
		self.batch_size = batch_size
		self.seed = seed
		self.num_eval_images = num_eval_images

		self.train_loader = None
		self.test_loader = None
		self.eval_images = None

	@staticmethod
	def set_seed(seed):
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)

	def build(self):
		image_paths = sorted(list(paths.list_images(config.IMAGES_PATH)))
		mask_paths = sorted(list(paths.list_images(config.MASKS_PATH)))

		split = train_test_split(image_paths, mask_paths, test_size=config.TEST_SPLIT, random_state=self.seed)
		(train_images, test_images) = split[:2]
		(train_masks, test_masks) = split[2:]

		tfms = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)),
			transforms.ToTensor(),
		])

		train_ds = SegmentationDataset(imagePaths=train_images, maskPaths=train_masks, transforms=tfms)
		test_ds = SegmentationDataset(imagePaths=test_images, maskPaths=test_masks, transforms=tfms)
		print(f'[INFO] found {len(train_ds)} examples in the training set...')
		print(f'[INFO] found {len(test_ds)} examples in the test set...')

		self.train_loader = DataLoader(train_ds, shuffle=True, batch_size=self.batch_size, pin_memory=config.PIN_MEMORY, num_workers=os.cpu_count())
		self.test_loader = DataLoader(test_ds, shuffle=False, batch_size=self.batch_size, pin_memory=config.PIN_MEMORY, num_workers=os.cpu_count())

		# same images get evaluated at every checkpoint and for every model,
		# so the grids/metrics/distributions are actually comparable
		rng = random.Random(self.seed)
		n = min(self.num_eval_images, len(test_images))
		self.eval_images = rng.sample(test_images, n)

		return self
