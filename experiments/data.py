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

		self.test_images = None
		self.eval_images = None
		self._train_images = None
		self._train_masks = None
		self._test_masks = None
		self._train_loader = None
		self._test_loader = None

		self.cpu_count = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))

	@staticmethod
	def set_seed(seed):
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)

	def build(self):
		# lists files + computes the split + eval-image subset -- always
		# needed (every model gets its eval grid re-rendered, resumed or
		# not). the actual DataLoaders are built lazily (see train_loader/
		# test_loader below): a fully-resumed run with nothing left to
		# train never needs them, and building them spawns worker processes
		# that are wasted work in that case.
		image_paths = sorted(list(paths.list_images(config.IMAGES_PATH)))
		mask_paths = sorted(list(paths.list_images(config.MASKS_PATH)))

		split = train_test_split(image_paths, mask_paths, test_size=config.TEST_SPLIT, random_state=self.seed)
		(self._train_images, self.test_images) = split[:2]
		(self._train_masks, self._test_masks) = split[2:]
		print(f'[INFO] found {len(self._train_images)} examples in the training set...')
		print(f'[INFO] found {len(self.test_images)} examples in the test set...')

		# same images get evaluated at every checkpoint and for every model,
		# so the grids/metrics/distributions are actually comparable
		rng = random.Random(self.seed)
		n = min(self.num_eval_images, len(self.test_images))
		self.eval_images = rng.sample(self.test_images, n)

		return self

	def _build_loaders(self):
		tfms = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)),
			transforms.ToTensor(),
		])
		train_ds = SegmentationDataset(imagePaths=self._train_images, maskPaths=self._train_masks, transforms=tfms)
		test_ds = SegmentationDataset(imagePaths=self.test_images, maskPaths=self._test_masks, transforms=tfms)

		self._train_loader = DataLoader(train_ds,
			shuffle=True,
			batch_size=self.batch_size,
			pin_memory=config.PIN_MEMORY,
			num_workers=self.cpu_count,
		)
		self._test_loader = DataLoader(test_ds,
			shuffle=False,
			batch_size=self.batch_size,
			pin_memory=config.PIN_MEMORY,
			num_workers=self.cpu_count,
		)

	@property
	def train_loader(self):
		if self._train_loader is None:
			self._build_loaders()
		return self._train_loader

	@property
	def test_loader(self):
		if self._test_loader is None:
			self._build_loaders()
		return self._test_loader
