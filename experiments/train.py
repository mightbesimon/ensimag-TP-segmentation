# everything responsible for creating, identifying, checkpointing, and
# training a model: "create and store the model."

import hashlib
import json
import os
import time

import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from tqdm import tqdm


class CheckpointManager:
	def __init__(self, root):
		self.dir = os.path.join(root, 'checkpoints')
		self.index_path = os.path.join(self.dir, 'index.json')
		os.makedirs(self.dir, exist_ok=True)

	def _read_index(self):
		if not os.path.exists(self.index_path):
			return {}
		with open(self.index_path) as f:
			return json.load(f)

	def _write_index(self, index):
		with open(self.index_path, 'w') as f:
			json.dump(index, f, indent='\t')

	def list_checkpoints(self, model_hash, seed):
		# a hash identifies an architecture, not an experiment -- different
		# seeds of the same architecture share one hash, so checkpoints must
		# also be filtered by seed or two seeds' epoch-20 checkpoints would
		# collide when callers index this list by epoch alone
		checkpoints = self._read_index().get(model_hash, [])
		return sorted((c for c in checkpoints if c['seed'] == seed), key=lambda c: c['epoch'])

	def find_latest(self, model_hash, seed):
		checkpoints = self.list_checkpoints(model_hash, seed)
		return checkpoints[-1] if checkpoints else None

	def find_at_or_before(self, model_hash, seed, epoch):
		# for reporting on a past run as it actually was: that run may not be
		# the one that trained the latest checkpoint for this hash, so
		# "latest overall" is wrong -- this is "latest as of that run"
		candidates = [c for c in self.list_checkpoints(model_hash, seed) if c['epoch'] <= epoch]
		return candidates[-1] if candidates else None

	def load_into(self, model, checkpoint_entry):
		ckpt = torch.load(checkpoint_entry['path'], weights_only=False)
		model.load_state_dict(ckpt['model_state_dict'])
		return ckpt

	@staticmethod
	def read_history(checkpoint_entry):
		# just the history frozen inside a checkpoint at save time (no model
		# object needed) -- for Report, which doesn't build models
		return torch.load(checkpoint_entry['path'], weights_only=False)['history']

	def save(self, model, run_spec, model_hash, epoch, training_time_seconds, history, seed):
		filename = f'{run_spec.model_key}_{run_spec.arch_label}_ep{epoch}_s{seed}.pth'
		path = os.path.join(self.dir, filename)

		torch.save({
			'model_state_dict': model.state_dict(),
			'epoch': epoch,
			'history': history,
		}, path)

		# index[model_hash] is just the list of checkpoints for that
		# architecture -- model_key/class_name/enc_channels/dec_channels
		# aren't duplicated here since they're never read back from this
		# file; the model's own config.json is the single source of truth
		index = self._read_index()
		index.setdefault(model_hash, []).append({
			'epoch': epoch,
			'seed': seed,
			'path': path,
			'training_time_seconds': training_time_seconds,
		})
		self._write_index(index)
		return path


class TrainingManager:
	def __init__(self, model, train_loader, test_loader, device, lr):
		self.model = model
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.device = device
		self.loss_fn = BCEWithLogitsLoss()
		self.opt = Adam(model.parameters(), lr=lr)

		self.train_steps = max(1, len(train_loader.dataset) // train_loader.batch_size)
		self.test_steps = max(1, len(test_loader.dataset) // test_loader.batch_size)

	@staticmethod
	def architecture_signature(model):
		# identifies a model by what it actually is, not by the config
		# numbers used to build it. two parts:
		#  - 'parameters': every learnable weight's name + shape. catches
		#    cases like adding a final head conv to HourGlass, where the
		#    enc/dec tuples stay the same but the architecture doesn't.
		#  - 'modules': the full declared module tree (str(model), which
		#    PyTorch builds from each submodule's repr). parameters alone
		#    are blind to parameterless structural changes -- e.g. inserting
		#    a ReLU into Block adds no weights, so it wouldn't change the
		#    parameter signature at all, but it does change this.
		# neither part can see a pure forward()-logic change that touches no
		# declared submodule (e.g. reordering ops without adding/removing
		# layers) -- this is a strong signal, not an absolute guarantee.
		parameters = sorted((name, tuple(p.shape)) for name, p in model.state_dict().items())
		return {'modules': str(model), 'parameters': parameters}

	@staticmethod
	def hash_signature(signature):
		return hashlib.sha256(repr(signature).encode('utf-8')).hexdigest()[:8]

	def train(self, epochs, history, label=''):
		start_time = time.time()
		for _ in tqdm(range(epochs), desc=label):
			self.model.train()
			total_train_loss = 0
			total_test_loss = 0

			for (x, y) in self.train_loader:
				(x, y) = (x.to(self.device), y.to(self.device))

				pred = self.model(x)
				loss = self.loss_fn(pred, y)

				self.opt.zero_grad()
				loss.backward()
				self.opt.step()

				total_train_loss += loss

			with torch.no_grad():
				self.model.eval()
				for (x, y) in self.test_loader:
					(x, y) = (x.to(self.device), y.to(self.device))
					pred = self.model(x)
					total_test_loss += self.loss_fn(pred, y)

			avg_train_loss = total_train_loss / self.train_steps
			avg_test_loss = total_test_loss / self.test_steps
			history['train_loss'].append(float(avg_train_loss.cpu().detach().numpy()))
			history['test_loss'].append(float(avg_test_loss.cpu().detach().numpy()))

		return time.time() - start_time
