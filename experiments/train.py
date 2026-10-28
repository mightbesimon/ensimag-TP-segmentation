# everything responsible for creating, identifying, checkpointing, and
# training a model: "create and store the model."

import hashlib
import json
import os
import time
from datetime import datetime

import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from tqdm import tqdm

TIMESTAMP_FORMAT = '%y%m%dT%H%M%S'


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

	def list_checkpoints(self, model_hash):
		entry = self._read_index().get(model_hash)
		return sorted(entry['checkpoints'], key=lambda c: c['epoch']) if entry else []

	def find_latest(self, model_hash):
		checkpoints = self.list_checkpoints(model_hash)
		return checkpoints[-1] if checkpoints else None

	def load_into(self, model, checkpoint_entry):
		ckpt = torch.load(checkpoint_entry['path'], weights_only=False)
		model.load_state_dict(ckpt['model_state_dict'])
		return ckpt

	def save(self, model, run_spec, model_hash, epoch, training_time_seconds, history):
		timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
		filename = f'{run_spec.model_key}_{model_hash}_ep{epoch}_{timestamp}.pth'
		path = os.path.join(self.dir, filename)

		torch.save({
			'model_state_dict': model.state_dict(),
			'epoch': epoch,
			'history': history,
		}, path)

		index = self._read_index()
		entry = index.setdefault(model_hash, {
			'model_key': run_spec.model_key,
			'class_name': run_spec.class_name,
			'enc_channels': list(run_spec.enc_channels),
			'dec_channels': list(run_spec.dec_channels),
			'checkpoints': [],
		})
		entry['checkpoints'].append({
			'epoch': epoch,
			'path': path,
			'timestamp': timestamp,
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
	def hash(model):
		# identifies a model by what it actually is, not by the config
		# numbers used to build it: a checkpoint is only safe to resume from
		# if the current code would produce a state_dict with the exact same
		# parameter names and shapes. catches cases like adding a final head
		# conv to HourGlass, where the enc/dec tuples stay the same but the
		# architecture doesn't.
		signature = sorted((name, tuple(p.shape)) for name, p in model.state_dict().items())
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
