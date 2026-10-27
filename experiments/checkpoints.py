# checkpoint storage shared across experiment runs (lives directly under
# experiment/checkpoints/, not inside a timestamped run folder), so a later
# run can find and resume a model trained by an earlier one.

import json
import os
from datetime import datetime

import torch


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
		index = self._read_index()
		entry = index.get(model_hash)
		if entry is None:
			return []
		return sorted(entry['checkpoints'], key=lambda c: c['epoch'])

	def find_latest(self, model_hash):
		checkpoints = self.list_checkpoints(model_hash)
		return checkpoints[-1] if checkpoints else None

	def load_into(self, model, checkpoint_entry):
		ckpt = torch.load(checkpoint_entry['path'], weights_only=False)
		model.load_state_dict(ckpt['model_state_dict'])
		return ckpt

	def save(self, model, run_spec, model_hash, epoch, training_time_seconds, history):
		timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
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
