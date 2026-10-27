# the epoch loop, copied from train.py and wrapped so it can be called
# repeatedly on the same model/optimizer to 'continue' training towards a
# target epoch count, reporting into a shared global progress bar.

import time

import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam


class Trainer:
	def __init__(self, model, train_loader, test_loader, device, lr):
		self.model = model
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.device = device
		self.loss_fn = BCEWithLogitsLoss()
		self.opt = Adam(model.parameters(), lr=lr)

		self.train_steps = max(1, len(train_loader.dataset) // train_loader.batch_size)
		self.test_steps = max(1, len(test_loader.dataset) // test_loader.batch_size)

	def train_epochs(self, epochs, pbar, history):
		start_time = time.time()
		for _ in range(epochs):
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

			pbar.update(1)

		return time.time() - start_time
