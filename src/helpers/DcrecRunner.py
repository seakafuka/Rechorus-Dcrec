# -*- coding: UTF-8 -*-

import os
import gc
import torch
import torch.nn as nn
import logging
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List

from utils import utils
from models.BaseModel import BaseModel


class DcrecRunner(object):
	@staticmethod
	def parse_runner_args(parser):
		parser.add_argument('--epoch', type=int, default=5,
							help='Number of epochs.')
		parser.add_argument('--check_epoch', type=int, default=1,
							help='Check some tensors every check_epoch.')
		parser.add_argument('--test_epoch', type=int, default=1,
							help='Print test results every test_epoch (-1 means no print).')
		parser.add_argument('--early_stop', type=int, default=10,
							help='The number of epochs when dev results drop continuously.')
		parser.add_argument('--lr', type=float, default=1e-3,
							help='Learning rate.')
		parser.add_argument('--l2', type=float, default=0,
							help='Weight decay in optimizer.')
		parser.add_argument('--batch_size', type=int, default=256,
							help='Batch size during training.')
		parser.add_argument('--eval_batch_size', type=int, default=256,
							help='Batch size during testing.')
		parser.add_argument('--optimizer', type=str, default='Adam',
							help='optimizer: SGD, Adam, Adagrad, Adadelta')
		parser.add_argument('--num_workers', type=int, default=0,
							help='Number of processors when prepare batches in DataLoader')
		parser.add_argument('--pin_memory', type=int, default=0,
							help='pin_memory in DataLoader')
		parser.add_argument('--topk', type=str, default='5,10,20,50',
							help='The number of items recommended to each user.')
		parser.add_argument('--metric', type=str, default='NDCG,HR',
							help='metrics: NDCG, HR')
		parser.add_argument('--main_metric', type=str, default='',
							help='Main metric to determine the best model.')
		return parser

	@staticmethod
	def evaluate_method(predictions: np.ndarray, topk: list, metrics: list) -> Dict[str, float]:
		"""
		:param predictions: (-1, n_candidates) shape, the first column is the score for ground-truth item
		:param topk: top-K value list
		:param metrics: metric string list
		:return: a result dict, the keys are metric@topk
		"""
		evaluations = dict()		
		gt_rank = (predictions >= predictions[:,0].reshape(-1,1)).sum(axis=-1)
		for k in topk:
			hit = (gt_rank <= k)
			for metric in metrics:
				key = '{}@{}'.format(metric, k)
				if metric == 'HR':
					evaluations[key] = hit.mean()
				elif metric == 'NDCG':
					evaluations[key] = (hit / np.log2(gt_rank + 1)).mean()
				else:
					raise ValueError('Undefined evaluation metric: {}.'.format(metric))
		return evaluations

	def __init__(self, args):
		self.train_models = args.train
		self.epoch = args.epoch
		self.check_epoch = args.check_epoch
		self.test_epoch = args.test_epoch
		self.early_stop = args.early_stop
		self.learning_rate = args.lr
		self.batch_size = args.batch_size
		self.eval_batch_size = args.eval_batch_size
		self.l2 = args.l2
		self.optimizer_name = args.optimizer
		self.num_workers = args.num_workers
		self.pin_memory = args.pin_memory
		self.topk = [int(x) for x in args.topk.split(',')]
		self.metrics = [m.strip().upper() for m in args.metric.split(',')]
		self.main_metric = '{}@{}'.format(self.metrics[0], self.topk[0]) if not len(args.main_metric) else args.main_metric # early stop based on main_metric
		self.main_topk = int(self.main_metric.split("@")[1])
		self.time = None  # will store [start_time, last_step_time]

		self.log_path = os.path.dirname(args.log_file) # path to save predictions
		self.save_appendix = args.log_file.split("/")[-1].split(".")[0] # appendix for prediction saving

	def _check_time(self, start=False):
		if self.time is None or start:
			self.time = [time()] * 2
			return self.time[0]
		tmp_time = self.time[1]
		self.time[1] = time()
		return self.time[1] - tmp_time

	def _build_optimizer(self, model):
		logging.info('Optimizer: ' + self.optimizer_name)
		optimizer = eval('torch.optim.{}'.format(self.optimizer_name))(
			model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2)
		return optimizer

	def train(self, data_dict: Dict[str, BaseModel.Dataset]):
		model = data_dict['train'].model
		main_metric_results, dev_results = list(), list()
		self._check_time(start=True)
		try:
			for epoch in range(self.epoch):
				# Fit
				self._check_time()
				gc.collect()
				torch.cuda.empty_cache()
				loss = self.fit(data_dict['train'], epoch=epoch + 1)
				print("loss",loss)
				if np.isnan(loss):
					logging.info("Loss is Nan. Stop training at %d."%(epoch+1))
					break
				training_time = self._check_time()

				# Observe selected tensors
				if len(model.check_list) > 0 and self.check_epoch > 0 and epoch % self.check_epoch == 0:
					utils.check(model.check_list)
				logging_str = ""
				if self.test_epoch > 0 and epoch % self.test_epoch  == 0:
					test_result = self.evaluate(data_dict['test'], self.topk, self.metrics)
					logging_str += ' test=({})'.format(utils.format_metric(test_result))
				testing_time = self._check_time()
				logging_str += ' [{:<.1f} s]'.format(testing_time)
				logging.info(logging_str)
		except KeyboardInterrupt:
			logging.info("Early stop manually")
			exit_here = input("Exit completely without evaluation? (y/n) (default n):")
			if exit_here.lower().startswith('y'):
				logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
				exit(1)

	def fit(self, dataset: BaseModel.Dataset, epoch=-1) -> float:
		model = dataset.model
		if model.optimizer is None:
			model.optimizer = self._build_optimizer(model)
		dataset.actions_before_epoch()  # must sample before multi thread start
		
		model.train()
		loss_lst = list()
		dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
						collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)
		for batch in tqdm(dl, leave=False, desc='Epoch {:<3}'.format(epoch), ncols=100, mininterval=1):
			batch = utils.batch_to_gpu(batch, model.device)
			item_ids = batch['item_id'].to('cuda:0')  # Ensure item_ids are on the correct device
			# print("itemid",item_ids.shape)
			indices = torch.argsort(torch.rand(*item_ids.shape), dim=-1).to('cuda:0')  # Move indices to the same device
			# Use torch.gather to shuffle the items based on indices
			batch['item_id'] = torch.gather(item_ids, dim=-1, index=indices)
			model.optimizer.zero_grad()
			out_dict = model(batch)
			loss,_ = model.loss(batch)
			loss.backward()
			model.optimizer.step()
			loss_lst.append(loss.detach().cpu().data.numpy())
		return np.mean(loss_lst).item()

	def eval_termination(self, criterion: List[float]) -> bool:
		if len(criterion) > self.early_stop and utils.non_increasing(criterion[-self.early_stop:]):
			return True
		elif len(criterion) - criterion.index(max(criterion)) > self.early_stop:
			return True
		return False

	def evaluate(self, dataset: BaseModel.Dataset, topks: list, metrics: list) -> Dict[str, float]:
		"""
		Evaluate the results for an input dataset.
		:return: result dict (key: metric@k)
		"""
		predictions = self.predict(dataset)
		return self.evaluate_method(predictions, topks, metrics)

	def predict(self, dataset: BaseModel.Dataset, save_prediction: bool = False) -> np.ndarray:
		"""
		The returned prediction is a 2D-array, each row corresponds to all the candidates,
		and the ground-truth item poses the first.
		Example: ground-truth items: [1, 2], 2 negative items for each instance: [[3,4], [5,6]]
				 predictions like: [[1,3,4], [2,5,6]]
		"""
		dataset.model.eval()
		predictions = list()
		dl = DataLoader(dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers,
						collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)
		for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict'):
			if hasattr(dataset.model,'inference'):
				prediction = dataset.model.inference(utils.batch_to_gpu(batch, dataset.model.device))['prediction']
			else:
				prediction = dataset.model(utils.batch_to_gpu(batch, dataset.model.device))['prediction']	
			predictions.extend(prediction.cpu().data.numpy())
		predictions = np.array(predictions)

		if dataset.model.test_all:
			print("have testall")
			rows, cols = list(), list()
			for i, u in enumerate(dataset.data['user_id']):
				clicked_items = list(dataset.corpus.train_clicked_set[u] | dataset.corpus.residual_clicked_set[u])
				idx = list(np.ones_like(clicked_items) * i)
				rows.extend(idx)
				cols.extend(clicked_items)
			predictions[rows, cols] = -np.inf
		return predictions

	def print_res(self, dataset: BaseModel.Dataset) -> str:
		"""
		Construct the final result string before/after training
		:return: test result string
		"""
		result_dict = self.evaluate(dataset, self.topk, self.metrics)
		res_str = '(' + utils.format_metric(result_dict) + ')'
		return res_str
