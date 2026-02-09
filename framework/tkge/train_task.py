import torch
import tqdm

import time
import os
import argparse

from typing import Dict, List
from collections import defaultdict

from tkge.task.task import Task
from tkge.data.dataset import DatasetProcessor, StampDataset, SpanEndDataset
from tkge.train.sampling import NegativeSampler
from tkge.train.regularization import Regularizer, InplaceRegularizer
from tkge.train.optim import get_optimizer, get_scheduler
from tkge.common.config import Config
from tkge.models.model import BaseModel
from tkge.models.pipeline_model import TransSimpleModel
from tkge.models.loss import Loss
from tkge.eval.metrics import Evaluation

import numpy as np


class TrainTask(Task):
    @staticmethod
    def parse_arguments(parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Train a model"""
        subparser = parser.add_parser("train", description=description, help="train a model.")

        subparser.add_argument(
            "-c",
            "--config",
            type=str,
            help="specify configuration file path"
        )

        # subparser.add_argument(
        #     "--resume",
        #     action="store_true",
        #     default=False,
        #     help="resume training from checkpoint in config file"
        # )

        subparser.add_argument(
            "--overrides",
            action="store_true",
            default=False,
            help="override the hyper-parameter stored in checkpoint with the configuration file"
        )

        return subparser

    def __init__(self, config: Config):
        super().__init__(config)

        self.dataset: DatasetProcessor = self.config.get("dataset.name")
        self.train_loader: torch.utils.data.DataLoader = None
        self.valid_loader: torch.utils.data.DataLoader = None
        # self.test_loader = None
        self.sampler: NegativeSampler = None
        self.model: BaseModel = None
        self.loss: Loss = None
        self.optimizer: torch.optim.optimizer.Optimizer = None
        self.lr_scheduler = None
        self.evaluation: Evaluation = None

        self.task_form = self.config.get("task.task_form")
        self.train_bs = self.config.get("train.batch_size")
        self.valid_bs = self.config.get("train.valid.batch_size")
        self.train_sub_bs = self.config.get("train.subbatch_size") if self.config.get(
            "train.subbatch_size") else self.train_bs
        self.valid_sub_bs = self.config.get("train.valid.subbatch_size") if self.config.get(
            "train.valid.subbatch_size") else self.valid_bs

        # [optional(id), optional(float)]
        if self.config.get("dataset.temporal.index"):
            self.datatype = ['timestamp_id']
        elif self.config.get("dataset.temporal.float"):
            self.datatype = ['timestamp_float']
        if self.config.get("dataset.temporal.index") and self.config.get("dataset.temporal.float"):
            print("index \ float error")
            return

        self.device = self.config.get("task.device")

        self._prepare()

    def _prepare(self):
        self.config.log(f"Preparing datasets {self.dataset} in folder {self.config.get('dataset.folder')}...")
        self.dataset = DatasetProcessor.create(config=self.config)
        self.dataset.info()

        self.config.log(f"Loading data")
        if self.task_form == 'stamp' or self.task_form == 'original':
            self.train_loader = torch.utils.data.DataLoader(
                StampDataset(self.dataset.get("train")[0], self.datatype),
                shuffle=True,
                batch_size=self.train_bs,
                num_workers=self.config.get("train.loader.num_workers"),
                pin_memory=self.config.get("train.loader.pin_memory"),
                drop_last=self.config.get("train.loader.drop_last"),
                timeout=self.config.get("train.loader.timeout")
            )

            self.valid_loader = torch.utils.data.DataLoader(
                StampDataset(self.dataset.get("valid")[0], self.datatype),
                shuffle=False,
                batch_size=self.valid_bs,
                num_workers=self.config.get("train.loader.num_workers"),
                pin_memory=self.config.get("train.loader.pin_memory"),
                drop_last=self.config.get("train.loader.drop_last"),
                timeout=self.config.get("train.loader.timeout")
            )

            self.test_loader = torch.utils.data.DataLoader(
                StampDataset(self.dataset.get("test")[0], self.datatype),
                shuffle=False,
                batch_size=self.valid_bs,
                num_workers=self.config.get("train.loader.num_workers"),
                pin_memory=self.config.get("train.loader.pin_memory"),
            )
        
        
        elif self.task_form == 'span_end':
            self.train_loader = torch.utils.data.DataLoader(
                SpanEndDataset(self.dataset.get("train")[1], self.datatype),
                shuffle=True,
                batch_size=self.train_bs,
                num_workers=self.config.get("train.loader.num_workers"),
                pin_memory=self.config.get("train.loader.pin_memory"),
                drop_last=self.config.get("train.loader.drop_last"),
                timeout=self.config.get("train.loader.timeout")
            )

            self.valid_loader = torch.utils.data.DataLoader(
                SpanEndDataset(self.dataset.get("valid")[1], self.datatype+["timestamp_id"]),
                shuffle=False,
                batch_size=self.valid_bs,
                num_workers=self.config.get("train.loader.num_workers"),
                pin_memory=self.config.get("train.loader.pin_memory"),
                drop_last=self.config.get("train.loader.drop_last"),
                timeout=self.config.get("train.loader.timeout")
            )

            self.test_loader = torch.utils.data.DataLoader(
                SpanEndDataset(self.dataset.get("test")[1], self.datatype+["timestamp_id"]),
                shuffle=False,
                batch_size=self.valid_bs,
                num_workers=self.config.get("train.loader.num_workers"),
                pin_memory=self.config.get("train.loader.pin_memory"),
            )

        self.config.log(f"Initializing negative sampling")
        self.sampler = NegativeSampler.create(config=self.config, dataset=self.dataset)

        self.config.log(f"Creating model {self.config.get('model.type')}")
        self.model = BaseModel.create(config=self.config, dataset=self.dataset)
        self.model.to(self.device)

        self.config.log(f"Initializing loss function")
        self.loss = Loss.create(config=self.config)

        self.config.log(f"Initializing optimizer")
        optimizer_type = self.config.get("train.optimizer.type")
        optimizer_args = self.config.get("train.optimizer.args")
        self.optimizer = get_optimizer(self.model.parameters(), optimizer_type, optimizer_args)

        self.config.log(f"Initializing lr scheduler")
        if self.config.get("train.lr_scheduler"):
            scheduler_type = self.config.get("train.lr_scheduler.type")
            scheduler_args = self.config.get("train.lr_scheduler.args")
            self.lr_scheduler = get_scheduler(self.optimizer, scheduler_type, scheduler_args)

        self.config.log(f"Initializing regularizer")
        self.regularizer = dict()
        self.inplace_regularizer = dict()

        if self.config.get("train.regularizer"):
            for name in self.config.get("train.regularizer"):
                self.regularizer[name] = Regularizer.create(self.config, name)

        if self.config.get("train.inplace_regularizer"):
            for name in self.config.get("train.inplace_regularizer"):
                self.inplace_regularizer[name] = InplaceRegularizer.create(self.config, name)

        self.config.log(f"Initializing evaluation")
        self.evaluation = Evaluation(config=self.config, dataset=self.dataset)

        # validity checks and warnings
        self.subbatch_adaptive = self.config.get("train.subbatch_adaptive")

        if self.train_sub_bs >= self.train_bs or self.train_sub_bs < 1:
            # TODO(max) improve logging with different hierarchies/labels, i.e. merge with branch advannced_log_and_ckpt_management
            self.config.log(f"Specified train.sub_batch_size={self.train_sub_bs} is greater or equal to "
                            f"train.batch_size={self.train_bs} or smaller than 1, so use no sub batches. "
                            f"Device(s) may run out of memory.", level="warning")
            self.train_sub_bs = self.train_bs

        if self.valid_sub_bs >= self.valid_bs or self.valid_sub_bs < 1:
            # TODO(max) improve logging with different hierarchies/labels, i.e. merge with branch advannced_log_and_ckpt_management
            self.config.log(f"Specified train.valid.sub_batch_size={self.valid_sub_bs} is greater or equal to "
                            f"train.valid.batch_size={self.valid_bs} or smaller than 1, so use no sub batches. "
                            f"Device(s) may run out of memory.", level="warning")
            self.valid_sub_bs = self.valid_bs

    def main(self):
        '''
        self.config.log(f"Test")
        self.config.log("load best model for test")
        self.config.checkpoint_folder = '/home/zhangjs/experiments/TKGbenchmark/results/de-simple/YAGO/ex000004/ckpt'
        self.ckpt = torch.load(os.path.join(self.config.checkpoint_folder, 'best.ckpt')) 
        self.model = BaseModel.create(config=self.config, dataset=self.dataset)
        self.model.load_state_dict(self.ckpt['state_dict'], strict=True)
        self.model.cuda()

        if self.task_form == 'original':
            metrics = self.eval('test')

            self.config.log(f"Metrics(head prediction) in iteration {epoch} : {metrics['head'].items()}")
            self.config.log(f"Metrics(tail prediction) in iteration {epoch} : {metrics['tail'].items()}")
            self.config.log(f"Metrics(both prediction) in iteration {epoch} : {metrics['avg'].items()} ")
        
        if self.task_form == 'stamp':
            self.evaluation = Evaluation(config=self.config, dataset=self.dataset)
            metrics = self.eval_ro('test')
            self.config.log(f"Metrics(ro prediction): {metrics['avg'].items()} ")
        
        if self.task_form == 'span_end':
            metrics = self.eval_span_end('test')
            self.config.log(f"Metrics(span_end prediction): {metrics['avg'].items()} ")
        return
        '''

        self.config.log("BEGIN TRAINING")

        save_freq = self.config.get("train.checkpoint.every")
        eval_freq = self.config.get("train.valid.every")

        if self.task_form in ['original','stamp']:
            self.best_metric = 0
        else:
            self.best_metric = 1000000
        self.best_epoch = 0

        epoch = 0
        metrics = self.eval_ro()
        self.config.log(f"Metrics(ro prediction) in iteration {epoch} : {metrics['avg'].items()} ")
        for epoch in range(1, self.config.get("train.max_epochs") + 1):
            self.model.train()

            total_epoch_loss = 0.0
            train_size = self.dataset.train_size

            start_time = time.time()
            batch_id = 0
            # processing batches
            for pos_batch in tqdm.tqdm(self.train_loader):
                done = False
                batch_id += 1
                while not done:
                    try:
                        self.optimizer.zero_grad()

                        batch_loss = 0.

                        # may be smaller than the specified batch size in last iteration
                        bs = pos_batch.size(0)

                        # processing subbatches
                        for start in range(0, bs, self.train_sub_bs):
                            stop = min(start + self.train_sub_bs, bs)
                            pos_subbatch = pos_batch[start:stop]
                            subbatch_loss, subbatch_factors = self._subbatch_forward(pos_subbatch)
                            subbatch_loss.backward()
                            batch_loss += subbatch_loss.cpu().item()

                        # batch_loss.backward()
                        self.optimizer.step()

                        total_epoch_loss += batch_loss

                        if subbatch_factors:
                            for name, tensors in subbatch_factors.items():
                                if name not in self.inplace_regularizer:
                                    continue

                                if not isinstance(tensors, (tuple, list)):
                                    tensors = [tensors]

                                self.inplace_regularizer[name](tensors)

                        done = True

                    except RuntimeError as e:
                        if ("CUDA out of memory" not in str(e) or not self.subbatch_adaptive):
                            raise e

                        self.train_sub_bs //= 2
                        if self.train_sub_bs > 0:
                            self.config.log(f"CUDA out of memory. Subbatch size reduced to {self.train_sub_bs}.",
                                            level="warning")
                        else:
                            self.config.log(f"CUDA out of memory. Subbatch size cannot be further reduces.",
                                            level="error")
                            raise e

            stop_time = time.time()
            avg_loss = total_epoch_loss / train_size

            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(avg_loss)
                else:
                    self.lr_scheduler.step()

            self.config.log(f"Loss in iteration {epoch} : {avg_loss} consuming {stop_time - start_time}s")

            if epoch % save_freq == 0:
                self.config.log(f"Save the model checkpoint to {self.config.checkpoint_folder} as file epoch_{epoch}.ckpt")
                self.save_ckpt(f"epoch_{epoch}", epoch=epoch)

            if epoch % eval_freq == 0:
                if self.task_form == 'original':
                    metrics = self.eval()

                    self.config.log(f"Metrics(head prediction) in iteration {epoch} : {metrics['head'].items()}")
                    self.config.log(f"Metrics(tail prediction) in iteration {epoch} : {metrics['tail'].items()}")
                    self.config.log(f"Metrics(both prediction) in iteration {epoch} : {metrics['avg'].items()} ")
                
                if self.task_form == 'stamp':
                    self.evaluation = Evaluation(config=self.config, dataset=self.dataset)
                    metrics = self.eval_ro()
                    self.config.log(f"Metrics(ro prediction) in iteration {epoch} : {metrics['avg'].items()} ")
                
                if self.task_form == 'span_end':
                    metrics = self.eval_span_end()
                    self.config.log(f"Metrics(span_end prediction) in iteration {epoch} : {metrics['avg'].items()} ")

                if self.task_form in ['original']:
                    if metrics['avg']['mean_reciprocal_ranking'] > self.best_metric:
                        self.best_metric = metrics['avg']['mean_reciprocal_ranking']
                        self.best_epoch = epoch

                        self.config.log(f"Save the model checkpoint to {self.config.checkpoint_folder} as file best.ckpt")
                        self.save_ckpt('best', epoch=epoch)
                    else:
                        if self.config.get('train.valid.early_stopping.early_stop'):
                            patience = self.config.get('train.valid.early_stopping.patience')
                            if epoch - self.best_epoch >= patience:
                                self.config.log(
                                    f"Early stopping: valid metrics not improved in {patience} epoch and training stopped at epoch {epoch}")
                                break
                
                if self.task_form in ['stamp']:
                    if metrics['avg']['recall@500'] > self.best_metric:
                        self.best_metric = metrics['avg']['recall@500']
                        self.best_epoch = epoch

                        self.config.log(f"Save the model checkpoint to {self.config.checkpoint_folder} as file best.ckpt")
                        self.save_ckpt('best', epoch=epoch)
                    else:
                        if self.config.get('train.valid.early_stopping.early_stop'):
                            patience = self.config.get('train.valid.early_stopping.patience')
                            if epoch - self.best_epoch >= patience:
                                self.config.log(
                                    f"Early stopping: valid metrics not improved in {patience} epoch and training stopped at epoch {epoch}")
                                break
                
                if self.task_form in ['span_end']:
                    if metrics['avg']['MAE'] < self.best_metric:
                        self.best_metric = metrics['avg']['MAE']
                        self.best_epoch = epoch

                        self.config.log(f"Save the model checkpoint to {self.config.checkpoint_folder} as file best.ckpt")
                        self.save_ckpt('best', epoch=epoch)
                    else:
                        if self.config.get('train.valid.early_stopping.early_stop'):
                            patience = self.config.get('train.valid.early_stopping.patience')
                            if epoch - self.best_epoch >= patience:
                                self.config.log(
                                    f"Early stopping: valid metrics not improved in {patience} epoch and training stopped at epoch {epoch}")
                                break

                if self.config.get('train.valid.early_stopping.early_stop'):
                    thresh_epoch = self.config.get('train.valid.early_stopping.epochs')
                    if epoch > thresh_epoch and self.best_metric < self.config.get(
                            'train.valid.early_stopping.metric_thresh'):
                        self.config.log(
                            f"Early stopping: within {thresh_epoch} metrics doesn't exceed threshold and training stopped at epoch {epoch}")
                        break
            self.save_ckpt('latest', epoch=epoch)

        self.config.log(f"TRAINING FINISHED: Best model achieved at epoch {self.best_epoch}")
        self.config.log("load best model for test")
        self.ckpt = torch.load(os.path.join(self.config.checkpoint_folder, 'best.ckpt')) 
        self.model = BaseModel.create(config=self.config, dataset=self.dataset)
        self.model.load_state_dict(self.ckpt['state_dict'], strict=True)
        self.model.cuda()

        if self.task_form == 'original':
            metrics = self.eval('test')

            self.config.log(f"Metrics(head prediction) in iteration {epoch} : {metrics['head'].items()}")
            self.config.log(f"Metrics(tail prediction) in iteration {epoch} : {metrics['tail'].items()}")
            self.config.log(f"Metrics(both prediction) in iteration {epoch} : {metrics['avg'].items()} ")
        
        if self.task_form == 'stamp':
            self.evaluation = Evaluation(config=self.config, dataset=self.dataset)
            metrics = self.eval_ro('test')
            self.config.log(f"Metrics(ro prediction) in iteration {epoch} : {metrics['avg'].items()} ")
        
        if self.task_form == 'span_end':
            metrics = self.eval_span_end('test')
            self.config.log(f"Metrics(span_end prediction) in iteration {epoch} : {metrics['avg'].items()} ")

    def _subbatch_forward(self, pos_subbatch):
        sample_target = self.config.get("negative_sampling.target")
        samples, labels = self.sampler.sample(pos_subbatch, sample_target)

        samples = samples.to(self.device)
        labels = labels.to(self.device)

        scores, factors = self.model.fit(samples)

        factors = {} if factors==None else factors
        self.config.assert_true(scores.size(0) == labels.size(
            0), f"Score's size {scores.shape} should match label's size {labels.shape}")
        loss = self.loss(scores, labels)

        self.config.assert_true(not (factors and set(factors.keys()) - (set(self.regularizer) | set(
            self.inplace_regularizer))),
                                f"Regularizer name defined in model {set(factors.keys())} should correspond to that in config file")

        if factors:
            for name, tensors in factors.items():
                if name not in self.regularizer:
                    continue

                if not isinstance(tensors, (tuple, list)):
                    tensors = [tensors]

                reg_loss = self.regularizer[name](tensors)
                loss += reg_loss

        return loss, factors

    def _subbatch_forward_predict(self, query_subbatch):
        bs = query_subbatch.size(0)
        queries_head = query_subbatch.clone() # batch_size*[s,r,o,y,m,d] or batch_size*[s,r,o,t]
        queries_tail = query_subbatch.clone()

        queries_head[:, 0] = float('nan') # batch_size*[?,r,o,y,m,d]
        queries_tail[:, 2] = float('nan') # batch_size*[s,r,?,y,m,d]

        batch_scores_head = self.model.predict(queries_head)
        self.config.assert_true(list(batch_scores_head.shape) == [bs,
                                                                  self.dataset.num_entities()],
                                f"Scores {batch_scores_head.shape} should be in shape [{bs}, {self.dataset.num_entities()}]")

        batch_scores_tail = self.model.predict(queries_tail)
        self.config.assert_true(list(batch_scores_tail.shape) == [bs,
                                                                  self.dataset.num_entities()],
                                f"Scores {batch_scores_head.shape} should be in shape [{bs}, {self.dataset.num_entities()}]")

        subbatch_metrics = dict()

        subbatch_metrics['head'] = self.evaluation.eval(query_subbatch, batch_scores_head, miss='s')
        subbatch_metrics['tail'] = self.evaluation.eval(query_subbatch, batch_scores_tail, miss='o')
        subbatch_metrics['size'] = bs

        del query_subbatch, batch_scores_head, batch_scores_tail
        torch.cuda.empty_cache()

        return subbatch_metrics
    
    def _subbatch_forward_predict_ro(self, query_subbatch):
        bs = query_subbatch.size(0)
        queries_ro = query_subbatch.clone() # batch_size*[s,r,o,y,m,d] or batch_size*[s,r,o,t]

        queries_ro[:, 2] = float('nan') # batch_size*[s,?,?,y,m,d]

        target_scores_ro = self.model(query_subbatch)[0]
        batch_scores_ro = self.model.predict_ro(queries_ro)
        self.config.assert_true(list(batch_scores_ro.shape) == [bs, self.dataset.num_relations()*self.dataset.num_entities()],
                                f"Scores {batch_scores_ro.shape} should be in shape [{bs}, {self.dataset.num_relations()*self.dataset.num_entities()}]")

        subbatch_metrics = dict()
        subbatch_metrics = self.evaluation.eval_ro(query_subbatch, batch_scores_ro, target_scores_ro, miss='ro')
        
        torch.cuda.empty_cache()

        return subbatch_metrics

    def _subbatch_forward_predict_span_end(self, query_subbatch, target_subbatch, filter_subbatch):
        bs = query_subbatch.size(0)
        queries_span_end = query_subbatch.clone() # batch_size*[s,r,o,y,m,d] or batch_size*[s,r,o,t]

        queries_span_end[:, 3:] = float('nan') # batch_size*[s,r,o,y,m,d]

        target_scores_span_end = self.model(query_subbatch)[0]
        batch_scores_span_end = self.model.predict_span_end(queries_span_end)
        self.config.assert_true(list(batch_scores_span_end.shape) == [bs, self.dataset.num_timestamps()],
                                f"Scores {batch_scores_span_end.shape} should be in shape [{bs}, {self.dataset.num_timestamps()}]")

        subbatch_metrics = dict()

        subbatch_metrics['avg'] = self.evaluation.eval_span_end(query_subbatch, batch_scores_span_end, target_scores_span_end, target_subbatch, filter_subbatch, miss='end_time')
        subbatch_metrics['size'] = bs

        del query_subbatch, batch_scores_span_end
        torch.cuda.empty_cache()

        return subbatch_metrics

    def eval_ro(self, eval_type = 'valid'):
        eval_data_loader = None
        if eval_type == 'valid':
            eval_data_loader = self.valid_loader
        elif eval_type == 'test':
            eval_data_loader = self.test_loader
        
        with torch.no_grad():
            self.model.eval()

            counter = 0

            metrics = {}
            metrics['avg'] = {'recall@50':[],'recall@100':[],'recall@500':[],'ndcg@50':[],'ndcg@100':[],'ndcg@500':[]}

            for batch in tqdm.tqdm(eval_data_loader):
                done = False
                if eval_type == 'valid' and len(metrics['avg']['recall@50']) >= 500:
                    break
                if eval_type == 'test' and len(metrics['avg']['recall@50']) >= 5000:
                    break
                while not done:
                    try:
                        bs = batch.size(0)
                        dim = batch.size(1)

                        batch = batch.to(self.device)

                        counter += bs

                        for start in range(0, bs, self.valid_sub_bs):
                            stop = min(start + self.valid_sub_bs, bs)
                            query_subbatch = batch[start:stop]
                            subbatch_metrics = self._subbatch_forward_predict_ro(query_subbatch)
                            #print(subbatch_metrics)

                            for key in subbatch_metrics.keys():
                                metrics['avg'][key] += subbatch_metrics[key]

                        done = True

                    except RuntimeError as e:
                        if ("CUDA out of memory" not in str(e) or not self.subbatch_adaptive):
                            raise e

                        self.valid_sub_bs //= 2
                        if self.valid_sub_bs > 0:
                            self.config.log(
                                f"CUDA out of memory. Subbatch size for validation reduced to {self.valid_sub_bs}.",
                                level="warning")
                        else:
                            self.config.log(
                                f"CUDA out of memory. Subbatch size for validation cannot be further reduces.",
                                level="error")
                            raise e

            for pos in ['avg']:
                for key in metrics[pos].keys():
                    if len(metrics[pos][key]) != 0:
                        metrics[pos][key] = float(sum(metrics[pos][key])) / float(len(metrics[pos][key]))

            #avg = {k: (metrics['head'][k] + metrics['tail'][k]) / 2 for k in metrics['head'].keys()}
            #metrics.update({'avg': avg})

            return metrics


    '''
    def eval_ro(self, eval_type = 'valid'):
        eval_data_loader = None
        if eval_type == 'valid':
            eval_data_loader = self.valid_loader
        elif eval_type == 'test':
            eval_data_loader = self.test_loader

        with torch.no_grad():
            self.model.eval()

            counter = 0

            metrics = dict()
            metrics['avg'] = defaultdict(float)
            metrics['size'] = 0

            for batch in tqdm.tqdm(eval_data_loader):
                done = False
                if eval_type == 'valid' and metrics['size'] >= 500:
                    break
                if eval_type == 'test' and metrics['size'] >= 5000:
                    break
                while not done:
                    try:
                        bs = batch.size(0)
                        dim = batch.size(1)

                        batch_metrics = dict()
                        batch_metrics['avg'] = defaultdict(float)
                        batch_metrics['size'] = 0

                        batch = batch.to(self.device)

                        counter += bs

                        for start in range(0, bs, self.valid_sub_bs):
                            stop = min(start + self.valid_sub_bs, bs)
                            query_subbatch = batch[start:stop]
                            subbatch_metrics = self._subbatch_forward_predict_ro(query_subbatch)

                            for pos in ['avg']:
                                for key in subbatch_metrics[pos].keys():
                                    batch_metrics[pos][key] += subbatch_metrics[pos][key] * subbatch_metrics['size']
                            batch_metrics['size'] += subbatch_metrics['size']

                        done = True

                        for pos in ['avg']:
                            for key in batch_metrics[pos].keys():
                                batch_metrics[pos][key] /= batch_metrics['size']

                    except RuntimeError as e:
                        if ("CUDA out of memory" not in str(e) or not self.subbatch_adaptive):
                            raise e

                        self.valid_sub_bs //= 2
                        if self.valid_sub_bs > 0:
                            self.config.log(
                                f"CUDA out of memory. Subbatch size for validation reduced to {self.valid_sub_bs}.",
                                level="warning")
                        else:
                            self.config.log(
                                f"CUDA out of memory. Subbatch size for validation cannot be further reduces.",
                                level="error")
                            raise e

                for pos in ['avg']:
                    for key in batch_metrics[pos].keys():
                        metrics[pos][key] += batch_metrics[pos][key] * batch_metrics['size']
                metrics['size'] += batch_metrics['size']

            # del batch
            # torch.cuda.empty_cache()

            for pos in ['avg']:
                for key in metrics[pos].keys():
                    metrics[pos][key] /= metrics['size']

            #avg = {k: (metrics['head'][k] + metrics['tail'][k]) / 2 for k in metrics['head'].keys()}
            #metrics.update({'avg': avg})

            return metrics
        '''

    def eval(self, eval_type = 'valid'):
        eval_data_loader = None
        if eval_type == 'valid':
            eval_data_loader = self.valid_loader
        elif eval_type == 'test':
            eval_data_loader = self.test_loader

        with torch.no_grad():
            self.model.eval()

            counter = 0

            metrics = dict()
            metrics['head'] = defaultdict(float)
            metrics['tail'] = defaultdict(float)
            metrics['size'] = 0

            for batch in tqdm.tqdm(eval_data_loader):
                done = False

                while not done:
                    try:
                        bs = batch.size(0)
                        dim = batch.size(1)

                        batch_metrics = dict()
                        batch_metrics['head'] = defaultdict(float)
                        batch_metrics['tail'] = defaultdict(float)
                        batch_metrics['size'] = 0

                        batch = batch.to(self.device)

                        counter += bs

                        for start in range(0, bs, self.valid_sub_bs):
                            stop = min(start + self.valid_sub_bs, bs)
                            query_subbatch = batch[start:stop]
                            subbatch_metrics = self._subbatch_forward_predict(query_subbatch)

                            for pos in ['head', 'tail']:
                                for key in subbatch_metrics[pos].keys():
                                    batch_metrics[pos][key] += subbatch_metrics[pos][key] * subbatch_metrics['size']
                            batch_metrics['size'] += subbatch_metrics['size']

                        done = True

                        for pos in ['head', 'tail']:
                            for key in batch_metrics[pos].keys():
                                batch_metrics[pos][key] /= batch_metrics['size']

                    except RuntimeError as e:
                        if ("CUDA out of memory" not in str(e) or not self.subbatch_adaptive):
                            raise e

                        self.valid_sub_bs //= 2
                        if self.valid_sub_bs > 0:
                            self.config.log(
                                f"CUDA out of memory. Subbatch size for validation reduced to {self.valid_sub_bs}.",
                                level="warning")
                        else:
                            self.config.log(
                                f"CUDA out of memory. Subbatch size for validation cannot be further reduces.",
                                level="error")
                            raise e

                for pos in ['head', 'tail']:
                    for key in batch_metrics[pos].keys():
                        metrics[pos][key] += batch_metrics[pos][key] * batch_metrics['size']
                metrics['size'] += batch_metrics['size']

            # del batch
            # torch.cuda.empty_cache()

            for pos in ['head', 'tail']:
                for key in metrics[pos].keys():
                    metrics[pos][key] /= metrics['size']

            avg = {k: (metrics['head'][k] + metrics['tail'][k]) / 2 for k in metrics['head'].keys()}

            metrics.update({'avg': avg})

            return metrics

    def save_ckpt(self, ckpt_name, epoch):
        filename = f"{ckpt_name}.ckpt"
        #filename = f"{ckpt_name}.pth"

        checkpoint = {
            'last_epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'best_metrics': self.best_metric,
            'best_epoch': self.best_epoch
        }
        #torch.save(save_model.cpu(), os.path.join(self.config.checkpoint_folder,filename))
        torch.save(checkpoint,
                   os.path.join(self.config.checkpoint_folder,
                                filename))  # os.path.join(model, dataset, folder, filename))

    def eval_span_end(self, eval_type = 'valid'):
        eval_data_loader = None
        if eval_type == 'valid':
            eval_data_loader = self.valid_loader
        elif eval_type == 'test':
            eval_data_loader = self.test_loader

        with torch.no_grad():
            self.model.eval()

            counter = 0

            metrics = dict()
            metrics['avg'] = defaultdict(float)
            metrics['size'] = 0

            for batch in tqdm.tqdm(eval_data_loader):
                done = False
                while not done:
                    try:
                        bs = batch.size(0)
                        dim = batch.size(1)

                        batch_metrics = dict()
                        batch_metrics['avg'] = defaultdict(float)
                        batch_metrics['size'] = 0

                        batch = batch.to(self.device)

                        counter += bs

                        for start in range(0, bs, self.valid_sub_bs):
                            stop = min(start + self.valid_sub_bs, bs)
                            query_subbatch = batch[start:stop]
                            fact_subbatch = query_subbatch[:, :-2]
                            target_subbatch = query_subbatch[:, -2]
                            filter_subbatch = query_subbatch[:, -1]
                            subbatch_metrics = self._subbatch_forward_predict_span_end(fact_subbatch, target_subbatch, filter_subbatch)

                            for pos in ['avg']:
                                for key in subbatch_metrics[pos].keys():
                                    batch_metrics[pos][key] += subbatch_metrics[pos][key] * subbatch_metrics['size']
                            batch_metrics['size'] += subbatch_metrics['size']

                        done = True

                        for pos in ['avg']:
                            for key in batch_metrics[pos].keys():
                                batch_metrics[pos][key] /= batch_metrics['size']

                    except RuntimeError as e:
                        if ("CUDA out of memory" not in str(e) or not self.subbatch_adaptive):
                            raise e

                        self.valid_sub_bs //= 2
                        if self.valid_sub_bs > 0:
                            self.config.log(
                                f"CUDA out of memory. Subbatch size for validation reduced to {self.valid_sub_bs}.",
                                level="warning")
                        else:
                            self.config.log(
                                f"CUDA out of memory. Subbatch size for validation cannot be further reduces.",
                                level="error")
                            raise e

                for pos in ['avg']:
                    for key in batch_metrics[pos].keys():
                        metrics[pos][key] += batch_metrics[pos][key] * batch_metrics['size']
                metrics['size'] += batch_metrics['size']

            # del batch
            # torch.cuda.empty_cache()

            for pos in ['avg']:
                for key in metrics[pos].keys():
                    metrics[pos][key] /= metrics['size']

            #avg = {k: (metrics['head'][k] + metrics['tail'][k]) / 2 for k in metrics['head'].keys()}
            #metrics.update({'avg': avg})

            return metrics

    def load_ckpt(self, ckpt_path):
        raise NotImplementedError
# CUDA_VISIBLE_DEVICES=1 python tkge.py train --config config/example_de_simple.yaml