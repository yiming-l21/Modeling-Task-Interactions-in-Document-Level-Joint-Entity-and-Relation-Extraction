import torch
import logging
from os.path import join, exists
from functools import cached_property
from abc import ABC, abstractmethod
from datetime import datetime
from transformers import AdamW, get_scheduler
from torch.optim.lr_scheduler import LambdaLR
import util
from collections import Iterable
from util_io import read_json, write_json, read_pickle, write_pickle, get_config

logger = logging.getLogger()


class BaseRunner(ABC):
    def __init__(self, config_name, gpu_id=None, seed=None, save_log=True):
        self.name = config_name
        self.name_suffix = datetime.now().strftime('%b%d_%H-%M-%S')
        self.gpu_id = gpu_id
        self.seed = seed
        self.config = get_config(config_name)

        if save_log:
            log_path = join(self.config['log_dir'], 'log_' + self.name_suffix + '.txt')
            logger.addHandler(logging.FileHandler(log_path, 'a'))
            logger.info(f'Log file path: {log_path}')

        if seed is not None:
            util.set_seed(seed)

        self.device = torch.device('cpu' if gpu_id is None else f'cuda:{gpu_id}')

        self.dataset_name = self.config['dataset_name']
        self.partition = {
            'train': self.config['train_partition'],
            'dev': self.config['dev_partition'],
            'test': self.config['test_partition']
        }

    @cached_property
    @abstractmethod
    def data(self):
        pass

    @abstractmethod
    def initialize_model(self, init_suffix=None):
        pass

    @classmethod
    def get_optimizer(cls, model, bert_lr, task_lr, bert_wd, task_wd, eps):
        no_decay = ['bias', 'LayerNorm.weight']
        bert_param, task_param = model.get_params(named=True)
        grouped_param = [
            {
                'params': [p for n, p in bert_param if
                           not any(nd in n for nd in no_decay) and p.requires_grad],
                'lr': bert_lr,
                'weight_decay': bert_wd
            }, {
                'params': [p for n, p in bert_param if
                           any(nd in n for nd in no_decay) and p.requires_grad],
                'lr': bert_lr,
                'weight_decay': 0.0
            }, {
                'params': [p for n, p in task_param if
                           not any(nd in n for nd in no_decay) and p.requires_grad],
                'lr': task_lr,
                'weight_decay': task_wd
            }, {
                'params': [p for n, p in task_param if
                           any(nd in n for nd in no_decay) and p.requires_grad],
                'lr': task_lr,
                'weight_decay': 0.0
            }
        ]
        optimizer = AdamW(grouped_param, eps=eps)
        return optimizer

    @classmethod
    def get_scheduler(cls, optimizer, total_update_steps, warmup_ratio=None, scheduler_name='linear'):
        warmup_ratio = warmup_ratio or 0
        if scheduler_name == 'custom':
            cooldown_start = int(total_update_steps * warmup_ratio)

            def lr_lambda(current_step: int):
                return 1 if current_step < cooldown_start else (1 - warmup_ratio)

            return LambdaLR(optimizer, lr_lambda, -1)
        else:
            warmup_steps = int(total_update_steps * (warmup_ratio or 0))
            scheduler = get_scheduler(scheduler_name, optimizer, warmup_steps, total_update_steps)
            return scheduler

    @classmethod
    def log_metrics(cls, metrics, tb_writer=None, step=0, skip_iterable=True):
        for metric, score in metrics.items():
            if isinstance(score, Iterable) and skip_iterable:
                if metric.endswith("re_prf"):
                    logger.info(f'dev_p: {score[0]:.2f}')
                    logger.info(f'dev_r: {score[1]:.2f}')
                if metric.endswith("re_ign_prf"):
                    logger.info(f'dev_p_ign: {score[0]:.2f}')
                    logger.info(f'dev_r_ign: {score[1]:.2f}')
                continue
            logger.info(f'{metric}: {score:.2f}')
            if tb_writer:
                tb_writer.add_scalar(metric, score, step)

    def save_model_checkpoint(self, model, step,type):
        path_ckpt = join(f"./models/{type}/", f'best_model.bin')
        torch.save(model.state_dict(), path_ckpt)

    def load_model_checkpoint(self, model, init_suffix, init_config=None):
        if init_suffix!=None:
            path_ckpt = init_suffix
        else:
            path_ckpt = join("./models/coref/", f'best_model_spanbert.bin')
        print(f'Loading model from {path_ckpt}')
        model.load_state_dict(torch.load(path_ckpt, map_location=torch.device('cpu')), strict=False)

    def save_results(self, dataset_name, partition, suffix, results, ext):
        assert ext in ('bin', 'json')
        write_fct = write_pickle if ext == 'bin' else write_json
        save_path = self.data.get_results_path(dataset_name, partition, suffix, ext=ext)
        write_fct(save_path, results)

    def load_results(self, dataset_name, partition, suffix, ext):
        assert ext in ('bin', 'json')
        read_fct = read_pickle if ext == 'bin' else read_json
        save_path = self.data.get_results_path(dataset_name, partition, suffix, ext=ext)
        if exists(save_path):
            results = read_fct(save_path)
            return results
        else:
            return None
