import time
import os
import re
import torch
from abc import ABC
import numpy as np
from blast_ct.trainer.metrics import to_np_cpu, Metric
from torch.utils.tensorboard import SummaryWriter


class Hook(ABC):
    def __init__(self):
        self.model_trainer = None

    def attach_hook(self, model_trainer):
        self.model_trainer = model_trainer

    def before_epoch(self):
        pass

    def after_batch(self):
        pass

    def after_epoch(self):
        pass


class Evaluator(object):
    def __init__(self, log_dir, metrics):
        for metric in metrics.values():
            assert isinstance(metric, Metric)
        self.metrics = metrics
        self.writer = SummaryWriter(log_dir=log_dir)

    def increment_state(self, model_state):
        for metric in self.metrics.values():
            metric.increment(model_state)

    def calculate_and_reset_metrics(self):
        for metric in self.metrics.values():
            metric.save_and_reset()

    def report(self):
        message = ''
        for name, metric in self.metrics.items():
            message += metric.report()
        return message

    def log_to_tensorboard(self, epoch):
        for name, metric in self.metrics.items():
            metric.log_to_tensorboard(epoch, self.writer, name)


class TrainingEvaluator(Hook, Evaluator):
    def __init__(self, log_dir, metrics):
        Hook.__init__(self)
        Evaluator.__init__(self, log_dir, metrics)
        self.time = 0

    def before_epoch(self):
        self.time = time.time()

    def after_batch(self):
        self.increment_state(self.model_trainer.current_state)

    def after_epoch(self):
        self.calculate_and_reset_metrics()
        epoch = self.model_trainer.current_state['epoch']
        num_epochs = self.model_trainer.current_state['num_epochs']
        elapsed_time = time.time() - self.time
        message = f'Training epoch {epoch:d}/{num_epochs - 1:d} completed in {elapsed_time:.0f}s\n' + self.report()
        self.model_trainer.logger.info(message)
        self.log_to_tensorboard(epoch)


class ValidationEvaluator(Hook, Evaluator):
    def __init__(self, log_dir, metrics, dataloader, eval_every, saver=None):
        Hook.__init__(self)
        Evaluator.__init__(self, log_dir, metrics)
        self.dataloader = dataloader
        self.eval_every = eval_every
        self.saver = saver

    def perform_evaluation(self):
        epoch = self.model_trainer.current_state['epoch']
        for state in self.model_trainer.step(epoch, self.dataloader, is_training=False):
            self.increment_state(state)
            if self.saver is not None:
                message = self.saver(state)
                if message is not None:
                    self.model_trainer.logger.info(message)

    def before_epoch(self):
        if self.saver is not None:
            self.saver.reset()

    def after_epoch(self):
        epoch = self.model_trainer.current_state['epoch']
        num_epochs = self.model_trainer.current_state['num_epochs']
        is_last_epoch = epoch == num_epochs - 1
        if not (epoch % self.eval_every == 0 or is_last_epoch) or epoch == 0:
            return

        start_time = time.time()
        self.perform_evaluation()
        self.calculate_and_reset_metrics()
        elapsed_time = time.time() - start_time
        message = f'Validation epoch {epoch:d}/{num_epochs - 1:d} completed in {elapsed_time:.0f}s\n' + self.report()
        self.model_trainer.logger.info(message)
        self.log_to_tensorboard(epoch)


class NaNLoss(Hook):
    def __init__(self):
        super().__init__()

    def after_epoch(self):
        if np.isnan(to_np_cpu(self.model_trainer.current_state['loss'])):
            self.model_trainer.stop_signal = True
            self.model_trainer.logger.error('Found NaN loss during training, stopping gracefully.')


class ModelSaverHook(Hook):
    def __init__(self, save_every=100, keep_model_every=100):
        super().__init__()
        self.save_every = save_every
        self.keep_model_every = keep_model_every

    @staticmethod
    def save_model_to_disk(model, model_path):
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        torch.save(model, model_path)

    def after_epoch(self):
        epoch = self.model_trainer.current_state['epoch']
        num_epochs = self.model_trainer.current_state['num_epochs']
        is_last_epoch = epoch == num_epochs - 1
        if not (epoch % self.save_every == 0 or is_last_epoch) or epoch == 0:
            return

        saved_model_dir = os.path.join(self.model_trainer.job_dir, 'saved_models')

        # save current model
        model_path = os.path.join(saved_model_dir, 'model_' + str(epoch) + '.torch_model')
        self.save_model_to_disk(self.model_trainer.model.state_dict(), model_path)

        # delete old models
        for file in os.listdir(saved_model_dir):
            matches = re.findall('(?<=model_)\d+', file)
            for match in matches:
                previous_epoch = int(match)
                if (previous_epoch % self.keep_model_every != 0 or previous_epoch == 0) and previous_epoch != epoch:
                    os.remove(os.path.join(saved_model_dir, file))

        # if last epoch save last model
        if is_last_epoch:
            model_path = os.path.join(saved_model_dir, 'model_last.torch_model')
            self.save_model_to_disk(self.model_trainer.model.state_dict(), model_path)
