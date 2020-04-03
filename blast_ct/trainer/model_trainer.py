import torch
import torch.nn.functional as F
import time
from .logger import get_logger


def predict_exclusive(logits):
    prob = F.softmax(logits, dim=1)
    _, pred = torch.max(logits, dim=1)
    return prob, pred


def predict_multi_target(logits):
    prob = torch.sigmoid(logits)
    pred = torch.round(prob)
    return prob, pred


def predict_regression(logits):
    return logits, logits


task_predict_fn_dict = {'segmentation': predict_exclusive,
                        'classification': predict_exclusive,
                        'multi_target_classification': predict_multi_target,
                        'regression': predict_regression}


class ModelTrainer(object):
    def __init__(self, job_dir, device, model, criterion, lr_scheduler, hooks, task):
        assert task in task_predict_fn_dict.keys()
        self.job_dir = job_dir
        self.device = device
        self.model = model
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.hooks = hooks
        self.predict_fn = task_predict_fn_dict[task]
        self.logger = get_logger(job_dir)
        self.stop_signal = False

    def step(self, epoch, dataloader, is_training=True):
        self.model.train() if is_training else self.model.eval()
        for inputs in dataloader:
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            self.lr_scheduler.optimizer.zero_grad()
            with torch.set_grad_enabled(is_training):
                logits, state = self.model(**inputs)
                state.update(inputs)
                loss = self.criterion(logits, **state)
                if is_training:
                    loss.backward()
                    self.lr_scheduler.optimizer.step()
            prob, pred = self.predict_fn(logits)
            state.update({'epoch': epoch, 'loss': loss, 'logits': logits, 'prob': prob, 'pred': pred})
            yield state

    def _run_epoch(self, epoch, dataloader):
        [hook.before_epoch() for hook in self.hooks]
        for state in self.step(epoch, dataloader):
            self.current_state.update(state)
            [hook.after_batch() for hook in self.hooks]
        [hook.after_epoch() for hook in self.hooks]
        self.lr_scheduler.step(epoch)
        return

    def __call__(self, dataloader, num_epochs):
        start_training = time.time()
        self.model.to(self.device)
        [hook.attach_hook(self) for hook in self.hooks]
        self.current_state = {'num_epochs': num_epochs}

        for epoch in range(num_epochs):
            if self.stop_signal:
                break
            self._run_epoch(epoch, dataloader)

        time_elapsed = time.time() - start_training
        self.logger.info(f'Training completed in {time_elapsed // 3600:.0f}h {time_elapsed % 3600 // 60:.0f}m '
                         f'{time_elapsed % 3600 % 60:.0f}s')
        return False if self.stop_signal else True
