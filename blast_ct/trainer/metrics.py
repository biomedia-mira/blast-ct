from abc import ABC
import numpy as np
import torch


def to_np_cpu(tensor):
    return tensor.to('cpu').detach().numpy()


def report_scalar(name, scalar):
    return '{0}: {1:.4f}\t'.format(name, scalar).ljust(20)


def report_mean_and_std(name, array):
    return '{}: {:.4f} ± {:.4f}\t'.format(name, np.mean(array), np.std(array))


class Metric(ABC):
    def __init__(self, initial_value):
        self.running_value = initial_value
        self.value = None

    def increment(self, model_state):
        raise NotImplementedError

    def save_and_reset(self):
        raise NotImplementedError

    def report(self):
        raise NotImplementedError

    def log_to_tensorboard(self, epoch, writer, tag):
        raise NotImplementedError


class Loss(Metric):
    def __init__(self, device):
        Metric.__init__(self, torch.zeros(1, device=device))

    def increment(self, model_state):
        self.running_value += model_state['loss']

    def save_and_reset(self):
        self.value = float(self.running_value.to('cpu').detach())
        self.running_value[:] = 0

    def report(self):
        return report_scalar('loss', self.value) + '\n'

    def log_to_tensorboard(self, epoch, writer, tag):
        writer.add_scalar(tag, self.value, global_step=epoch)


class TrackLogitDistribution(Metric):
    # assumes shape (batch_size, num_maps, ...)
    def __init__(self, name, num_maps: int):
        self.name = name
        Metric.__init__(self, [[0, 0, 0] for _ in range(num_maps)])
        self.num_maps = num_maps
        self.value = [None for _ in range(num_maps)]

    def increment(self, model_state):
        tensor = model_state[self.name].detach()
        for i in range(self.num_maps):
            t = tensor[:, i]
            self.running_value[i][0] += t.numel()
            self.running_value[i][1] += torch.sum(t)
            self.running_value[i][2] += torch.sum(t * t)

    def save_and_reset(self):
        for i in range(self.num_maps):
            n = self.running_value[i][0]
            sum_ = self.running_value[i][1]
            sum_square = self.running_value[i][2]
            var = (sum_square - ((sum_ ** 2) / n)) / (n - 1)
            mean = sum_ / n
            self.value[i] = [mean.cpu().numpy(), torch.sqrt(var).cpu().numpy()]
        self.running_value = [[0, 0, 0] for _ in range(self.num_maps)]

    def report(self):
        messages = []
        for i, value in enumerate(self.value):
            messages.append(f'{self.name:s}_{i:d}: {value[0]:.4f} ± {value[1]:.4f}')
        l = max([len(m) for m in messages])
        message = ''
        for m in messages:
            message += m.ljust(l) + '\t'
        message += '\n'
        return message

    def log_to_tensorboard(self, epoch, writer, tag):
        for i, value in enumerate(self.value):
            writer.add_scalar(f'{tag:s}_{i:d}/mean', value[0], global_step=epoch)
            writer.add_scalar(f'{tag:s}_{i:d}/stddev', value[1], global_step=epoch)


class TrackTensor(Metric):

    def __init__(self, name):
        self.name = name
        Metric.__init__(self, [])

    def increment(self, model_state):
        self.value += list(model_state[self.name])

    def save_and_reset(self):
        self.value = to_np_cpu(torch.squeeze(torch.stack(tuple(self.running_value))))
        self.running_value = []

    def report(self):
        return report_mean_and_std(self.name, self.value)

    def log_to_tensorboard(self, epoch, writer, tag):
        pass


class RunningConfusionMatrix(Metric):

    def __init__(self, num_classes, device):
        super().__init__(torch.zeros((num_classes, num_classes), device=device))
        self.eye = torch.eye(num_classes, num_classes, device=device)

    def compute_confusion_matrix(self, labels, preds):
        return torch.einsum('nd,ne->de', self.eye[labels.flatten()], self.eye[preds.flatten()])

    def increment(self, model_state):
        self.running_value += self.compute_confusion_matrix(model_state['target'], model_state['pred'])

    def save_and_reset(self):
        self.value = to_np_cpu(self.running_value)
        self.running_value[:] = 0

    def report(self):
        return ''

    def log_to_tensorboard(self, epoch, writer, tag):
        pass


def calc_accuracy(cm):
    return np.repeat(np.sum(np.diag(cm)) / np.sum(cm), cm.shape[0])


def calc_precision(cm):
    return np.diag(cm) / np.sum(cm, axis=0)


def calc_recall(cm):
    return np.diag(cm) / np.sum(cm, axis=1)


def calc_f1_score(cm):
    return np.diag(cm) / np.sum(cm, axis=1)


class ClassificationMetrics(RunningConfusionMatrix):
    metric_fns = {'accuracy': calc_accuracy,
                  'precision': calc_precision,
                  'recall': calc_recall,
                  'f1_score': calc_f1_score}

    def __init__(self, device, class_names):

        self.class_names = class_names

        RunningConfusionMatrix.__init__(self, len(class_names), device)
        self.metrics = dict.fromkeys(self.metric_fns.keys())

    def save_and_reset(self):
        super().save_and_reset()
        cm = self.value
        for metric in self.metrics:
            self.metrics[metric] = self.metric_fns[metric](cm)

    def log_to_tensorboard(self, epoch, writer, tag):
        for i, class_name in enumerate(self.class_names):
            for metric, value in self.metrics.items():
                writer.add_scalar(class_name + '/' + str(metric), value[i], global_step=epoch)

    def report(self):
        message = ''
        for i, class_name in enumerate(self.class_names):
            message += class_name.upper().ljust(20) + ':\t'
            for metric, value in self.metrics.items():
                message += report_scalar(metric, value[i])
            message += '\n'
        return message


class SegmentationMetrics(ClassificationMetrics):
    def __init__(self, device, class_names):
        class_names[0] = 'Foreground'
        super().__init__(device, class_names)

    @staticmethod
    def merge_cm_classes(cm):
        new_cm = np.zeros(shape=(2, 2))
        fi = list(range(1, cm.shape[0]))
        new_cm[1, 1] = np.sum(cm[fi, fi])
        new_cm[0, 0] = cm[0, 0]
        new_cm[1, 0] = np.sum(cm[fi, 0])
        new_cm[0, 1] = np.sum(cm[0, fi])
        return new_cm

    def save_and_reset(self):
        super().save_and_reset()
        cm = self.value
        f_cm = self.merge_cm_classes(cm)
        for metric in self.metrics:
            self.metrics[metric][0] = self.metric_fns[metric](f_cm)[1]
