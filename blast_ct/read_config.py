import torch.nn
from torch.utils.data.dataloader import DataLoader

from blast_ct import models as models
from blast_ct.nifti import augmention
from blast_ct.nifti import patch_samplers
from blast_ct.nifti import transformation
from blast_ct.nifti.datasets import PatchWiseNiftiDataset, FullImageToOverlappingPatchesNiftiDataset, worker_init_fn
from blast_ct.nifti.savers import NiftiPatchSaver
from blast_ct.trainer import losses as losses
from blast_ct.trainer.hooks import TrainingEvaluator, ValidationEvaluator, ModelSaverHook, NaNLoss
from blast_ct.trainer.metrics import Loss
from blast_ct.trainer.metrics import SegmentationMetrics


def get_augmentation(augmentation_dict):
    return [getattr(augmention, name)(**kwargs) for name, kwargs in augmentation_dict.items()]


def get_transformation(transformation_dict):
    return [getattr(transformation, name)(**kwargs) for name, kwargs in transformation_dict.items()]


def get_train_loader(config, model, train_csv_path, use_cuda):
    input_patch_size = tuple(config['training']['input_patch_size'])
    output_patch_size = model.get_output_size(input_patch_size)

    transformation = get_transformation(config['data']['transformation'])
    augmentation = get_augmentation(config['training']['augmentation'])
    patch_augmentation = get_augmentation(config['training']['patch_augmentation'])

    sampler_type = list(config['training']['sampler'].keys())[0]
    config['training']['sampler'][sampler_type].update({'augmentation': patch_augmentation})
    sampler_class = getattr(patch_samplers, sampler_type)
    sampler = sampler_class(input_patch_size, output_patch_size, **config['training']['sampler'][sampler_type])

    sampling_mask = config['data']['sampling_mask'] if 'sampling_mask' in config['data'] else None
    sample_weight = config['data']['sample_weight'] if 'sample_weight' in config['data'] else None
    resolution = config['data']['resolution'] if 'resolution' in config['data'] else None

    train_set = PatchWiseNiftiDataset(patch_sampler=sampler,
                                      images_per_epoch=config['training']['images_per_epoch'],
                                      patches_per_image=config['training']['patches_per_image'],
                                      data_csv_path=train_csv_path,
                                      channels=config['data']['channels'],
                                      target=config['data']['target'],
                                      sampling_mask=sampling_mask,
                                      sample_weight=sample_weight,
                                      transformation=transformation,
                                      augmentation=augmentation,
                                      max_cases_in_memory=config['training']['max_cases_in_memory'],
                                      resolution=resolution)

    train_loader = DataLoader(train_set,
                              batch_size=config['training']['batch_size'],
                              num_workers=config['training']['num_workers'],
                              worker_init_fn=worker_init_fn,
                              pin_memory=True if use_cuda else False)
    return train_loader


def get_valid_loader(config, model, test_csv_path, use_cuda):
    if config['valid'] is None:
        return None
    input_patch_size = tuple(config['valid']['input_patch_size'])
    output_patch_size = model.get_output_size(input_patch_size)

    transformation = get_transformation(config['data']['transformation'])

    sampling_mask = config['data']['sampling_mask'] if 'sampling_mask' in config['data'] else None
    sample_weight = config['data']['sample_weight'] if 'sample_weight' in config['data'] else None
    resolution = config['data']['resolution'] if 'resolution' in config['data'] else None

    sampler_type = list(config['training']['sampler'].keys())[0]
    sampler_class = getattr(patch_samplers, sampler_type)
    sampler = sampler_class(input_patch_size, output_patch_size, **config['training']['sampler'][sampler_type])

    # set sequential to True to reduce disk access,
    valid_set = PatchWiseNiftiDataset(patch_sampler=sampler,
                                      images_per_epoch=config['valid']['images_per_epoch'],
                                      patches_per_image=config['valid']['patches_per_image'],
                                      data_csv_path=test_csv_path,
                                      channels=config['data']['channels'],
                                      target=config['data']['target'],
                                      sampling_mask=sampling_mask,
                                      sample_weight=sample_weight,
                                      transformation=transformation,
                                      max_cases_in_memory=config['training']['max_cases_in_memory'],
                                      sequential=True,
                                      resolution=resolution)

    valid_loader = DataLoader(valid_set,
                              batch_size=config['valid']['batch_size'],
                              num_workers=config['valid']['num_workers'],
                              worker_init_fn=worker_init_fn,
                              pin_memory=True if use_cuda else False)

    return valid_loader


def get_test_loader(config, model, test_csv_path, use_cuda):
    if config['test'] is None:
        return None
    input_patch_size = tuple(config['test']['input_patch_size'])
    output_patch_size = model.get_output_size(input_patch_size)
    transformation = get_transformation(config['data']['transformation'])

    sampling_mask = config['data']['sampling_mask'] if 'sampling_mask' in config['data'] else None
    resolution = config['data']['resolution'] if 'resolution' in config['data'] else None

    test_set = FullImageToOverlappingPatchesNiftiDataset(image_patch_shape=input_patch_size,
                                                         target_patch_shape=output_patch_size,
                                                         data_csv_path=test_csv_path,
                                                         channels=config['data']['channels'],
                                                         target=config['data']['target'],
                                                         sampling_mask=sampling_mask,
                                                         transformation=transformation,
                                                         resolution=resolution)

    test_loader = DataLoader(test_set,
                             batch_size=config['test']['batch_size'],
                             shuffle=False,
                             num_workers=config['test']['num_workers'],
                             worker_init_fn=worker_init_fn,
                             pin_memory=True if use_cuda else False)
    return test_loader


def get_training_hooks(job_dir, config, device, valid_loader, test_loader):
    hooks = []

    def get_metrics():
        return {'loss': Loss(device),
                'metrics': SegmentationMetrics(device, config['data']['class_names'])}

    hooks.append(TrainingEvaluator(job_dir + '/train', get_metrics()))
    hooks.append(NaNLoss())
    hooks.append(ModelSaverHook(config['valid']['eval_every'], config['valid']['keep_model_every']))

    if valid_loader is not None:
        hooks.append(ValidationEvaluator(job_dir + '/val', get_metrics(), valid_loader, config['valid']['eval_every']))

    # test only done at the end
    if test_loader is not None:
        extra_output_names = config['test']['extra_output_names'] if 'extra_output_names' in config['test'] else None
        saver = NiftiPatchSaver(job_dir + '/test', test_loader, extra_output_names=extra_output_names)
        hooks.append(ValidationEvaluator(job_dir + '/test',
                                         get_metrics(),
                                         test_loader,
                                         config['test']['eval_every'],
                                         saver))
    return hooks


def get_model(config):
    model_type = list(config['model'].keys())[0]
    model_class = getattr(models, model_type)
    model = model_class(input_channels=config['data']['input_channels'],
                        num_classes=config['data']['num_classes'],
                        **config['model'][model_type])
    return model


def get_loss(config):
    loss_type = list(config['loss'].keys())[0]
    loss_class = getattr(losses, loss_type)
    loss = loss_class(**config['loss'][loss_type])
    return loss


def get_optimizer(config, model):
    optimizer_type = list(config['optimizer'].keys())[0]
    optimizer_class = getattr(torch.optim, optimizer_type)
    optimizer = optimizer_class(model.parameters(), **config['optimizer'][optimizer_type])
    scheduler_type = list(config['scheduler'].keys())[0]
    scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_type)
    scheduler = scheduler_class(optimizer, **config['scheduler'][scheduler_type])
    return scheduler
