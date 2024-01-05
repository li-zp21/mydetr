"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
from tqdm import tqdm

import torch

import utils.misc as utils
from datasets.eval import Evaluator
from utils.plot_utils import plot_imgs


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict.items()}
        loss_dict_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict.items() if k in weight_dict}
        losses_scaled = sum(loss_dict_scaled.values())

        loss_value = losses_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_scaled, **loss_dict_unscaled)
        metric_logger.update(class_error=loss_dict['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'
    
    evaluator = Evaluator(base_ds)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        loss_dict_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict.items() if k in weight_dict}
        loss_dict_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict.items()}
        metric_logger.update(loss=sum(loss_dict_scaled.values()),
                             **loss_dict_scaled,
                             **loss_dict_unscaled)
        metric_logger.update(class_error=loss_dict['class_error'])
        
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors(outputs, orig_target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if evaluator is not None:
            evaluator.update(res)

    print("Averaged stats:", metric_logger)

    evaluator.synchronize_between_processes()
    # accumulate predictions from all images
    if evaluator is not None:
        evaluator.accumulate()
        evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if evaluator is not None:
        stats['coco_eval_bbox'] = evaluator.coco_eval.stats.tolist()
    return stats, evaluator


@torch.no_grad()
def plot_images(model, postprocessors, data_loader, device, output_dir):
    model.eval()

    for samples, targets in tqdm(data_loader):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors(outputs, orig_target_sizes)
        
        image_dir = 'data/images'
        plot_imgs(targets, results, image_dir, output_dir)
        