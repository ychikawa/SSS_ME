# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
from timm.utils import accuracy
from typing import Iterable, Optional
from timm.data import Mixup
import utils
import torch
import time
import math
import sys
import pickle

def train_one_epoch(model: torch.nn.Module,criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq  = int(len(data_loader) / 20)
    for idx, (samples, targets_ori) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if args.debug and idx == 10:
            break
        start = time.time()
        samples = samples.to(device, non_blocking=True)
        targets = targets_ori.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        loss, loss_student, loss_teacher, loss_consistency = torch.tensor([0.0], device=device), torch.tensor([0.0], device=device), torch.tensor([0.0], device=device), torch.tensor([0.0], device=device)
        with torch.amp.autocast('cuda'):
            outputs = model(samples)
            if args.task_type[0]:
                loss_student = criterion(outputs, targets) * args.task_weight[0]

            if args.task_type[1]:
                loss_teacher = criterion(outputs, targets) * args.task_weight[1]

            if args.task_type[2]:
                logits = torch.softmax(outputs["x_student"], dim=-1)
                confidence_score = logits[range(len(targets_ori)), targets_ori] + logits[range(len(targets_ori)), targets_ori.flip(0)] if args.mixup else logits[range(len(targets_ori)), targets_ori]
                confidence_pass = (confidence_score > args.task_type[2]).float()
                if confidence_pass.sum():
                    loss_consistency = (outputs["loss_consistency"] * confidence_pass).sum() / confidence_pass.sum() * args.task_weight[2]
                    metric_logger.update(confidence_score = confidence_score.mean().item())
                    metric_logger.update(confidence_pass  = confidence_pass.mean().item())
            loss = loss_student + loss_teacher + loss_consistency

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(time=time.time() - start)
        metric_logger.update(loss_student=loss_student.item())
        metric_logger.update(loss_teacher=loss_teacher.item())
        metric_logger.update(loss_cls_consis=loss_consistency.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_with_ratio(model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, set_training_mode=True,
                    args=None, ratio: float = 1.0):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq  = int(len(data_loader) / 20)
    total_steps = int(len(data_loader) * ratio)

    for idx, (samples, targets_ori) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if args.debug and idx == 10:
            break
        if idx >= total_steps:
            break

        start = time.time()
        samples = samples.to(device, non_blocking=True)
        targets = targets_ori.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        loss, loss_student, loss_teacher, loss_consistency = torch.tensor([0.0], device=device), torch.tensor([0.0], device=device), torch.tensor([0.0], device=device), torch.tensor([0.0], device=device)
        with torch.amp.autocast('cuda'):
            outputs = model(samples)
            if args.task_type[0]:
                loss_student = criterion(outputs["x_student"], targets) * args.task_weight[0]

            if args.task_type[1]:
                loss_teacher = criterion(outputs["x_teacher"], targets) * args.task_weight[1]

            if args.task_type[2]:
                logits = torch.softmax(outputs["x_student"], dim=-1)
                confidence_score = logits[range(len(targets_ori)), targets_ori] + logits[range(len(targets_ori)), targets_ori.flip(0)] if args.mixup else logits[range(len(targets_ori)), targets_ori]
                confidence_pass = (confidence_score > args.task_type[2]).float()
                if confidence_pass.sum():
                    loss_consistency = (outputs["loss_consistency"] * confidence_pass).sum() / confidence_pass.sum() * args.task_weight[2]
                    metric_logger.update(confidence_score = confidence_score.mean().item())
                    metric_logger.update(confidence_pass  = confidence_pass.mean().item())
            loss = loss_student + loss_teacher + loss_consistency

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(time=time.time() - start)
        metric_logger.update(loss_student=loss_student.item())
        metric_logger.update(loss_teacher=loss_teacher.item())
        metric_logger.update(loss_cls_consis=loss_consistency.item())

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(data_loader, model, device):

    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter = "  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # compute output
        with torch.amp.autocast('cuda'):
            x = model(samples)
        loss = criterion(x, targets)
        acc1, acc5 = accuracy(x, targets, topk=(1, 5))

        batch_size = samples.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def calibrate(data_loader, model, device, ratio):

    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter = "  ")
    header = 'Calibration:'

    # clamp ratio to [0, 1] and derive print frequency and total steps if possible
    ratio = max(0.0, min(1.0, float(ratio)))
    try:
        data_len = len(data_loader)
    except TypeError:
        data_len = None
    print_freq = 10 if data_len is None else max(1, int(data_len / 20))
    total_steps = None if data_len is None else int(data_len * ratio)

    # switch to evaluation mode
    model.train()
    used_samples = 0
    first_batch_size = None
    for idx, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if total_steps is not None and idx >= total_steps:
            break
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # compute output
        with torch.amp.autocast('cuda'):
            x = model(samples)
        loss = criterion(x, targets)
        acc1, acc5 = accuracy(x, targets, topk=(1, 5))

        batch_size = samples.shape[0]
        if first_batch_size is None:
            first_batch_size = batch_size
        used_samples += batch_size
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    if first_batch_size is not None and data_len is not None:
        approx_total = first_batch_size * data_len
        ratio_pct = (used_samples / approx_total) * 100.0 if approx_total > 0 else 0.0
        print('Used training samples: {} (~{:.2f}% of approx dataset size {})'.format(used_samples, ratio_pct, approx_total))
    else:
        print('Used training samples: {}'.format(used_samples))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
