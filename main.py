# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import time
import torch
import json
import pickle
from pathlib import Path
import numpy as np

import os
import sys

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer_v2
from timm.utils import NativeScaler, get_state_dict
from torch.cuda.amp import GradScaler
from engine import train_one_epoch, train_one_epoch_with_ratio, evaluate, calibrate
import utils
from data.augment import new_data_aug_generator
from data.datasets import build_dataset
from data.samplers import RASampler
from utils import benchmark, flops, str2bool, str2list
import models.deit
import models.lvvit
import models.mae
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns

from timm.utils import accuracy

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--batch-size',         default=1024, type=int)
    parser.add_argument('--epochs', default=30, type=int)


    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT', help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME', help='Use AutoAugment policy. "v0" or "original". (default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')

    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
    
    parser.add_argument('--ThreeAugment', action='store_true') #3augment
    
    parser.add_argument('--src', action='store_true') #simple random crop
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT', help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False, help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8, help='mixup alpha, mixup enabled if > 0. (default: 0.0)')
    parser.add_argument('--cutmix', type=float, default=1.0, help='cutmix alpha, cutmix enabled if > 0. (default: 0.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None, help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0, help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5, help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch', help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data_path', type=str, required=True, help='dataset path')

    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--resume-file', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist_eval', default=False, type=str2bool, help='Enabling distributed evaluation')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")

    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_mem', default=False, type=str2bool, help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--debug',              default=None,       type=str)
    parser.add_argument('--save_every',         default=None,       type=int, help='save model every epochs')

    parser.add_argument('--train-interpolation', type=str, default='bicubic', help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--sampling_ratio', default=1., type=float, help='fraction of samples to keep in the training set of imagenet')
    parser.add_argument('--sampling_ratio_test', default=1., type=float, help='fraction of samples to keep in the training set of imagenet')
    parser.add_argument('--pretrained', default=True, type=str2bool, help='Start with pretrained version of specified network (if avail)')

    # algorithm related parameters
    parser.add_argument('--lr_times',        default=1.0,  type=float)
    parser.add_argument('--min_lr_times',    default=1.0,  type=float)
    parser.add_argument('--task_type', default=[1, 0, 0], type=str2list, help='0 : student model'
                                                                              '1 : teacher model'
                                                                              '2 : consistency threshold')
    parser.add_argument('--task_weight', default=[1., 1., 1.], type=str2list, help='0 : student model loss weight'
                                                                                   '1 : teacher model loss weight'
                                                                                   '2 : consistency loss weight')

    parser.add_argument('--algo',            default='default')
    parser.add_argument('--r',               default=0,     type=int)
    parser.add_argument('--load_schedule',   action='store_true', help='eval by exiting compression rate in compression_rate.json')
    parser.add_argument('--tgt_flops',       default=1.0,  type=float)
    parser.add_argument('--modular', action='store_true', help='modular training')
    parser.add_argument('--benchmark', action='store_true', help='measure performance')

    return parser

def r2schedule(r, num_protected=1, num_tokens=197, num_layers=12):
    ret = []
    cur_tokens = num_tokens
    for i in range(num_layers):
        r_ = min(r, (cur_tokens - num_protected) // 2)
        cur_tokens -= r_
        ret.append(cur_tokens)
    return ret

def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_train = RASampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)

    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    if args.load_schedule:
        # https://github.com/OpenGVLab/DiffRate/blob/main/compression_rate.json
        with open('compression_rate.json', 'r') as f:
            compression_rate = json.load(f)
            if not str(args.tgt_flops) in compression_rate[args.model]:
                raise ValueError(f"compression_rate.json does not contaion {args.model} with {args.tgt_flops}G flops")
            prune_schedule = eval(compression_rate[args.model][str(args.tgt_flops)]['prune_kept_num'])
            merge_schedule = eval(compression_rate[args.model][str(args.tgt_flops)]['merge_kept_num'])
    else:
        if args.model=="vit_large_patch16_mae":
            num_layers=24
        elif args.model=="lvvit_s":
            num_layers=16
        else:
            num_layers=12
        prune_schedule = r2schedule(args.r, num_layers=num_layers)
        merge_schedule = r2schedule(args.r, num_layers=num_layers)

    print(f"Creating model: {args.model}")

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size,

        # Options
        task_type      = args.task_type,
        algo           = args.algo,
        prune_schedule = prune_schedule,
        merge_schedule = merge_schedule,
    )

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size = int(num_patches ** 0.5)
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if utils.is_main_process() and args.benchmark:
        model.eval()
        args.n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        args.fv_flops = flops(model, (1, 3, args.input_size, args.input_size), device=device, round_num=3)
        print(f"Params : {args.n_parameters}M,  Flops : {args.fv_flops}G")
    else:
        args.fv_flops = -1
    
    opt_keys = ['opt', 'weight_decay', 'momentum', 'eps', 'betas']
    opt_args = {k: v for k, v in vars(args).items() if k in opt_keys}

    if not args.eval:
        if args.modular:
            param_groups = [
                {'params': model_without_ddp.arch_parameters(), 'lr': 1e-2}
            ]
        else:
            param_groups = [
                {'params': model_without_ddp.parameters(), 'lr': 3e-5},
                {'params': model_without_ddp.arch_parameters(), 'lr': 1e-3}
            ]
        optimizer = create_optimizer_v2(
            param_groups,
            **opt_args
        )
        loss_scaler = NativeScaler()

        lr_scheduler, total_epochs = create_scheduler(args, optimizer)
        args.epochs = total_epochs

    if mixup_active:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if os.path.isfile(args.resume_file):
        print('>>>>>> resume from {}'.format(args.resume_file))
        if args.resume_file.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume_file, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume_file, map_location='cpu')

        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            lr_scheduler.step(args.start_epoch)
            args.start_epoch += 1


    if args.eval:
        log_stats = {"model": args.model, "algo": args.algo, "resume": args.resume_file,
                     "p_sched": prune_schedule, "m_sched": merge_schedule, "throughput": [], "flops": [], "acc1": []}

        log_stats["flops"].append(args.fv_flops)

        if args.benchmark:
            model = torch.compile(model, mode="reduce-overhead")
            throughput = benchmark(
                model,
                device=device,
                verbose=True,
                runs=200,
                batch_size=256,
                input_size=(3, args.input_size, args.input_size),
                use_fp16=True,
            )
            log_stats["throughput"].append(round(throughput))


        test_stats = evaluate(loader_val, model, device)
        log_stats["acc1"].append(round(test_stats["acc1"], 3))

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        folder_path = "./result"
        os.makedirs(folder_path, exist_ok=True)
        with open(os.path.join(folder_path, "result.txt"), 'a') as f:
            f.write(json.dumps(log_stats) + "\n")
        print(log_stats.values())

        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            loader_train.sampler.set_epoch(epoch)


        train_stats = train_one_epoch(
            model, criterion, loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            set_training_mode=args.train_mode,
            args = args
        )
        lr_scheduler.step(epoch)

        test_stats = evaluate(loader_val, model, device)

        if args.algo=="learnable_size":
            print(list(model_without_ddp.arch_parameters()))

        if test_stats["acc1"] > max_accuracy:
            max_accuracy = test_stats["acc1"]

        print(f"[Epoch {epoch}] Accuracy of the network on  test images: {test_stats['acc1']:.1f}%")
        print(f"[Epoch {epoch}] Max Accuracy : {max_accuracy:.1f}%")

        if args.output_dir and utils.is_main_process():
            savepoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }
            utils.save_on_master(savepoint, os.path.join(args.output_dir, 'checkpoint.pth'))
            if args.save_every is not None:
                if epoch % args.save_every == 0:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'args': args
                    }, os.path.join(args.output_dir, 'checkpoint_{}.pth'.format(epoch)))


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    args.min_lr = 1e-6
    if args.modular:
        args.cooldown_epochs = 0
    else:
        args.cooldown_epochs = 0
    args.warmup_epochs = 0

    # file overwrite check
    output_file = os.path.join(args.output_dir, "checkpoint.pth")
    if os.path.isfile(output_file):
        print(f"File '{output_file}' exists. Overwriting the existing file.")
    else:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # set resume path
    if args.resume:
        if os.path.isfile(args.resume_file):
            resume_file = args.resume_file
        elif not args.resume_file and os.path.isfile(output_file):
            resume_file = output_file
        else:
            print(f"Resume file does not exist. Exiting.")
            sys.exit(1)
    else:
        args.resume_file = ''

    # load args
    if args.resume and not args.eval:
        checkpoint = torch.load(resume_file, map_location='cpu')
        if checkpoint['epoch'] + 1 == args.epochs + args.cooldown_epochs: sys.exit("Break")
        args = checkpoint['args']
        args.resume_file = resume_file
        args.start_epoch = checkpoint['epoch']

    main(args)
