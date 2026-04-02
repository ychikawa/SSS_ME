import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, \
    IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform
from timm.data.transforms import str_to_interp_mode

from data.imagenet import ImageNetDataset


def build_dataset(is_train, args):
    model = args.model
    if model.startswith('sam') or model.startswith('augreg'):
        transform = build_transform_V3(is_train)
    else:
        transform = build_transform_V2(is_train, args)
    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = ImageNetDataset(root, transform=transform,
                                sampling_ratio=(args.sampling_ratio if is_train else args.sampling_ratio_test))
    nb_classes = 1000
    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    scale = getattr(args, 'scale', None)
    imagenet_default_mean_and_std = getattr(args, 'imagenet_default_mean_and_std', True)
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            scale=scale,
            mean=IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    crop = args.input_size < 320
    test_interpolation = str_to_interp_mode(getattr(args, 'test_interpolation', 'bicubic'))
    if resize_im:
        if crop:
            size = int((256 / 224) * args.input_size)
            t.append(transforms.Resize(size, interpolation=test_interpolation))
            t.append(transforms.CenterCrop(args.input_size))
        else:
            print(args.input_size)
            t.append(transforms.Resize((args.input_size,args.input_size), interpolation=test_interpolation))

    t.append(transforms.ToTensor())
    if imagenet_default_mean_and_std:
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    else:
        t.append(transforms.Normalize(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD))
    return transforms.Compose(t)


def build_transform_V2(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

from PIL import Image

def build_transform_V3(is_train):
    input_size = 224
    crop_pct = 0.9
    interpolation = Image.BICUBIC
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    resize_size = int(input_size / crop_pct)  # ≈ 248

    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, interpolation=interpolation),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    return transform