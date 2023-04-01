import argparse
import os
import time

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DistributedSampler, RandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import get_model

from rpcdataloader import RPCDataloader, RPCDataset


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data-path")
    argparser.add_argument("--model", default="resnet50")
    argparser.add_argument("--workers", type=str, nargs="+")
    argparser.add_argument("--batch-size", default=2, type=int)
    argparser.add_argument("--lr", default=0.1, type=float)
    argparser.add_argument("--momentum", default=0.9, type=float)
    argparser.add_argument("--weight-decay", default=1e-4, type=float)
    argparser.add_argument("--epochs", default=100, type=int)
    argparser.add_argument("--amp", action="store_true")
    args = argparser.parse_args()

    # Distributed
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:  # torchrun launch
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    elif int(os.environ.get("SLURM_NPROCS", 1)) > 1:  # srun launch
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NPROCS"])
    else:  # single gpu & process launch
        rank = 0
        local_rank = 0
        world_size = 0

    if world_size > 0:
        torch.distributed.init_process_group(
            backend="nccl", world_size=world_size, rank=rank
        )

        # split workers between GPUs (optional but recommended)
        if len(args.workers) > 0:
            args.workers = args.workers[rank::world_size]

    print(args)

    # Device
    device = torch.device("cuda", index=local_rank)

    # Preprocessing
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandAugment(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Datasets
    train_dataset = RPCDataset(
        args.workers,
        ImageFolder,
        root=args.data_path + "/train",
        transform=train_transform,
    )

    # Data loading
    if torch.distributed.is_initialized():
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = RandomSampler(train_dataset)

    train_loader = RPCDataloader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        pin_memory=True,
    )

    # Model
    model = get_model(args.model, num_classes=1000)
    model.to(device)
    if torch.distributed.is_initialized():
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
        )

    # Optimization
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss().to(device)

    # Training
    for epoch in range(args.epochs):
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)

        for it, (images, targets) in enumerate(train_loader):
            t0 = time.monotonic()

            optimizer.zero_grad(set_to_none=True)

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                predictions = model(images)
                loss = loss_fn(predictions, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (it + 1) % 20 == 0 and rank == 0:
                t1 = time.monotonic()
                print(
                    f"[epoch {epoch:<3d}"
                    f"  it {it:-5d}/{len(train_loader)}]"
                    f"  loss: {loss.item():2.3f}"
                    f"  time: {t1 - t0:.1f}"
                )

        scheduler.step()


if __name__ == "__main__":
    main()
