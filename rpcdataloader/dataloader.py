# (C) Copyright 2022 CEA LIST. All Rights Reserved.
# Contributor(s): Nicolas Granger <nicolas.granger@cea.fr>
#
# This software is governed by the CeCILL-C license under French law and
# abiding by the rules of distribution of free software.  You can  use,
# modify and/ or redistribute the software under the terms of the CeCILL-C
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL-C license and that you accept its terms.

import itertools
import socket
import uuid
import weakref
from typing import Optional, Sequence

import torch.distributed
import torch.futures
from torch.utils.data import (
    BatchSampler,
    DistributedSampler,
    RandomSampler,
    Sampler,
    SequentialSampler,
    default_collate,
    Dataset,
)

from .rpc import rpc_async


_datasets = {}


def _create_dataset(uid, dataset_t, args, kwargs):
    _datasets[uid] = dataset_t(*args, **kwargs)


def _delete_dataset(uid):
    _datasets.pop(uid)


def _len_dataset(uid):
    return len(_datasets[uid])


def _get_item(uid, item):
    _datasets[uid][item]


def _get_batch(uid, items, collate_fn=None):
    d = _datasets[uid]
    values = [d[i] for i in items]
    return values if collate_fn is None else collate_fn(values)


class _SizedPlaceholder(Dataset):
    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n


class RPCDataloader:
    """A dataloader using remote rpc-based workers.

    :param workers:
        list of rpc workers formatted as: "address:port"
    :param dataset:
        dataset constructor
    :param args:
        positional arguments for dataset constructor
    :param kwargs:
        keyword arguments for dataset constructor
    :param batch_size:
        how many samples per batch to load.
    :param shuffle:
        set to ``True`` to have the data reshuffled at every epoch.
    :param sampler:
        defines the strategy to draw samples from the dataset. Can be any
        ``Iterable`` with ``__len__`` implemented. If specified,
        :attr:`shuffle` must not be specified.
    :param batch_sampler:
        like :attr:`sampler`, but returns a batch of indices at a time.
        Mutually exclusive with :attr:`batch_size`, :attr:`shuffle`,
        :attr:`sampler`, and :attr:`drop_last`.
    :param num_workers:
        how many subprocesses to use for data loading. ``0`` means that the
        data will be loaded in the main process.
    :param collate_fn:
        merges a list of samples to form a mini-batch of Tensor(s). Used
        when using batched loading from a map-style dataset.
    :param pin_memory:
        If ``True``, the data loader will copy Tensors into CUDA pinned
        memory before returning them. If your data elements are a custom
        type, or your :attr:`collate_fn` returns a batch that is a custom
        type, see the example below.
    :param drop_last: set to ``True`` to drop the last incomplete batch, if
        the dataset size is not divisible by the batch size. If ``False``
        and the size of dataset is not divisible by the batch size, then the
        last batch will be smaller.

    Differences with pytorch dataloader:

    * The default sampler automatically uses DistributedSampler if distributed mode is initialized.
    * Only mappable dataset are supported (Dataset, not IterableDataset)
    * timeout is the timeout on individual network operations
    * :attr:`worker_init_fn` and :attr:`generator` are not supported.

    .. note::
        In a distributed setup, you should probably split the workers between
        the trainers (ie: :code:`worker=workers[rank::world_size]`).
    """
    def __init__(
        self,
        workers: list[str],
        dataset,
        args=[],
        kwargs={},
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler[int]] = None,
        batch_sampler: Optional[Sampler[Sequence[int]]] = None,
        collate_fn=None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 120,
        *,
        prefetch_factor: int = 2,
    ):
        # Identify remotes
        self.remotes = []
        for r in workers:
            h, p = r.split(":")
            h = socket.gethostbyname(h)
            p = int(p)
            self.remotes.append((h, p, uuid.uuid1()))

        # Instanciate dataset on remote workers
        futures = []
        for h, p, u in self.remotes:
            futures.append(
                rpc_async(
                    h,
                    p,
                    _create_dataset,
                    args=[u, dataset, args, kwargs],
                    pin_memory=pin_memory,
                )
            )
            weakref.finalize(self, rpc_async, h, p, _delete_dataset, [u])

        torch.futures.collect_all(futures).wait()

        # Check dataset size
        h0, p0, u0 = self.remotes[0]
        size = rpc_async(h0, p0, _len_dataset, args=[u0]).wait()

        # Samplers
        if sampler is None:
            placeholder = _SizedPlaceholder(size)

            if torch.distributed.is_initialized():
                sampler = DistributedSampler(placeholder, shuffle=shuffle)
            elif shuffle:
                sampler = RandomSampler(placeholder)
            else:
                sampler = SequentialSampler(placeholder)

        if batch_size is not None and batch_sampler is None:
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        # Remaining attributes
        self.batch_size = batch_size
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.collate_fn = default_collate if collate_fn is None else collate_fn
        self.pin_memory = pin_memory
        self.timeout = timeout
        self.prefetch_factor = prefetch_factor

    def __len__(self):
        if self.batch_sampler is None:
            return len(self.sampler)  # type: ignore
        else:
            return len(self.batch_sampler)  # type: ignore

    def _iter_tasks(self):
        if self.batch_sampler is None:
            for (h, p, u), i in zip(itertools.cycle(self.remotes), self.sampler):
                yield h, p, (u, i)

        else:
            for (h, p, u), i in zip(itertools.cycle(self.remotes), self.batch_sampler):
                yield h, p, (u, i, self.collate_fn)

    def __iter__(self):
        get_fn = _get_item if self.batch_sampler is None else _get_batch
        task_it = iter(self._iter_tasks())

        # RPC to create dataset
        queue = []

        try:
            # preload jobs
            for _ in range(self.prefetch_factor * len(self.remotes)):
                try:
                    h, p, get_args = next(task_it)
                except StopIteration:
                    break
                else:
                    queue.append(
                        rpc_async(h, p, get_fn, get_args, timeout=self.timeout)
                    )

            while len(queue) > 0:
                result = queue.pop(0).wait()

                # queue another job
                try:
                    h, p, get_args = next(task_it)
                except StopIteration:
                    break
                else:
                    queue.append(
                        rpc_async(h, p, get_fn, get_args, timeout=self.timeout)
                    )

                # return value
                yield result

        finally:
            torch.futures.collect_all(queue).wait()
