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

import torch
from torch.utils.data import (
    BatchSampler,
    RandomSampler,
    SequentialSampler,
    default_collate,
)
from .rpc import rpc_async


_datasets = {}


def _create_dataset(uid, dataset_t, args, kwargs):
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    _datasets[uid] = dataset_t(*args, **kwargs)


def _delete_dataset(uid):
    _datasets.pop(uid)


def _len_dataset(uid):
    return len(_datasets[uid])


def _get_item(uid, item):
    return _datasets[uid][item]


def _get_batch(uid, items, collate_fn=None):
    d = _datasets[uid]
    values = [d[i] for i in items]
    return values if collate_fn is None else collate_fn(values)


class _SizedPlaceholder:
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

    - Only mappable dataset are supported (Dataset, not IterableDataset)
    - If distributed mode is initialized, and :attr:`sampler` is not specified,
      it defaults to :class:`torch.utils.data.distributed.DistributedSampler`.
      Don't forget to call :code:`set_epoch` on :attr:`RPCDataloader.sampler`
      at every epoch.
    - :attr:`timeout` is the timeout on individual network operations
    - :attr:`worker_init_fn` and :attr:`generator` are not supported.
    - Random seeds are not set automatically because workers are persistent
      and not tied to a specific dataloader. See :func:`set_random_seeds` for
      a way to set the seeds.

    .. note::
        In a distributed setup, you should probably split the workers between
        the trainers (ie: :code:`worker=workers[rank::world_size]`).
    """

    def __init__(
        self,
        workers,
        dataset,
        args=None,
        kwargs=None,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=120,
        *,
        prefetch_factor: int = 2,
    ):
        # Identify remotes
        self.remotes = []
        for w in workers:
            host, port = w.split(":")
            host = socket.gethostbyname(host)
            port = int(port)
            self.remotes.append((host, port, uuid.uuid1()))

        # Instanciate dataset on remote workers
        futures = []
        for host, port, dataset_uid in self.remotes:
            futures.append(
                rpc_async(
                    host,
                    port,
                    _create_dataset,
                    args=[dataset_uid, dataset, args, kwargs],
                    pin_memory=pin_memory,
                )
            )
            weakref.finalize(
                self, rpc_async, host, port, _delete_dataset, [dataset_uid]
            )

        for f in futures:
            f.wait()

        # Check dataset size
        host0, port0, dataset_uid0 = self.remotes[0]

        # Samplers
        if sampler is None:
            size = rpc_async(host0, port0, _len_dataset, args=[dataset_uid0]).wait()
            placeholder = _SizedPlaceholder(size)

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                sampler = torch.utils.data.distributed.DistributedSampler(
                    placeholder, shuffle=shuffle
                )

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
            for (host, port, dataset_uid), i in zip(
                itertools.cycle(self.remotes), self.sampler
            ):
                yield host, port, (dataset_uid, i)

        else:
            for (host, port, dataset_uid), i in zip(
                itertools.cycle(self.remotes), self.batch_sampler
            ):
                yield host, port, (dataset_uid, i, self.collate_fn)

    def __iter__(self):
        get_fn = _get_item if self.batch_sampler is None else _get_batch
        task_it = iter(self._iter_tasks())

        # RPC to create dataset
        queue = []

        try:
            # preload jobs
            for _ in range(self.prefetch_factor * len(self.remotes)):
                try:
                    host, port, get_args = next(task_it)
                except StopIteration:
                    break
                else:
                    queue.append(
                        rpc_async(host, port, get_fn, get_args, timeout=self.timeout)
                    )

            while len(queue) > 0:
                result = queue.pop(0).wait()

                # queue another job
                try:
                    host, port, get_args = next(task_it)
                except StopIteration:
                    pass
                else:
                    queue.append(
                        rpc_async(host, port, get_fn, get_args, timeout=self.timeout)
                    )

                # return value
                yield result

        finally:
            for f in queue:
                try:
                    f.wait()
                except BaseException:
                    pass
