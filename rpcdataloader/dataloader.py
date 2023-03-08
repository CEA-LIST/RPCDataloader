# (C) Copyright 2022-2023 CEA LIST. All Rights Reserved.
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
from typing import Any, Callable, List

from torch.utils.data import (
    BatchSampler,
    Dataset,
    RandomSampler,
    SequentialSampler,
    default_collate,
)

from .rpc import rpc_async


def _get_item(dataset, item):
    return dataset[item]


def _get_batch(dataset, items, collate_fn=None):
    values = [dataset[i] for i in items]
    return values if collate_fn is None else collate_fn(values)


class RPCDataset(Dataset):
    """Handle to instanciate and manage datasets on remote workers.

    :param workers:
        a list of workers with the format `host:port`
    :param dataset:
        dataset class or equivalent callable that returns a dataset instance
    :param args:
        positional arguments for :attr:`dataset`
    :param kwargs:
        keyword arguments for :attr:`dataset`

    .. note::
        In a distributed setup, you should probably split the workers between
        the trainers (ie: :code:`workers = workers[rank::world_size]`).
    """

    def __init__(
        self, workers: List[str], dataset: Callable[[Any], Dataset], *args, **kwargs
    ):
        futures = [rpc_async(w, dataset, args, kwargs, rref=True) for w in workers]
        self.workers = workers
        self.rrefs = [f.wait() for f in futures]

    def __len__(self):
        return rpc_async(self.workers[0], len, [self.rrefs[0]]).wait()


class RPCDataloader:
    """A dataloader using remote rpc-based workers.

    :param dataset:
        A remote dataset
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
    :param prefetch_factor: Number of samples loaded in advance by each worker.
        ``2`` means there will be a total of 2 * num_workers samples
        prefetched across all workers. (default: ``2``)

    Notable differences with pytorch dataloader:

    - :attr:`timeout` is the timeout on individual network operations.
    - :attr:`worker_init_fn` and :attr:`generator` are not supported.
    - Random seeds are not supported because workers may execute requests
      out of order anyway, thus breaking reproducibility.
    """

    def __init__(
        self,
        dataset: RPCDataset,
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
        # Samplers
        if sampler is None:
            if shuffle:
                sampler = RandomSampler(dataset)

            else:
                sampler = SequentialSampler(dataset)

        if batch_size is not None and batch_sampler is None:
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        # Remaining attributes
        self.dataset = dataset
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
        remotes = zip(self.dataset.workers, self.dataset.rrefs)
        remotes_it = itertools.cycle(remotes)

        if self.batch_sampler is None:
            for (worker, rref), i in zip(remotes_it, self.sampler):
                yield worker, _get_item, (rref, i)

        else:
            for (worker, rref), i in zip(remotes_it, self.batch_sampler):
                yield worker, _get_batch, (rref, i, self.collate_fn)

    def __iter__(self):
        task_it = iter(self._iter_tasks())

        # RPC to create dataset
        queue = []

        try:
            # preload jobs
            for _ in range(self.prefetch_factor * len(self.dataset.workers)):
                try:
                    task = next(task_it)
                except StopIteration:
                    break
                else:
                    queue.append(rpc_async(*task, timeout=self.timeout))

            while len(queue) > 0:
                result = queue.pop(0).wait()

                # queue another job
                try:
                    task = next(task_it)
                except StopIteration:
                    pass
                else:
                    queue.append(rpc_async(*task, timeout=self.timeout))

                # return value
                yield result

        finally:
            for f in queue:
                try:
                    f.wait()
                except BaseException:
                    pass
