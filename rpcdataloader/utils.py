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

from __future__ import annotations

try:
    import torch
    from torch.futures import Future
    from torch.utils.data import (
        BatchSampler as BatchSampler,
        RandomSampler as RandomSampler,
        SequentialSampler as SequentialSampler,
        default_collate as default_collate
    )

    def unpickle_tensor(buffer, dtype, shape):
        return torch.frombuffer(buffer, dtype=dtype).view(*shape)

    def pickle_tensor(t):
        return unpickle_tensor, (t.contiguous().numpy().view("b"), t.dtype, t.shape)

    pkl_dispatch_table = {torch.Tensor: pickle_tensor}

except ImportError:
    import random
    import threading
    from typing import Callable, Generic, TypeVar, NamedTuple

    try:
        import numpy as np
    except ImportError:
        np = None

    pkl_dispatch_table = {}

    S = TypeVar("S")
    T = TypeVar("T")

    class Future(Generic[T]):
        def __init__(self, *, devices=None):
            self._value = None
            self._is_exception = None
            self._lock = threading.Lock()
            self._done_event = threading.Event()
            self._callbacks = []

        def done(self) -> bool:
            return self._done_event.is_set()

        def wait(self) -> T:
            self._done_event.wait()
            return self.value()

        def value(self) -> T:
            if not self._done_event.is_set():
                raise RuntimeError("value is not set")

            if self._is_exception:
                raise self._value
            else:
                return self._value

        def then(self, callback: Callable[[Future[T]], S]) -> Future[S]:
            f = self.__class__()

            def do(this):
                try:
                    v = callback(this)
                except BaseException as e:  #
                    f.set_exception(e)
                else:
                    f.set_result(v)

            self.add_done_callback(do)

            return f

        def add_done_callback(self, callback: Callable[[Future[T]], None]) -> None:
            self._callbacks.append(callback)

        def set_result(self, result: T) -> None:
            self._value = result
            self._is_exception = False

            threading.Thread(target=self._on_set).start()

        def set_exception(self, result: T) -> None:
            assert isinstance(
                result, Exception
            ), f"{result} is of type {type(result)}, not an Exception."

            self._value = result
            self._is_exception = True

            threading.Thread(target=self._on_set).start()

        def _on_set(self):
            self._done_event.set()

            for cb in self._callbacks:
                cb(self)

    class BatchSampler:
        def __init__(self, sampler, batch_size, drop_last):
            self.sampler = sampler
            self.batch_size = batch_size
            self.drop_last = drop_last

        def __len__(self):
            n = len(self.sampler)
            bs = self.batch_size

            if self.drop_last:
                return n // bs
            else:
                return n // bs + (1 if n % bs > 0 else 0)

        def __iter__(self):
            batch_items = []
            for i in self.sampler:
                batch_items.append(i)
                if len(batch_items) == self.batch_size:
                    yield batch_items
                    batch_items = []

            if not self.drop_last and len(batch_items) > 0:
                yield batch_items

    class RandomSampler:
        def __init__(self, dataset):
            self.n = len(dataset)

        def __len__(self):
            return self.n

        def __iter__(self):
            return iter(random.shuffle(range(self.n)))

    class SequentialSampler:
        def __init__(self, dataset):
            self.n = len(dataset)

        def __len__(self):
            return self.n

        def __iter__(self):
            return iter(range(self.n))

    def default_collate(batch):
        item = batch[0]

        if isinstance(item, NamedTuple):
            return item.__class__(*[default_collate(v) for v in batch])
        elif isinstance(item, (list, tuple)):
            return item.__class__(default_collate(v) for v in batch)
        elif isinstance(item, dict):
            return item.__class__((k, default_collate(v)) for k, v in batch.items())
        elif np is not None and isinstance(item, np.ndarray):
            return np.stack(batch)
        else:
            raise TypeError(f"unsupported batch item type: {item.__class__.__name__}")
