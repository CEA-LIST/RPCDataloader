import os
import random
import multiprocessing
import pytest
import torch

from rpcdataloader import run_worker, rpc_async


def do(a):
    return a, os.urandom(1024) if torch is None else torch.rand(1024)


@pytest.fixture(scope='function')
def worker():
    host = '127.0.0.1'
    port = random.randint(32000, 65536)

    worker = multiprocessing.Process(target=run_worker, args=(host, port, 10))
    worker.start()

    yield host, port

    worker.terminate()


def test_rpc_async(worker):
    host, port = worker

    a = bytes([random.randint(0, 255) for _ in range(0, 1000)])
    f = rpc_async(host, port, do, (a,))
    b, c = f.wait()

    assert a == b
    assert len(c) == 1024


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda unvailable")
def test_rpc_pin(worker):
    host, port = worker

    a = bytes([random.randint(0, 255) for _ in range(0, 1000)])
    f = rpc_async(host, port, do, (a,), pin_memory=True)
    _, c = f.wait()

    assert c.is_pinned
