import os
import random
import multiprocessing
import pytest
import torch

from rpcdataloader import run_worker, rpc_async


def do(a):
    return a, torch.rand(1024)


@pytest.fixture(scope='function')
def worker():
    host = '127.0.0.1'
    port = random.randint(32000, 65536)

    worker = multiprocessing.Process(target=run_worker, args=(host, port, 10))
    worker.start()

    yield f"{host}:{port}"

    worker.terminate()


def test_rpc_async(worker):
    a = bytes([random.randint(0, 255) for _ in range(0, 1000)])
    f = rpc_async(worker, do, (a,))
    b, c = f.wait()

    assert a == b
    assert len(c) == 1024


def test_rref(worker):
    a = torch.rand([500])
    b = torch.rand([500])
    f = rpc_async(worker, torch.add, (a, b), rref=True).wait()
    actual = rpc_async(worker, torch.sum, (f,), rref=False).wait().item()
    expected = (a + b).sum().item()

    assert actual == pytest.approx(expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda unvailable")
def test_rpc_pin(worker):
    a = bytes([random.randint(0, 255) for _ in range(0, 1000)])
    f = rpc_async(worker, do, (a,), pin_memory=True)
    _, c = f.wait()

    assert c.is_pinned
