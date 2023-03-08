import random
import multiprocessing
import pytest
import torch

from rpcdataloader import run_worker, RPCDataloader, RPCDataset


@pytest.fixture(scope="function")
def workers():
    host = "127.0.0.1"
    port = random.randint(32000, 65536)

    workers = [(host, port + i) for i in range(2)]
    procs = [multiprocessing.Process(target=run_worker, args=(host, port, 10))
             for host, port in workers]
    for p in procs:
        p.start()

    yield [f"{h}:{p}" for h, p in workers]

    for p in procs:
        p.terminate()


def test_rpcdataloader(workers):
    dataset = RPCDataset(
        workers=workers,
        dataset=torch.rand,
        size=(1000, 128))

    dataloader = RPCDataloader(
        dataset,
        batch_size=5,
    )

    i = 0
    for d in dataloader:
        assert isinstance(d, torch.Tensor) and d.shape == (5, 128)
        i += 1

    assert i == 200
