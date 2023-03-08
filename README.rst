.. image:: https://img.shields.io/badge/doc-latest-brightgreen
   :target: https://cea-list.github.io/RPCDataloader
   :alt: Documentation
.. image:: https://github.com/CEA-LIST/RPCDataloader/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/CEA-LIST/RPCDataloader/actions/workflows/tests.yml
   :alt: Continuous tests

==============
RPC Dataloader
==============

This library implements a variant of the PyTorch Dataloader using remote workers.
It allows to distribute workers over remote servers rather than the one running the main script.

To use it, start one or several worker daemons on remote computers.
The machines running the data loaders will dispatch requests for items to the workers and await the returned values.

Though similar to `torch.rpc <https://pytorch.org/docs/stable/rpc.html>`_, this library uses its own implementation of RPC (Remote Procedure Call) which is simpler (no initialization) and does not conflict with the one from pytorch.


Installation
============

.. code:: shell

    pip install rpcdataloader


.. _Usage:

Usage
=====

To use the RPC dataloader, start a few workers either from the command line:

.. code:: shell

    python -m rpcdataloader.launch --host=0.0.0.0 --port=6543

or by calling :code:`rpcdataloader.run_worker` directly from a python script.

Then instantiate a remote dataset and dataloader:

.. code:: python

    dataset = rpcdataloader.RPCDataset(
        workers=['node01:6543', 'node02:5432'],
        dataset=torchvision.datasets.ImageFolder,
        root=args.data_path + "/train",
        transform=train_transform,
    )

    dataloader = rpcdataloader.RPCDataloader(
        dataset
        batch_size=2,
        shuffle=True,
        pin_memory=True)

    for minibatch in dataloader:
        ...


Further reading
===============

- `API documentation <https://cea-list.github.io/RPCDataloader>`_
- `ResNet50 training on ImageNet dataset <docs/example_rpc.py>`_
- `Slurm integration using heterogeneous jobs <docs/example_rpc.slurm>`_
