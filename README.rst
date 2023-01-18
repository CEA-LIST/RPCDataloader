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
This allows to distribute workers over remote servers rather than the one running the main script.

To use it, start one or several worker daemons on remote computers.
The RPCDataloader on the main computer will dispatch requests for items to the workers and await the returned value.

Though similar to `torch.rpc <https://pytorch.org/docs/stable/rpc.html>`_, this library uses its own implementation of RPC (Remote Procedure Call) which is simpler (no initialization) and does not conflict with the one from pytorch.


Installation
============

.. code:: shell

    pip install rpcdataloader


Usage
=====

To use the RPC dataloader, start a few workers either from the command line:

.. code:: shell

    python -m rpcdataloader.launch --host=0.0.0.0 --port=6543

or by calling :code:`rpcdataloader.run_worker` directly from a python script.

Then instantiate the dataloader:

.. code:: python

    dataloader = rpcdataloader.RPCDataloader(
        workers=['node01:6543', 'node02:6543'],
        dataset=torchvision.datasets.FakeData,
        kwargs={'transform': torchvision.transforms.ToTensor()},
        batch_size=2,
        shuffle=True,
        pin_memory=True)

    for minibatch in dataloader:
        ...


Slurm integration
=================

Slurm integration is a little tricky as it relies on a rather exotic functionality: `heterogeneous jobs <https://slurm.schedmd.com/heterogeneous_jobs.html>`_.
To distribute your workers on cpu nodes and your trainers on GPU nodes, use the following slurm script template:

.. code:: shell

    #!/usr/bin/env sh
    #SBATCH --time=3-00:00:00

    #SBATCH --partition=gpu
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=2
    #SBATCH --cpus-per-task=2
    #SBATCH --mem=64G
    #SBATCH --gres=gpu:2

    #SBATCH hetjob

    #SBATCH --partition=cpu
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=16
    #SBATCH --cpus-per-task=2
    #SBATCH --mem=72G

    # create an output dir
    export OUT_DIR="./outputs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}"
    mkdir -p $OUT_DIR

    # start workers and collect host and port list
    rm -f ${OUT_DIR}/workers && touch ${OUT_DIR}/workers
    srun --het-group=1 -I --exclusive --kill-on-bad-exit=1 \
        sh -c '
            export port=$(( 16384 + $RANDOM % 49182 ))
            echo $(hostname):$port \
                | flock ${OUT_DIR}/workers tee -a ${OUT_DIR}/workers \
                &> /dev/null
            python -u -m rpcdataloader.launch --host=0.0.0.0 --port=$port
            ' &
    worker_task_pid=$!

    # block until all workers have written their address and port
    tail -f ${OUT_DIR}/workers | head -n $SLURM_NTASKS_PER_NODE_HET_GROUP_1

    # parse worker list
    export workers=$(tr '\n' ' ' < ${OUT_DIR}/workers)

    # run training script
    export MASTER_ADDR="$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)"
    export MASTER_PORT=$(( 16384 + $RANDOM % 49182 ))
    srun --het-group=0 -I --exclusive --kill-on-bad-exit=1 \
        python -u example.py \
            --workers $workers

    # stop workers
    kill $worker_task_pid
