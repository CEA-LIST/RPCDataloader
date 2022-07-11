==============
RPC Dataloader
==============

This library implements a variant of the PyTorch Dataloader using remote workers.
This allows to distribute workers over remote servers rather than the one running the main script.

To use it, start one or several worker daemons on remote computers.
The RPCDataloader on the main computer will dispatch requests for items to the workers and await the returned value.

Though similar to `torch.rpc <https://pytorch.org/docs/stable/rpc.html>`_, this library uses its own implementation of RPC (Remote Procedure Call) which is simpler (no initialization) and does not conflict with the one from pytorch.


Usage
=====

To use the RPC dataloader, start a few workers either from the command line:

.. code:: shell

    python -m rpcdataloader.launch --host=0.0.0.0 --port=6543

or by calling :code:`rpcdataloader.run_worker`.

Then in your script, instantiate the dataloader:

.. code:: python

    class MyDataset(torch.utils.data.Dataset):
        def __init__(self, root, train=True):
            ...

    dataloader = rpcdataloader.RPCDataloader(
        workers=['node01:6543'],
        dataset=MyDataset,
        kwargs={'root': '/data/mydataset', 'train': True},
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

    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate myenv

    export OMP_NUM_THREADS=2
    export MASTER_ADDR="$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)"
    export MASTER_PORT=$((29400 + $SLURM_JOB_ID % 10000))
    export PYTHONPATH=$PWD

    # create an output dir
    export OUT_DIR="${HOME}/Experiments/dirange/${SLURM_JOB_NAME}.${SLURM_JOB_ID}"
    mkdir -p $OUT_DIR

    # start workers and collect host and port list
    # the subshell is needed because SLURM_PROCID is a task variable
    rm -f ${OUT_DIR}/workers
    srun --het-group=1 -I --exclusive --exact --kill-on-bad-exit=1 sh -c '
        port=$(( $MASTER_PORT + $SLURM_PROCID - $SLURM_NTASKS_HET_GROUP_0 + 1 ))
        python -u -m rpc.launch --host=0.0.0.0 --port=$port >> ${OUT_DIR}/workers
        ' &
    worker_task_pid=$?

    # wait for workers to start and parse worker list
    tail -F -f ${OUT_DIR}/workers | head -n $SLURM_NTASKS_PER_NODE_HET_GROUP_1 > /dev/null
    export workers=$(tr '\n' ' ' < ${OUT_DIR}/workers)

    # run training script
    srun --het-group=0 -I --exclusive --exact --kill-on-bad-exit=1 \
        python -u experiments/sem/train_rpc.py --workers $workers

    # stop workers
    kill $worker_task_pid
