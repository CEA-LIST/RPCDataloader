#!/usr/bin/env sh
#SBATCH --time=0-01:00:00

# Resource specfification for training scripts
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:2

#SBATCH hetjob

# Resource specfification for workers
#SBATCH --partition=cpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rpcdataloader

export rpc_port_start=15000

# identify workers
export tmpfile="${TMPDIR:-/tmp}/rpcdataloader_workers.$SLURM_JOB_ID"
srun --het-group=1 -I --exclusive --kill-on-bad-exit=1 sh -c '
    echo $(hostname):$(( $rpc_port_start + $SLURM_LOCALID ))
    ' > "${tmpfile}"
readarray -t workers < "${tmpfile}"
rm $tmpfile

# start workers in background
srun --het-group=1 -I --exclusive --kill-on-bad-exit=1 sh -c '
    python -u -m rpcdataloader.launch \
        --host=0.0.0.0 \
        --port=$(( $rpc_port_start + $SLURM_LOCALID ))
    ' &
worker_task_pid=$!

# run training script
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_0 | head -n 1)
export MASTER_PORT=16000
srun --het-group=0 -I --exclusive --kill-on-bad-exit=1 \
    python -u example_rpc.py \
        --workers ${workers[@]} \
        --data-path=/media/ILSVRC \
        --batch-size=128

# stop workers
kill $worker_task_pid
