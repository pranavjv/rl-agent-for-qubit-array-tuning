#!/bin/bash
#SBATCH -A nvr_asicvlsi_quantum #<account name>
#SBATCH -J nvr_asicvlsi_quantum:rl_qarray_train1 #<job name of my choosing>
#SBATCH -t 03:58:00 #set time limit of job allocation. Limit of 4 hours.
#SBATCH -p batch #<partition name> probably batch
#SBATCH -N 1 #<number of nodes> each node is a DGX
#SBATCH --dependency=singleton #leave as singleton, give each job a different name
#SBATCH -o ./sbatch_logs/multi/%x_%j.out #%x chosen name %j automatically generated job ID
#SBATCH -e ./sbatch_logs/multi/%x_%j.err

#--mail-type=FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_50,TIME_LIMIT_80,TIME_LIMIT_90,END #send a request to put my account on allocation
#set -euxo pipefail #leave this because mystery

readonly _code_root="$(pwd)"

# mount the data and code directories
# Adjust this based on your cluster's filesystem structure
readonly _cont_mounts="${_code_root}:${_code_root},/tmp:/tmp"

# pull image from NGC/GitLab registry
readonly _cont_image='gitlab-master.nvidia.com/pvaidhyanath/rlqarray/rl-qarray-training:latest'

export NPROC_PER_NODE=8 #GPUs per node. Leave this.
export TOTAL_GPU=$(($SLURM_JOB_NUM_NODES * $NPROC_PER_NODE)) #Leave this.

# Create logs directory if it doesn't exist
mkdir -p sbatch_logs/multi

# Command to run inside the container
RUN_CMD="python -u src/swarm/training/train.py"

echo "Running on hosts: $(echo $(scontrol show hostname))"

srun -A nvr_asicvlsi_quantum \
     --container-image="${_cont_image}" \
     --container-mounts="${_cont_mounts}" \
     --ntasks-per-node=${NPROC_PER_NODE} \
     bash -c "
     ldconfig
     set -x
     export WORLD_SIZE=${TOTAL_GPU}
     export WORLD_RANK=\${PMIX_RANK}
     export HDF5_USE_FILE_LOCKING=FALSE
     export CUDNN_V8_API_ENABLED=1
     export OMP_NUM_THREADS=\${SLURM_CPUS_ON_NODE}
     unset TORCH_DISTRIBUTED_DEBUG
     cd ${_code_root}
     export PYTHONPATH=${_code_root}:\${PYTHONPATH}
     ${RUN_CMD}"