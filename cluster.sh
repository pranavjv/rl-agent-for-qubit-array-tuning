#SBATCH -A nvr_asicvlsi_quantum PPP - ours is coreai_libraries_cuquantum
#SBATCH -J nvr_asicvlsi_quantum:demo_train1 #<job name of my choosing>? Looping script for each name.
#SBATCH -t 03:58:00 #set time limit of job allocation. Limit of 4 hours.
#SBATCH -p batch #<partition name> probably batch
#SBATCH -N 1 #<number of nodes> each node is a DGX, so I will have to modify my script to run 2 jobs at once
#SBATCH --dependency=singleton #leave as singleton, give each job a different name
#SBATCH -o ./sbatch_logs/multi/%x_%j.out #%x chosen name %j automatically generated job ID
#SBATCH -e ./sbatch_logs/multi/%x_%j.err


#--mail-type=FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_50,TIME_LIMIT_80,TIME_LIMIT_90,END #send a request to put my account on allocation
#set -euxo pipefail #leave this because mystery

set -e
set -u

mkdir -p logs #sbatch logs placed in here

DOCKER_IMAGE="gitlab-master.nvidia.com/pvaidhyanath/rlqarray/rl-qarray-training:latest"

# Define the working directory (current project directory)
WORK_DIR=$(pwd)

# Define any additional data directories you need to mount
# DATA_DIR="/path/to/your/data"  # Uncomment and set if needed

# Number of GPUs per node
NPROC_PER_NODE=2
TOTAL_GPU=$(($SLURM_JOB_NUM_NODES * $NPROC_PER_NODE))

# Command to run inside the container
RUN_CMD="python src/swarm/training/train.py"

echo "Running on hosts: $(scontrol show hostname)"
echo "Using Docker image: ${DOCKER_IMAGE}"

# Check if nvidia-docker is available, otherwise use regular docker with GPU flags
if command -v nvidia-docker &> /dev/null; then
    DOCKER_CMD="nvidia-docker"
else
    DOCKER_CMD="docker"
fi

srun --ntasks=1 \
     bash -c "
     set -x
     # Export environment variables needed for distributed training
     export WORLD_SIZE=${TOTAL_GPU}
     export WORLD_RANK=\${PMIX_RANK:-0}
     export CUDA_VISIBLE_DEVICES=0,1
     
     # Pull the latest image (will use cached version if already present)
     ${DOCKER_CMD} pull ${DOCKER_IMAGE}
     
     # Run the Docker container
     ${DOCKER_CMD} run --rm \
                   --gpus all \
                   --ipc=host \
                   -v ${WORK_DIR}:${WORK_DIR} \
                   -w ${WORK_DIR} \
                   -e WORLD_SIZE=${WORLD_SIZE} \
                   -e WORLD_RANK=\${WORLD_RANK} \
                   -e CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES} \
                   ${DOCKER_IMAGE} \
                   ${RUN_CMD}
     "


