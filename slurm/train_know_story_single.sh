#!/usr/bin/env bash
#SBATCH -o /home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH -e /home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:4
#SBATCH --mem=0  # Memory
#SBATCH --cpus-per-task=12  # number of cpus to use - there are 32 on each node.

# Set EXP_BASE_NAME and BATCH_FILE_PATH

set -e # fail fast

export CURRENT_TIME=$(date "+%Y_%m_%d_%H%M%S")

# Activate Conda
source /home/${USER}/miniconda3/bin/activate allennlp

echo "I'm running on ${SLURM_JOB_NODELIST}"
dt=$(date '+%d_%m_%y__%H_%M');
echo ${dt}

# Env variables
export STUDENT_ID=${USER}

export CLUSTER_HOME="/home/${STUDENT_ID}"
export EMBEDDER_VOCAB_SIZE=50269
export NUM_GPUS=4
export NUM_CPUS=12

declare -a ScratchPathArray=(/disk/scratch_big/${STUDENT_ID} /disk/scratch1/${STUDENT_ID} /disk/scratch2/${STUDENT_ID} /disk/scratch/${STUDENT_ID} /disk/scratch_fast/${STUDENT_ID} ${CLUSTER_HOME}/scratch/${STUDENT_ID})

# Iterate the string array using for loop
for i in "${ScratchPathArray[@]}"
do
    echo ${i}
    if [ -w ${i} ];then
      echo "WRITABLE"
      mkdir -p ${i}
      export SCRATCH_HOME=${i}
      break
   fi
done

echo ${SCRATCH_HOME}

export EXP_ROOT="${CLUSTER_HOME}/projects/knowledgeable-stories"

export SERIAL_DIR="${SCRATCH_HOME}/${EXP_NAME}_${CURRENT_TIME}"


# Ensure the scratch home exists and CD to the experiment root level.
cd "${EXP_ROOT}" # helps AllenNLP behave

rm -rf "${SERIAL_DIR}"
mkdir -p ${SERIAL_DIR}

echo "ALLENNLP Task========"

allennlp train --file-friendly-logging --include-package knowledgeablestories \
    -s  ${SERIAL_DIR}/${EXP_NAME}/ \
    --cache-directory ${SERIAL_DIR}/cache/ \
    ${EXP_CONFIG}

echo "============"
echo "ALLENNLP Task finished"

mkdir -p "${CLUSTER_HOME}/runs/${EXP_NAME}/"
rsync -avuzhP "${SERIAL_DIR}/${EXP_NAME}/" "${CLUSTER_HOME}/runs/${EXP_NAME}/" # Copy output onto headnode

rm -rf "${SERIAL_DIR}"

echo "============"
echo "results synced"