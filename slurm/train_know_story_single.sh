#!/usr/bin/env bash
#SBATCH -o /home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH -e /home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1
#SBATCH --mem=16g  # Memory
#SBATCH --cpus-per-task=8  # number of cpus to use - there are 32 on each node.
#SBATCH --mail-type=all          # send email on job start, end and fail
#SBATCH --mail-user=david.wilmot@ed.ac.uk

# Set EXP_BASE_NAME and BATCH_FILE_PATH

echo "============"
echo "Initialize Env ========"

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
export DATASET_SOURCE="${CLUSTER_HOME}/datasets/story_datasets/"
export EMBEDDER_VOCAB_SIZE=50268
export NUM_GPUS=1
export NUM_CPUS=8

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

#rm -rf "${SCRATCH_HOME}/*"

export EXP_ROOT="${CLUSTER_HOME}/projects/knowledgeable-stories"

export EXP_ID="${EXP_NAME}_${CURRENT_TIME}"
export SERIAL_DIR="${SCRATCH_HOME}/${EXP_ID}"
export CACHE_DIR="${SCRATCH_HOME}/${EXP_ID}_cache"

if [ ! -v COPY_DATASET ]; then
  export DATASET_ROOT=${DATASET_SOURCE}
else
  export DATASET_ROOT="${SCRATCH_HOME}/${EXP_ID}_dataset"
fi

echo "============"
echo "Copy Datasets locally ========"

if [ ! -v COPY_DATASET ]; then
  echo "Don't copy dataset"
else
  mkdir -p ${DATASET_ROOT}
  rsync -avuzhP "${DATASET_SOURCE}/" "${DATASET_ROOT}/"
fi

# Ensure the scratch home exists and CD to the experiment root level.
cd "${EXP_ROOT}" # helps AllenNLP behave

mkdir -p ${SERIAL_DIR}
mkdir -p ${CACHE_DIR}

echo "============"
echo "ALLENNLP Task========"

allennlp train --file-friendly-logging --include-package knowledgeablestories \
    -s  ${SERIAL_DIR}/ \
    --cache-directory ${CACHE_DIR} \
    ${EXP_CONFIG}

echo "============"
echo "ALLENNLP Task finished"

export HEAD_EXP_DIR="${CLUSTER_HOME}/runs/${EXP_ID}"
mkdir -p "${HEAD_EXP_DIR}"
rsync -avuzhP "${SERIAL_DIR}/" "${HEAD_EXP_DIR}/" # Copy output onto headnode

rm -rf "${SERIAL_DIR}"
rm -rf "${CACHE_DIR}"

if [ ! -v COPY_DATASET ]; then
  echo "No dataset to delete"
else
  rm -rf ${DATASET_ROOT}
fi

echo "============"
echo "results synced"