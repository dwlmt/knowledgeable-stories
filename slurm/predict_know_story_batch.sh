#!/usr/bin/env bash
#SBATCH -o /home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH -e /home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1
#SBATCH --mem=48g  # Memory
#SBATCH --cpus-per-task=12  # number of cpus to use - there are 32 on each node.

# Set EXP_BASE_NAME and BATCH_FILE_PATH

echo "============"
echo "Initialize Env ========"

set -e # fail fast

export CURRENT_TIME=$(date "+%Y_%m_%d_%H%M%S")

# Activate Conda
source /home/${USER}/miniconda3/bin/activate allennlp-0.9

echo "I'm running on ${SLURM_JOB_NODELIST}"
dt=$(date '+%d_%m_%y__%H_%M')
echo ${dt}

# Env variables
export STUDENT_ID=${USER}

# General training parameters
export CLUSTER_HOME="/home/${STUDENT_ID}"
export DATASET_SOURCE="${CLUSTER_HOME}/datasets/story_datasets/"
export EMBEDDER_VOCAB_SIZE=50259
export NUM_GPUS=1
export NUM_CPUS=12

declare -a ScratchPathArray=(/disk/scratch_big/ /disk/scratch1/ /disk/scratch2/ /disk/scratch/ /disk/scratch_fast/)

# Iterate the string array using for loop
for i in "${ScratchPathArray[@]}"; do
  echo ${i}
  if [ -d ${i} ]; then
    export SCRATCH_HOME="${i}/${STUDENT_ID}"
    mkdir -p ${SCRATCH_HOME}
    if [ -w ${SCRATCH_HOME} ]; then
      break
    fi
  fi
done



echo ${SCRATCH_HOME}

export EXP_ROOT="${CLUSTER_HOME}/git/knowledgeable-stories"

export EXP_ID="${EXP_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${CURRENT_TIME}"
export SERIAL_DIR="${SCRATCH_HOME}/${EXP_ID}"
export ALLENNLP_CACHE_ROOT="${SCRATCH_HOME}/allennlp_cache/"

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
export LINE=$(sed "${SLURM_ARRAY_TASK_ID}q;d" ${CLUSTER_HOME}/${BATCH_FILE_PATH}/${BATCH_FILE_NAME})
export PREDICTION_STORY_FILE="${CLUSTER_HOME}/${BATCH_FILE_PATH}/${LINE}"

export MODEL_ZIP=${CLUSTER_HOME}/${MODEL_PATH}

# Ensure the scratch home exists and CD to the experiment root level.
cd "${EXP_ROOT}" # helps AllenNLP behave

mkdir -p ${SERIAL_DIR}

echo "============"
echo "ALLENNLP Task========"

allennlp predict --include-package knowledgeablestories --predictor ${PREDICTOR} \
  ${MODEL_ZIP} \
  ${PREDICTION_STORY_FILE} --cuda-device -1 \
  --batch-size 1 \
  --overrides '{"model.lm_memory_cuda_device": 0, "model.lm_device": 0, "model.tdvae_device": 0}' \
  --output-file ${SERIAL_DIR}/${EXP_ID}_prediction_output.jsonl

echo "============"
echo "ALLENNLP Task finished"

export HEAD_EXP_DIR="${CLUSTER_HOME}/runs/${EXP_ID}"
mkdir -p "${HEAD_EXP_DIR}"
rsync -avuzhP "${SERIAL_DIR}/" "${HEAD_EXP_DIR}/" # Copy output onto headnode

rm -rf "${SERIAL_DIR}"

if [ ! -v COPY_DATASET ]; then
  echo "No dataset to delete"
else
  rm -rf ${DATASET_ROOT}
fi

echo "============"
echo "results synced"
