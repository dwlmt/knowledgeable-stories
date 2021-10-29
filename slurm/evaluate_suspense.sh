#!/usr/bin/env bash
#SBATCH -o /home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH -e /home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1
#SBATCH --mem=24g  # Memory
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

export EXP_ID="${EXP_NAME}_${SLURM_JOB_ID}_${CURRENT_TIME}"

# General training parameters
export CLUSTER_HOME="/home/${STUDENT_ID}"

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

export SERIAL_DIR="${SCRATCH_HOME}/${EXP_ID}/"

export PREDICTION_STORY_FILE="${CLUSTER_HOME}/${BATCH_FILE_PATH}"


# Ensure the scratch home exists and CD to the experiment root level.
cd "${EXP_ROOT}" # helps AllenNLP behave

mkdir -p ${SERIAL_DIR}

echo "============"
echo "Evaluate Suspense Task========"


python ./scripts/evaluate_stories.py \
--prediction-json ${PREDICTION_JSON} --annotator-targets ${ANNOTATOR_TARGETS} \
 --output-dir  ${SERIAL_DIR}/ \
 --exclude-worker-ids "${EXCLUDE_WORKER_IDS}"

echo "============"
echo "Evaluate Suspense Task finished"

export HEAD_EXP_DIR="${CLUSTER_HOME}/runs/${EXP_ID}/"
mkdir -p "${HEAD_EXP_DIR}"
rsync -avuzhP "${SERIAL_DIR}/" "${HEAD_EXP_DIR}/" # Copy output onto headnode

rm -rf "${SERIAL_DIR}"

echo "============"
echo "results synced"
