#!/bin/bash
#SBATCH -p astro
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=log/eval_%j.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=t.ranasinghe@lancaster.ac.uk



# ============================================================
# Environment
# ============================================================

source /etc/profile

module add anaconda3/2023.09
module add cuda/12.0

source activate \
    /storage/hpc/37/ranasint/conda_envs/llm_exp


# ============================================================
# Scratch locations
# ============================================================

SCRATCH=/scratch/hpc/37/ranasint

# ------------------------------------------------------------
# Hugging Face
# ------------------------------------------------------------

export HF_HOME="${SCRATCH}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_TOKEN=

# ------------------------------------------------------------
# MTEB
#
# IMPORTANT:
# MTEB has its own result cache. HF_HOME does not control it.
# ------------------------------------------------------------

export MTEB_CACHE="${SCRATCH}/mteb_cache"


# ------------------------------------------------------------
# General Python/library caches
# ------------------------------------------------------------

export XDG_CACHE_HOME="${SCRATCH}/cache"
export TORCH_HOME="${SCRATCH}/torch_cache"


# ------------------------------------------------------------
# Runtime
# ------------------------------------------------------------

export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8


# ============================================================
# Project
# ============================================================

cd "$SLURM_SUBMIT_DIR"

mkdir -p log

mkdir -p "${HF_HOME}"
mkdir -p "${HF_HUB_CACHE}"
mkdir -p "${HF_DATASETS_CACHE}"
mkdir -p "${MTEB_CACHE}"
mkdir -p "${XDG_CACHE_HOME}"
mkdir -p "${TORCH_HOME}"


# ============================================================
# Python package location
#
# Our project uses:
#
#   src/simteb/
#
# Therefore src must be on PYTHONPATH.
# ============================================================

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"


# ============================================================
# Experiment
# ============================================================

MODEL="intfloat/multilingual-e5-large"


BATCH_SIZE=128


# ============================================================
# Information
# ============================================================

echo "============================================================"
echo "SiMTEB evaluation"
echo "============================================================"

echo "Job ID:        ${SLURM_JOB_ID}"
echo "Host:          $(hostname)"
echo "Working dir:   $(pwd)"

echo

echo "Model:         ${MODEL}"
echo "Batch size:    ${BATCH_SIZE}"

echo

echo "HF_HOME:       ${HF_HOME}"
echo "HF_HUB_CACHE:  ${HF_HUB_CACHE}"
echo "HF_DATASETS:   ${HF_DATASETS_CACHE}"
echo "MTEB_CACHE:    ${MTEB_CACHE}"
echo "XDG_CACHE:     ${XDG_CACHE_HOME}"
echo "TORCH_HOME:    ${TORCH_HOME}"
echo "PYTHONPATH:    ${PYTHONPATH}"

echo "============================================================"


# ============================================================
# GPU information
# ============================================================

nvidia-smi



# ============================================================
# Run evaluation
# ============================================================

echo
echo "Starting evaluation..."
echo

python -u evaluate.py \
    --model "${MODEL}" \
    --batch-size "${BATCH_SIZE}" \
    --cache-folder "${MTEB_CACHE}"


# ============================================================
# Successful completion
# ============================================================

echo
echo "============================================================"
echo "Evaluation completed successfully."
echo "MTEB results:"
echo "${MTEB_CACHE}"
echo "============================================================"