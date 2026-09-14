#!/bin/bash
#SBATCH -p astro
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=log/eval_%j.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=t.ranasinghe@lancaster.ac.uk


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

source /etc/profile
module add anaconda3/2023.09
module add cuda/12.0

source activate /storage/hpc/37/ranasint/conda_envs/llm_exp

export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

# Hugging Face cache on scratch
SCRATCH=/scratch/hpc/37/ranasint
export HF_HOME="${SCRATCH}/hf_cache"
export HF_TOKEN=

# ---------------------------------------------------------
# Project
# ---------------------------------------------------------

cd "$SLURM_SUBMIT_DIR"
mkdir -p log
mkdir -p results

MODEL="intfloat/multilingual-e5-large"

echo "=================================================="
echo "SiMTEB evaluation"
echo "=================================================="
echo "Model:       $MODEL"
echo "Host:        $(hostname)"
echo "Job ID:      $SLURM_JOB_ID"
echo "GPU(s):      ${CUDA_VISIBLE_DEVICES:-unknown}"
echo "Project dir: $(pwd)"
echo "=================================================="

nvidia-smi

# ---------------------------------------------------------
# Run SOLD
# ---------------------------------------------------------

python -u evaluate.py \
    --model "$MODEL" \
    --batch-size 128

echo "=================================================="
echo "Evaluation finished."
echo "=================================================="