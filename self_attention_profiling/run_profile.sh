#!/bin/bash
#SBATCH --job-name=profile_self_attention
#SBATCH --output=profile_self_attention_%j.out
#SBATCH --error=profile_self_attention_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=mig_class
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

source /data/apps/go.sh
source venv/bin/activate

python profile_self_attention.py