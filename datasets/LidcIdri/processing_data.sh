#!/bin/bash
#SBATCH -J analysis
#SBATCH --partition=monakhova --gres=gpu:nvidia_rtx_a6000:1
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 48:00:00                           # Time limit (hh:mm:ss)
#SBATCH --mem 128gb           
#SBATCH --requeue

# This is Cassandra's env path
export LD_LIBRARY_PATH=$HOME/ty392/.conda/envs/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
conda activate qutcc

export TMPDIR=/tmp
export XDG_RUNTIME_DIR=/tmp

python -c "import torch; print(torch.cuda.is_available())"
python data_processing.py