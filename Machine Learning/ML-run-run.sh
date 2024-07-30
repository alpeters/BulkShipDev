#!/bin/bash
#SBATCH --account=def-kasahara-ab
#SBATCH --mem=15G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=1:00:0

module load StdEnv/2020 python/3.8.10
virtualenv --no-download $SLURM_TMPDIR/envML
source $SLURM_TMPDIR/envML/bin/activate
pip install --no-index -r python_requirements_MLonly.txt
python ML-run.py
