#!/bin/bash
#SBATCH --account=def-kasahara-ab
#SBATCH --mem=15G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:30:0    

module load StdEnv/2020
module load gcc/9.3.0
module load arrow/7.0.0
module load python/3.8.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r requirements_env_comp.txt

python AIS_Data_CompatFix.py
