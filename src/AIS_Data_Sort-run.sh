#!/bin/bash
#SBATCH --account=def-kasahara-ab
#SBATCH --mem=187G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=1:00:0    

module load StdEnv/2020 gcc/9.3.0 arrow/7.0.0 python/3.8.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r req_daskdf202321.txt
python AIS_Data_Sort.py
