#!/bin/bash
#SBATCH --account=def-kasahara-ab
#SBATCH --mem=17G
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:0    

module load StdEnv/2023 arrow/19.0.1 python/3.12
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r python_requirements_ShipDist_CC.txt
python AIS_Calculations_PP_Coast_EU.py
