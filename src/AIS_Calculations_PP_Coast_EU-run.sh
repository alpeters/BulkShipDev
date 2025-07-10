#!/bin/bash
#SBATCH --account=def-kasahara-ab
#SBATCH --mem=17G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=1:00:0    

module load StdEnv/2023 proj arrow/19.0.1 python/3.12

virtualenv --no-download $SLURM_TMPDIR/env

source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ../../ShippingEmissions/python_requirements_ShipDist_CC.txt

python AIS_Calculations_PP_Coast_EU.py
