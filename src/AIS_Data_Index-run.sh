#!/bin/bash
#SBATCH --account=def-kasahara-ab
#SBATCH --mem=120G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=0:45:0    
#SBATCH --mail-user=apeters@protonmail.com
#SBATCH --mail-type=ALL

module load python/3.8.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index dask[dataframe] fastparquet pandas numpy
python AIS_Data_Index.py