ssh -Y petersal@cedar.computecanada.ca

module load python/3.8.10
virtualenv --no-download ~/virtualenv/dask
source ~/virtualenv/dask/bin/activate
pip install --no-index --upgrade pip
pip install jupyter dask dask-jobqueue distributed graphviz bokeh fastparquet --no-index
pip freeze > requirements.txt


### Jupyter


sshuttle --dns -Nr petersal@cedar.computecanada.ca

salloc --account=def-kasahara-ab --ntasks=1
cd $SCRATCH/ShippingEmissions
source ~/virtualenv/dask/bin/activate
jupyter-notebook --ip `hostname -f` --no-browser

#####

#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=15:0:0    
#SBATCH --mail-user=<youremail@gmail.com>
#SBATCH --mail-type=ALL


module load python/3.8.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r requirements.txt
python ...


pip install --no-index -r requirements.txt




cp -r /scratch/petersal/ShippingEmissions/src/data/AIS/ais_bulkers $SLURM_TMPDIR
cp -r $SLURM_TMPDIR/ais_bulkers_indexed /scratch/petersal/ShippingEmissions/src/data/AIS