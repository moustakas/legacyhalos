#!/bin/bash
#SBATCH -A desi
#SBATCH -C haswell
#SBATCH --qos=bigmem
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --module=mpich-cle6
#SBATCH --image=legacysurvey/legacyhalos:v0.0.3
#SBATCH --licenses=SCRATCH,cfs
#SBATCH --job-name=SGA-customredux-coadds.log-%j
#SBATCH --mail-user=jmoustakas@siena.edu
#SBATCH --mail-type=ALL
#SBATCH --mem=750GB

# coadds
time srun -N 1 -n 1 -c 32 shifter $LEGACYHALOS_CODE_DIR/bin/SGA/SGA-customredux-mpi.sh coadds 32
