#! /bin/bash
#SBATCH -A desi
#SBATCH --module=mpich-cle6
#SBATCH -C haswell
#SBATCH -L cfs,SCRATCH
#SBATCH -o SGA.log-debug-%j
#SBATCH --mail-user=jmoustakas@siena.edu
#SBATCH --mail-type=ALL
#SBATCH --image=docker:legacysurvey/legacyhalos:latest

#SBATCH -p debug
#SBATCH -N 64
#SBATCH -n 512
#SBATCH -t 00:30:00

# sbatch SGA-mpi.slurm

# coadds
time srun -N 64 -n 512 -c 4 shifter /global/u2/i/ioannis/repos/git/legacyhalos/bin/SGA/SGA-mpi.sh
