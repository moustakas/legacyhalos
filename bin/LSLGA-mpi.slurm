#! /bin/bash
#SBATCH -A desi
#SBATCH --module=mpich-cle6
#SBATCH -C haswell
#SBATCH -L project
#SBATCH -o LSLGA.log-debug-%j
#SBATCH --mail-user=jmoustakas@siena.edu
#SBATCH --mail-type=ALL
#SBATCH --image=docker:flagnarg/legacyhalos:v0.0.6

#SBATCH -p debug
#SBATCH -N 2
#SBATCH -n 16
#SBATCH -t 00:30:00

# sbatch LSLGA-mpi.slurm

# htmlplots
#time srun -N 1 -n 32 shifter ./hsc-mpi-shifter.sh

# ellipse
#time srun -N 2 -n 16 -c 4 shifter ./hsc-mpi-shifter.sh
#time srun -N 8 -n 128 -c 4 shifter ./hsc-mpi-shifter.sh

# coadds, custom coadds
time srun -N 2 -n 16 -c 4 shifter ./LSLGA-mpi.sh

# htmlplots
# time srun -N 1 -n 32 shifter ./LSLGA-mpi.sh
