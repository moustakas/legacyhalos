#! /bin/bash
#SBATCH -A desi
#SBATCH --module=mpich-cle6
#SBATCH -C haswell
#SBATCH -L cfs,SCRATCH
#SBATCH -o LSLGA.log-debug-%j
#SBATCH --mail-user=jmoustakas@siena.edu
#SBATCH --mail-type=ALL
#SBATCH --image=docker:flagnarg/legacyhalos:v0.0.6

#SBATCH -p debug
#SBATCH -N 64
#SBATCH -n 512
#SBATCH -t 00:30:00

# sbatch LSLGA-mpi.slurm

# htmlplots
#time srun -N 1 -n 32 shifter ./hsc-mpi-shifter.sh

# ellipse
#time srun -N 2 -n 16 -c 4 shifter ./hsc-mpi-shifter.sh
#time srun -N 8 -n 128 -c 4 shifter ./hsc-mpi-shifter.sh

# pipeline, largegalaxy coadds
time srun -N 64 -n 512 -c 4 shifter /global/u2/i/ioannis/repos/git/legacyhalos/bin/LSLGA-mpi.sh

# htmlplots
#time srun -N 1 -n 32 -c 1 shifter /global/u2/i/ioannis/repos/git/legacyhalos/bin/LSLGA-mpi.sh
