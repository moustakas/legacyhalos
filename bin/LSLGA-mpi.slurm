#! /bin/bash
#SBATCH -A desi
#SBATCH --module=mpich-cle6
# #SBATCH -C knl
#SBATCH -C haswell
#SBATCH -L project
#SBATCH -o LSLGA.log.%j
#SBATCH --mail-user=jmoustakas@siena.edu
#SBATCH --mail-type=ALL
#SBATCH --image=docker:flagnarg/legacyhalos:latest

# #SBATCH --image=docker:flagnarg/legacyhalos:latest

#SBATCH -p debug
#SBATCH -N 1 # 10
#SBATCH -n 32 # 40
#SBATCH -t 00:30:00

# sbatch LSLGA-mpi.slurm

# htmlplots
#time srun -N 1 -n 32 shifter --image=docker:flagnarg/legacyhalos:latest ./hsc-mpi-shifter.sh

# ellipse
#time srun -N 2 -n 16 -c 4 shifter --image=docker:flagnarg/legacyhalos:latest ./hsc-mpi-shifter.sh
#time srun -N 8 -n 128 -c 4 shifter --image=docker:flagnarg/legacyhalos:latest ./hsc-mpi-shifter.sh

# coadds, custom coadds
#time srun -N 4 -n 16 -c 8 shifter --image=docker:flagnarg/legacyhalos:latest ./LSLGA-mpi.sh

# htmlplots
time srun -N 1 -n 32 shifter --image=docker:flagnarg/legacyhalos:latest ./LSLGA-mpi.sh
