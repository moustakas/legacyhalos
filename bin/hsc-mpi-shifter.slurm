#! /bin/bash
#SBATCH -A desi
# #SBATCH -C knl
#SBATCH -C haswell
#SBATCH -L project
#SBATCH -o hsc.log.%j
#SBATCH --mail-user=jmoustakas@siena.edu
#SBATCH --mail-type=ALL
#SBATCH --image=docker:flagnarg/legacyhalos:latest

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 00:30:00

# sbatch hsc-mpi-shifter.slurm

# htmlplots
#time srun -N 1 -n 32 shifter --image=docker:flagnarg/legacyhalos:latest ./hsc-mpi-shifter.sh

# ellipse
#time srun -N 2 -n 16 -c 4 shifter --image=docker:flagnarg/legacyhalos:latest ./hsc-mpi-shifter.sh
#time srun -N 8 -n 128 -c 4 shifter --image=docker:flagnarg/legacyhalos:latest ./hsc-mpi-shifter.sh

# coadds, custom coadds
time srun -N 1 -n 4 -c 8 shifter --image=docker:flagnarg/legacyhalos:latest ./hsc-mpi-shifter.sh
