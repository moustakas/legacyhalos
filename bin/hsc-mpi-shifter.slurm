#! /bin/bash
#SBATCH -A desi
#SBATCH -C haswell # knl
#SBATCH -L project
#SBATCH -o hsc.log.%j
#SBATCH --mail-user=jmoustakas@siena.edu
#SBATCH --mail-type=ALL
#SBATCH --image=docker:flagnarg/legacyhalos:latest

# #SBATCH -p regular
# #SBATCH -N 1
# #SBATCH -n 3
# #SBATCH -t 02:00:00

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -t 00:30:00

# sbatch hsc-mpi-shifter.slurm

# htmlplots
time srun -N 1 -n 32 shifter --image=docker:flagnarg/legacyhalos:latest ./hsc-mpi-shifter.sh

# ellipse
#time srun -N 2 -n 8 -c 8 shifter --image=docker:flagnarg/legacyhalos:latest ./hsc-mpi-shifter.sh

# coadds, custom coadds
#time srun -N 4 -n 16 -c 8 shifter --image=docker:flagnarg/legacyhalos:latest ./hsc-mpi-shifter.sh
