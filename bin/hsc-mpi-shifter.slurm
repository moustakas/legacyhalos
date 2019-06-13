#! /bin/bash
#SBATCH -A desi
#SBATCH -C knl
# #SBATCH -C haswell
#SBATCH -L project
#SBATCH -o hsc.log.%j
#SBATCH --mail-user=jmoustakas@siena.edu
#SBATCH --mail-type=ALL
#SBATCH --image=docker:flagnarg/legacyhalos:latest

# #SBATCH -p regular
# #SBATCH -N 1
# #SBATCH -n 3
# #SBATCH -t 02:00:00

#SBATCH -p regular
#SBATCH -N 5
#SBATCH -n 40
#SBATCH -t 01:00:00

# sbatch hsc-mpi-shifter.slurm

# htmlplots
#time srun -N 1 -n 32 shifter --image=docker:flagnarg/legacyhalos:latest ./hsc-mpi-shifter.sh

# ellipse
#time srun -N 2 -n 8 -c 8 shifter --image=docker:flagnarg/legacyhalos:latest ./hsc-mpi-shifter.sh

# coadds, custom coadds
time srun -N 5 -n 40 -c 4 shifter --image=docker:flagnarg/legacyhalos:latest ./hsc-mpi-shifter.sh
