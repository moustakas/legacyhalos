#! /bin/bash
#SBATCH -A desi
#SBATCH --module=mpich-cle6
#SBATCH -C haswell
#SBATCH -L cfs,SCRATCH
#SBATCH -o SGA-test.log-%j
#SBATCH --image=legacysurvey/legacyhalos:v0.0.1

#SBATCH -p debug
#SBATCH -N 2
#SBATCH -n 64
#SBATCH -t 00:02:00

# sbatch SGA-mpi.slurm

# test
time srun -N 2 -n 64 -c 1 shifter $LEGACYHALOS_CODE_DIR/bin/SGA/SGA-mpi.sh test 1

# coadds
#time srun -N 64 -n 512 -c 4 shifter /global/u2/i/ioannis/repos/git/legacyhalos/bin/SGA/SGA-mpi.sh
#time srun -N 1 -n 1 -c 1 shifter /global/u2/i/ioannis/repos/git/legacyhalos/bin/SGA/SGA-mpi.sh
