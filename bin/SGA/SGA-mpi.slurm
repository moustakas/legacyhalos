#! /bin/bash
#SBATCH -A desi
#SBATCH --module=mpich-cle6
#SBATCH -C haswell
#SBATCH -L cfs,SCRATCH
#SBATCH -o SGA.log-debug-%j
#SBATCH --mail-user=jmoustakas@siena.edu
#SBATCH --mail-type=ALL
#SBATCH --image=legacysurvey/legacyhalos:v0.0.1

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:02:00

# sbatch SGA-mpi.slurm

echo $LEGACYHALOS_CODE_DIR
time srun -N 1 -n 1 -c 1 shifter $LEGACYHALOS_CODE_DIR/bin/SGA/SGA-mpi.sh bob 1

# coadds
#time srun -N 64 -n 512 -c 4 shifter /global/u2/i/ioannis/repos/git/legacyhalos/bin/SGA/SGA-mpi.sh
#time srun -N 1 -n 1 -c 1 shifter /global/u2/i/ioannis/repos/git/legacyhalos/bin/SGA/SGA-mpi.sh
