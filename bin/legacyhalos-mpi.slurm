#! /bin/bash

#SBATCH -A desi
#SBATCH -L project
#SBATCH -o legacyhalos.log.%j
#SBATCH --mail-user=jmoustakas@siena.edu
#SBATCH --mail-type=ALL

# #SBATCH -p regular
# #SBATCH -N 4
# #SBATCH -n 16
# #SBATCH -t 02:00:00

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 00:30:00

# for cori only
# #SBATCH -C haswell

# sbatch ~/scratch/repos/legacyhalos/bin/legacyhalos-coadds.slurm

# Example calculation of how many nodes I need, given a number of MPI tasks:
  # On Cori:   20 tasks: -c 4 -n 20 -N 20*4/32=2.5
  # On Edison: 20 tasks: -c 4 -n 20 -N 20*4/24=3.3

# Example calculation of scaling to the full sample on edison.  Edison limits
# the number of tasks (processes per node) to 48, so we have to increase the
# number of nodes until the total time is a reasonable number:
#
#   ngal = 1562      # [number of galaxies]
#   hourpergal = 0.5 # [hours per galaxy]
#   ntask = 48
#   nproc = 4
#   nnode = 32
#   totaltime = int(ngal*hourpergal*nproc/nnode/24)
# 
#     print('Edison: -n {} -N {:.0f} -c {} -t 00:{:02d}:00'.format(ntask, nnode, nproc, totaltime))
#       Edison: -n 48 -N 32 -c 4 -t 00:04:00
#
# Some notes: Running over 1562 galaxies with "-n 48 -N 32 -c 4 -t 00:04:00" left 917/1562 galaxies.
#
# Example calculation of scaling to the full sample on cori.  Here, we simply 
# increase the number of tasks (as multiples of 32) until the total time is a
# reasonable number:
#
#   ngal = 100      # [number of galaxies]
#   hourpergal = 0.5 # [hours per galaxy]
#   ntask = 32*1
#   nproc = 8
#   nnode = ntask*nproc//32
#   _totaltime = ngal/ntask*hourpergal
#   totaltime = np.round(ngal/ntask*hourpergal).astype(int)
#
#   print('Cori: -n {} -N {:.0f} -c {} -t {:02d}:00:00'.format(ntask, nnode, nproc, totaltime))
#
#     Cori: -n 160 -N 20 -c 4 -t 00:04:00

export IMPY_DIR=$HOME/repos/git/impy
source $IMPY_DIR/bin/legacyhalos-env-nersc

# do the whole sample!
# time srun -N 32 -n 48 -c 4 $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --coadds --nproc 4 --mpi

# pipeline coadds
#time srun -N 1 -n 4 -c 8 $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --first 0 --last 99 --coadds --nproc 8 --mpi
#time srun -N 1 -n 2 -c 16 $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --first 0 --last 49 --coadds --nproc 16 --mpi

# custom coadds
#time srun -N 6 -n 18 -c 8 $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --first 0 --last 99 --custom-coadds --nproc 8 --mpi --clobber

# ellipse
#time srun -N 1 -n 21 $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --first 0 --last 99 --ellipse --mpi

# html
time srun -N 1 -n 2 -c 12 $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --last 99 --html --nproc 12 --mpi --clobber
