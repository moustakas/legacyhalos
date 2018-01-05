#! /bin/bash

#SBATCH -p debug
#SBATCH -N 10
#SBATCH -n 30
#SBATCH -c 16
#SBATCH -L cscratch1,project
#SBATCH -C haswell
#SBATCH -t 00:30:00
#SBATCH -o legacyhalos-coadds.log.%j

export OMP_NUM_THREADS=1
export MPICH_GNI_FORK_MODE=1
export KMP_AFFINITY=disabled

dr=dr5
desiconda_version=20180102-1.2.2-img

module use /global/common/software/desi/$NERSC_HOST/desiconda/$desiconda_version/modulefiles
module load desiconda

export LEGACYHALOS_DIR=${CSCRATCH}/legacyhalos
export LEGACYPIPE_DIR=${CSCRATCH}/repos/legacypipe
export LEGACYHALOS_CODE_DIR=${CSCRATCH}/repos/legacyhalos

export PATH=$LEGACYPIPE_DIR/bin:${PATH}
export PATH=$LEGACYHALOS_CODE_DIR/bin:${PATH}

export PYTHONPATH=$LEGACYPIPE_DIR/py:${PYTHONPATH}
export PYTHONPATH=$LEGACYHALOS_CODE_DIR:${PYTHONPATH}

export LEGACY_SURVEY_DIR=/global/cscratch1/sd/dstn/$dr

module use $LEGACYPIPE_DIR/bin/modulefiles/cori
module load unwise_coadds
module load unwise_coadds_timeresolved
module load dust

time srun --cpu_bind=cores -N 3 -n 3 mpi-legacyhalos-coadds -v
