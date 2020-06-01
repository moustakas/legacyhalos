# New Projects

This README briefly describes how the `legacyhalos` code base can be used on new
projects.  Every project is different, so using this repository will require
some customized code.

## Initial Setup

In this example let's create a project called `myproject`:

1. Clone the `legacyhalos` repository and set an environment variable pointing
to your local checkout. Unfortunately we're going to need this environment
variable outside of any of the `legacyhalos` software, so it's a good idea to
add this path to your `.bashrc` (or similar) startup file:

```bash
export LEGACYHALOS_CODE_DIR=/path/to/legacyhalos
export PATH=${LEGACYHALOS_CODE_DIR}/bin/myproject:${PATH}
```

2. Next, create a new dedicated subdirectory in `$LEGACYHALOS_CODE_DIR/bin` with
the name of your project:

```bash
mkdir -p $LEGACYHALOS_CODE_DIR/bin/myproject
```

    In this directory we'll need a handful of files, each of which are described
in more detail below:

```bash
$LEGACYHALOS_CODE_DIR/bin/myproject/myproject-env
$LEGACYHALOS_CODE_DIR/bin/myproject/myproject-mpi
$LEGACYHALOS_CODE_DIR/bin/myproject/myproject-mpi.sh
$LEGACYHALOS_CODE_DIR/bin/myproject/myproject-mpi-slurm
$LEGACYHALOS_CODE_DIR/bin/myproject/myproject-shifter
```

    (Note: the `-mpi` suffix is historical and not required.)

3. Next, create a Python module in `$LEGACYHALOS_CODE_DIR/py/legacyhalos` which
will contain the custom code for your project:

```bash
$LEGACYHALOS_CODE_DIR/py/legacyhalos/myproject.py
```

    This module can contain whatever code you want, but at minimum it will need to
contain functions to:

  - read your galaxy sample;
  - find missing/completed files;
  - specify the command-line arguments for your wrapper script; and
  - generate the HTML output.

## Running the code

Once all these pieces are in place you can process the sample three different
ways, as described in more detail below: (1) interactively with a single node;
(2) interactively with many nodes; or (3) using Slurm with one or many nodes.

1. To run the code interactively with a single node from a clean login, do:

```bash
salloc -N 1 -C haswell -A ACCOUNT -t 00:02:00 --qos interactive -L SCRATCH,cfs
myproject-shifter
source myproject-env 
myproject-mpi --help
```

2. To leverage the `MPI` capabilities of the code we'll need to use the
`myproject-mpi.sh` script and load the shifter image when we request the
resources. In this example, we execute `n=64` MPI tasks across `N=2` cori nodes
and `c=1` core per task:

```bash
salloc -N 2 -C haswell -A ACCOUNT -L cfs,SCRATCH -t 00:02:00 --qos interactive --image=legacysurvey/legacyhalos:v0.0.1
srun -n 64 -c 1 shifter --module=mpich-cle6 $LEGACYHALOS_CODE_DIR/bin/myproject/myproject-mpi.sh test 1 > myproject-test.log 2>&1 &
```

3. Finally, in production you can use the `SLURM` script and any of the
available NERSC queues. For example, to execute the same example as in the
previous use case using the `debug` queue, do:

```
sbatch $LEGACYHALOS_CODE_DIR/bin/myproject/myproject-mpi.slurm
```
## Customizing your scripts

**Write me.**
