# New Projects

This README briefly describes how the `legacyhalos` code base can be used on new
projects.  Every project is different, so using this repository will require
some customized code.

In this example let's create a project called 'myproject':

1. Clone the `legacyhalos` repository and set an environment variable pointing
to your local checkout. Unfortunately we're going to need this environment
variable outside of any of the `legacyhalos` software, so it's a good idea to
add this variable to your *.bashrc* startup file (and the `bin` subdirectory
pointing to your project, which we will create in a moment, to your path):

```bash
export LEGACYHALOS_CODE_DIR=/path/to/legacyhalos
export PATH=${LEGACYHALOS_CODE_DIR}/bin/myproject:${PATH}
```

2. Next, create a new directory in `$LEGACYHALOS_CODE_DIR/bin` with the name of
your project, e.g.:

```bash
mkdir $LEGACYHALOS_CODE_DIR/bin/myproject
```

In this directory we'll need a handful of files, each of which are described in
more detail below:

```bash
$LEGACYHALOS_CODE_DIR/bin/myproject/myproject-env
$LEGACYHALOS_CODE_DIR/bin/myproject/myproject-mpi
$LEGACYHALOS_CODE_DIR/bin/myproject/myproject-mpi.sh
$LEGACYHALOS_CODE_DIR/bin/myproject/myproject-mpi-slurm
$LEGACYHALOS_CODE_DIR/bin/myproject/myproject-shifter
```

(Note: the `-mpi` suffix is historical and not required.)

3. Create a Python module in `$LEGACYHALOS_CODE_DIR/py/legacyhalos` which will
contain the custom code for your project, e.g.:

```bash
$LEGACYHALOS_CODE_DIR/py/legacyhalos/myproject.py
```

This module can contain whatever you want, but at minimum it will need to
contain functions to:

  - read your galaxy sample;
  - find missing/completed files;
  - specify the command-line arguments for your wrapper script; and
  - generate the HTML output.

Once all these pieces are in place you can process the sample three different
ways, as described in more detail below: (1) interactively with a single node;
(2) interactively with many nodes; or (3) using Slurm with one or many nodes.

1. To run the code interactively with a single node from a clean login, run:

```bash
salloc -N 1 -C haswell -A ACCOUNT -t 01:00:00 --qos interactive -L SCRATCH,cfs
myproject-shifter
source myproject-env 
myproject-mpi -h
```

2. 


a file containing various
environment variables; a shell script which has been customized to your project;
and a slurm script to be used in production, e.g.:  
  
  
