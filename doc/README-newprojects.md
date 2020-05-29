New Projects
============

This README briefly describes how the *legacyhalos* code base can be used on new
projects.  Every project different, so using this repository will require some
customized code.

In this example let's create a project called 'myproject':

1. Clone the *legacyhalos* repository and set an environment variable pointing
to your local checkout:

  export LEGACYHALOS_CODE_DIR=/path/to/legacyhalos

2. Create a new directory in '$LEGACYHALOS_CODE_DIR/bin' with the name of your
project, e.g.:

  mkdir $LEGACYHALOS_CODE_DIR/bin/myproject

In this directory we'll need three files: a file containing various environment
variables; a shell script which has been customized to your project; and a slurm
script to be used in production, e.g.:

  $LEGACYHALOS_CODE_DIR/bin/myproject/myproject-mpi.sh
  $LEGACYHALOS_CODE_DIR/bin/myproject/myproject-mpi.env
  $LEGACYHALOS_CODE_DIR/bin/myproject/myproject-mpi.slurm

[Note: The '-mpi' suffix is historical and not needed.]

3. Create a module in '$LEGACYHALOS_CODE_DIR/py/legacyhalos' which will contain
the custom code for your project, e.g.:

  $LEGACYHALOS_CODE_DIR/py/legacyhalos/myproject.py

This module can contain whatever you want, but at minimum it will need to
contain functions to:

  - specify the command-line arguments for your wrapper script;
  - read the sample;
  - find missing/completed files;
  - generate your webpages

Once all these pieces are in place you can process the sample three different
ways, as described in more detail below: (1) interactively with a single node;
(2) interactively with many nodes; or (3) using Slurm with one or many nodes.

1. To run the code interactively with a single node from a clean login, do:

  salloc -N 1 -C haswell -A desi -t 04:00:00 --qos interactive -L SCRATCH,cfs
  
