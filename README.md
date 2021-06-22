ase-espresso
============

[ase-espresso](https://github.com/aionics-io/ase-espresso) is an [ase](https://wiki.fysik.dtu.dk/ase/) interface for [Quantum Espresso](http://www.quantum-espresso.org/).

This version has been tuned up for internal use at Aionics, Inc.:
  - The C routines have been dropped
  - The Python modules have been packaged, and a pip-installable setup.py file has been implemented
  - The espsite.py file has been configured to run via mpiexec on the local host
  - The espresso calculator object creates a site object as an instance attribute
  - The calculator __init__ function takes a number of processors (nproc) as an argument, and passes it along to the site object, where the execution commands are formed
  - A library of pseudopotentials is being collected in the *pp* directory.

We will do our best to continue documenting our changes here.


