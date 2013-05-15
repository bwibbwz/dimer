================
The Dimer method
================

.. module:: dimer
   :synopsis: The Dimer method.

.. default-role:: math

The Dimer method is an algorithm that can be used for finding saddle points, starting from any given atomic configuration.
Saddle points often represent the energy barrier for transitioning from one state to another.

This is accomplished in two indepedant steps.

#. Estimating the eigenmode corresponding to the lowest eigenvalue of the Hessian matrix of the total energy with regards to the atomic coordinates.

#. Inverting the forces along this eigenmode and translating the system accordingly.

In effect this transforms saddle points, locally, into minima which can then be found using traditional methods for structure optimization (:mod:`optimize`).

There are four classes that combine to perform the above steps:

- :class:`ase.dimer.DimerAtoms` (alias for :epydoc:`ase.dimer.MinModeAtoms`)
    A wrapper for :class:`~ase.atoms.Atoms` containing extra information needed for the Dimer method, such as the current eigenmode estimate, the curvature along that estimate and the modified force.

- :class:`ase.dimer.DimerControl` (:epydoc:`ase.dimer.DimerControl`)
    Takes care of the parameters needed for a Dimer calculation. If a given parameter is not set, a default value is used.

- :class:`ase.dimer.DimerEigenmodeSearch` (:epydoc:`ase.dimer.DimerEigenmodeSearch`)
    Takes care of finding the eigenmode corresponding to the lowest eigenvalue of the Hessian matrix.
    This class is not used directly by the general user.

- :class:`ase.dimer.DimerTranslate` (alias for :epydoc:`ase.dimer.MinModeTranslate`)
    An optimizer class specifically tailored for use within the Dimer method.

Calculations
------------

A simple example of use, starting from a trajectory: initial.traj::

  # Import the relevant classes and methods
  from ase.dimer import DimerControl, DimerAtoms, DimerTranslate
  from ase.io import read
  # Read in the initial position of the atoms (often an energy minimum)
  atoms = read('initial.traj')
  # Set up the parameters for the dimer search
  control = DimerControl(logfile = 'dimer.dimlog', eigenmode_logfile = 'dimer.mlog')
  # Set up the DimerAtoms wrapper for the Atoms object
  d_atoms = DimerAtoms(atoms, control)
  # Randomly displace the atoms (away from the minimum)
  d_atoms.displace()
  # Set up the optimizer
  opt = DimerTranslate(d_atoms, logfile = 'dimer.optlog', trajectory = 'dimer.traj')
  # Optimize the system
  opt.run(fmax = 0.01)

Restarting from a trajectory, old.traj, and eigenmode estimate, old.mlog::

  # Import the relevant classes and methods
  from ase.dimer import DimerControl, DimerAtoms, DimerTranslate, read_eigenmode
  from ase.io import read
  # Read in the last position of the atoms
  atoms = read('old.traj')
  # Set up the parameters for the dimer search
  control = DimerControl(logfile = 'contd.dimlog', eigenmode_logfile = 'contd.mlog')
  # Read the last eigenmode
  eigenmode = read_eigenmode('old.mlog', index = -1)
  # Set up the DimerAtoms wrapper for the Atoms object and initialize the eigenmode.
  d_atoms = DimerAtoms(atoms, control, eigenmodes = [eigenmode])
  # Set up the optimizer
  opt = DimerTranslate(d_atoms, logfile = 'dimer.optlog', trajectory = 'dimer.traj')
  # Optimize the system
  opt.run(fmax = 0.01)

Besides setting the convergence parameters in :class:`ase.dimer.DimerControl`, deciding which atoms are displaced (:meth:`ase.dimer.MinModeAtoms.displace`) and by how much is the most important feature of the Dimer method.
The parameters can be set in the :class:`DimerControl` object and/or the displace method.

Changing control parameters 4 rotations per translate step and a dimer separation of 0.01Å and displacement, the 4 closest atoms to (0.0, 0.0, 0.0)::

  # Import the relevant classes and methods
  from ase.dimer import DimerControl, DimerAtoms, DimerTranslate
  from ase.io import read
  # Read in the initial position of the atoms
  atoms = read('initial.traj')
  # Set up the parameters for the dimer search
  control = DimerControl(max_num_rot = 4, dimer_separation = 0.01)
  # Read the last eigenmode
  eigenmode = read_eigenmode('old.mlog', index = -1)
  # Set up the DimerAtoms wrapper for the Atoms object and initialize the eigenmode.
  d_atoms = DimerAtoms(atoms, control, eigenmodes = [eigenmode])
  # Randomly displace the atoms (away from the minimum)
  d_atoms.displace(displacement_center = (0.0, 0.0, 0.0), number_of_atoms = 4)
  # Set up the optimizer
  opt = DimerTranslate(d_atoms, logfile = 'dimer.optlog', trajectory = 'dimer.traj')
  # Optimize the system
  opt.run(fmax = 0.01)

DFT
===

The default settings for the Dimer method are optimized for use with calculators without numerical noise.
When used with DFT calculators, the dimer_separation parameter should be increased as to minimize the effect of numerical noise.

Symmetries
==========

The Dimer method is designed to locate non-equilibrium structures.
Due to this using symmetries to reduce calculational efforts must be used with care.
As an example, when using GPAW_, ``calc.set(usesymm = False)``, will avoid the breaking of k-point symmetries when far away from equilibrium structures (usesymm_).

.. _GPAW: http://wiki.fysik.dtu.dk/gpaw
.. _usesymm: https://wiki.fysik.dtu.dk/gpaw/documentation/manual.html#manual-usesymm

Relevant literature references
------------------------------

#. 'A dimer method for ﬁnding saddle points on high on high dimensional potential surfaces using only ﬁrst derivatives.', G. Henkelman and H. Jonsson, The Journal of Chemical Physics, 111(15):7010–7022, 1999. [Original formulation]

#. 'Comparison of methods for ﬁnding saddle points without knowledge of the ﬁnal states', R. A. Olsen, G. J. Kroes, G. Henkelman, A. Arnaldsson, and H. Jonsson, The Journal of Chemical Physics, 121(20):9776–9792, 2004. [...]

List of the relevant Classes and Methods
----------------------------------------

.. autoclass:: ase.dimer.DimerAtoms
.. autoclass:: ase.dimer.MinModeAtoms
   :members:

------------

.. autoclass:: ase.dimer.DimerTranslate
.. autoclass:: ase.dimer.MinModeTranslate
   :members:

------------

.. autoclass:: ase.dimer.DimerControl
   :members:

