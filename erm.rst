===============================
The Energy Ridge Mapping method
===============================

.. module:: erm
   :synopsis: Energy Ridge Mapping method.

.. default-role:: math

The Energy Ridge Mapping (ERM) method is an algorithm used for finding energy ridges between neighboring first order saddle points and the highest points along these paths, the second order saddle points.
This is accomplished by first inverting any force components along minimum curvature eigenmode (using the :mod:`dimer` method) and then using placing a series of images, with controlled spacing (using the :mod:`neb` method).

Two classes are required to calculate a ridge.

- :class:`ase.erm.ERM` (:epydoc:`ase.erm.ERM`)
    MISSING

- :class:`ase.dimer.DimerControl` (:epydoc:`ase.dimer.DimerControl`)
    Takes care of the parameters needed for the minimum mode (dimer) calculation. If a given parameter is not set a default value is used. Only a sinlge DimerControl object is needed as the ERM class will create replicas with incremented numbers for the relevant logfiles.

Calculations
------------

Example of use, between initial and final state which have been previously saved in A.traj and B.traj:
  
  # Load the relevant modules
  from ase.io import read
  from ase.calculators.emt import EMT
  from ase.erm import ERM
  from ase.dimer import DimerControl
  # Set up the logfiles (and other dimer parameters). The final logfile names 
  # will be erm-X.Ylog, where X is the number of the image and Y is either 
  # 'dim' (short for dimer) or 'm' (short for minimum mode)
  control = DimerControl(logfile = 'erm.dimlog', eigenmode_logfile = 'erm.mlog')
  # Load the endpoints. They should be first order saddle points.
  initial = read('A.traj')
  final = read('B.traj')
  # Populate the images (initial and final are Atoms objects).
  images = [initial.copy()]
  for k in range(n_img - 2):
      images += [initial.copy()]
  images += [final.copy()]
  # Set up the ERM object
  erm = ERM(images, control)
  # Set the initial configuration of the images as a linear interpolation between
  # the endpoints
  erm.interpolate()
  # Attach a calculator to each image
  for img in images:
      img.set_calculator(EMT())
  # Optimize the ERM object
  opt = FIRE(erm)
  opt.run(fmax = 0.01)

Relevant literature references
------------------------------

#. 'A method for finding the ridge between saddle points applied to rare event rate estimates', J. B. Maronsson, H. Jónsson, T. Vegge, PCCP, 14, 2884 (2012)

