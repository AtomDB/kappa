============
Introduction
============
The AtomDB Kappa model is designed to create spectra for spectra with non-Maxwellian electron distributions of electron energies. The coefficients of Hahn and Savin [1]_ are used to create similar electron distributions from a sum of Maxwellian electron distributions. 

Given the available atomic data in AtomDB and other similar databases, it is important to ensure that the Maxwellian temperatures required to model the spectrum are present in the underlying atomic spectra database. Important work on this, including exploring the validity of the approximations is in [2]_. Note that there are no warnings included in this package for when you enter into areas where the model is not particularly valid.

============
How it Works
============

The AtomDB Kappa model uses the non-equilibrium emissivity data from the AtomDB project, combined with their ionization and recombination rates (mostly from [3]_) to determine the spectrum. When a spectrum is required, the following steps are carried out:

1. The Hahn and Savin coefficients are calculated, to seee what Maxwellian temperatures (Tm) are required and with what normalizations to create the kappa distribution

2. The ionization and recombination rates are calculated at each Tm, and then summed to provide kappa ionization and recombination rates. For Tm outside the nominal 10^4 to 10^9 Kelvin range of AtomDB data, the ionization and recombination data can be extrappolated fairly safely.
3. The ionization fraction for each ion is calculated using these kappa ionization and recombination rates
4. For each ion present in the resulting ionization fraction, the spectrum is calculated at each Tm and then summed appropriately to give a final specturm. As spectra cannot easily be extrappolated, and Tm outside 10^4 to 10^9 K are ignored. Again, this can lead to unrealistic results. 


.. [1] Hahn and Savin Paper
.. [2] Cui, in prep
.. [3] Bryans 2009

=======================
Installation
=======================
Standard python installation

.. code-block:: bash

   python setup.py install

.. note::
   Kronos is a python 3 only module. Depending on your system's setup, you may need to substitute ``python3`` for all references to ``python``.

There are several useful flags that can be provided to this call, depending on your system:

  - ``--user`` flag causes installation in the user's home directory (useful if you lack root priviliges)

  - ``develop`` instead of ``install`` will install links to the current directory. This is useful if you want to edit/debug/develop the files further.

=====
Usage
=====

Mostly just make a kappa object, then call it

.. code-block:: python

  import kappa
  # make a quick, 2-elements kapp model
  a=kappa.kappamodel(elements=[6,8])
  ebins = numpy.linspace(0.01,5,1001)
  kp = 5.0
  T = 2e5
  spectrum = a.calc_spectrum(kp, T, ebins)


-----
XSPEC
-----

To use the model in XSPEC, one can ignore the class details above. Unfortunately, the code only works with the XSPEC python interface, pyxspec_ for now. Before loading the code, you will need to edit the ``kappa_xspec.py`` file to change the data file paths.

The kappa_xspec module contains all you need to load the kappa model into XSPEC. From a python3 shell:

.. code-block:: python

  # import the xspec python module
  import xspec
  # import  acx2 wrapper
  import kappa_xspec

Once this is done, the data will load.


Three different models are loaded:

- kappa : Emission from kappa spectrum with the 14 main elements. Abundance is tied between all elements (so there is only 1 abundance keyword). Analogous to the apec model.
- vkappa : Emission from kappa spectrum with the 14 main elements. Abundance is free to vary between all the elements (though it starts frozen). Analagous to the vapec model.
- vvkappa : Emission from 27 elements, H through Ni excluding Co. Abundance is free to vary between all the elements.

.. note::
  Note that in the kappa and vkappa cases, unlike in the apec and vapec models, the effective abundance of the minor recombining elements is 0, not solar. This speeds up calculation time and does not significantly affect the resulting emission.

Once you have this, models can be used in pyxspec in the usual way, e.g.

.. code-block:: python

  m = xspec.Model('tbabs(pow+kappa)')


++++++++++++++++
Model parameters
++++++++++++++++

+--------------+-----------------------------------------------------------------------------------+
| Parameter    | Definition                                                                        |
+==============+===================================================================================+
| kT           | Plasma temperature (keV). Used for recombining particle ion fraction              |
+--------------+-----------------------------------------------------------------------------------+
| kappa        | The degree of non-Maxwellianness of the electrons                                 |
+--------------+-----------------------------------------------------------------------------------+
| abund        | recombining elemental abundances. (given by individual element in vacx and vvacx) |
+--------------+-----------------------------------------------------------------------------------+


===============
Version History
===============
0.1.0
May 27th 2019
Initial release



.. _pyxspec: https://heasarc.gsfc.nasa.gov/xanadu/xspec/python/html/index.html
