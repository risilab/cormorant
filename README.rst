========
Overview
========

This is the home of the cormorant software package for learning on atomic environments.


Documentation
=============

To install the documentation, go to the docs folder and run "make html".  You will need to install the sphinx_rtd_theme (this can be done using pip install).

Getting started
===============

Installation
------------

Cloning the git repo
`````````````````````

Cormorant can be cloned directly from the git repo using::

    TO BE IMPLEMENTED!!!

Using pip
`````````

Cormorant is installable using pip.  You can currently install it from
source by going to the directory with setup.py::

    pip install cormorant .

If you would like to modify the source code directly, note that Cormorant
can also be installed in "development mode" using the command::

    pip install cormorant -e .


Training example
----------------

The example training script is in :examples/train_cormorant.py:. The same script
can train both the datasets QM9 and MD17, and can also be extended to more general datasets.
::

    python examples/train_cormorant.py --dataset=qm9

::

    python examples/train_cormorant.py --dataset=md17

Note that if no GPU is available, the the training script will throw an error.
To force CPU-based training, add the : --cpu: flag

..
  ================
  Architecture
  ================

  A more detailed description of the Cormorant architecture is available here.
