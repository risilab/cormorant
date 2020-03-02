==========
IMPORTANT!
==========
This code is intended for reference only while licensing issues are sorted out.  For now, it may not be used, copied, or distributed without written permission of the authors.  Check back soon for a version with the final license. 

Also, it is important to note that cormorant will have issues with Python<=3.6.  

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

    git clone https://github.com/risilab/cormorant.git

Using pip
`````````

Cormorant is installable from source using pip.  You can currently install it from
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

    python examples/train_qm9.py

::

    python examples/train_md17.py

Note that if no GPU is available, the the training script will throw an error.
To force CPU-based training, add the : --cpu: flag

================
Architecture
================

A more detailed description of the Cormorant architecture is available in `the Cormorant paper <https://arxiv.org/abs/1906.04015>`_.
