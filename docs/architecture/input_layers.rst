.. _InputLayers:
=============
Input Layers
=============

The role of the input layers is to take an initial feature embedding for atoms
and/or pairs of atoms, and process the features in a way that does not involve
Clebsch-Gordan products. For the example network architectures in package,
only scalar inputs are included.

As discussed in the `Cormorant paper <https://arxiv.org/abs/1906.04015>`_,
we found the type input layers that worked best depended on the dataset
used. Here we present two examples: a :ref:`Linear mixing layer` for MD-17, and
a :ref:`Message passing neural network` for QM-9.

Linear mixing layer
-------------------

The dataset MD-17 needs only minimal mixing to prepare the atoms.
Each atom's identity is embeded using a one-hot encoding. This encoding
is then mixed to a pre-defined number of channels with a `torch.nn.Linear`
module.

????? REFERENCE nn.InputLinear

Message passing neural network
------------------------------

The dataset QM9 needs a more advanced featurization. We used a basic Message
Passing Neural Network (MPNN) to generate an improved representation of each
atom's local environment before passing the representation to the CG layers.

The MPNN does not use an edge network. Edges are constructed entirely using
masked position functions as defined in ????? (nn.RadialFilters).

????? REFERENCE nn.InputMPNN
