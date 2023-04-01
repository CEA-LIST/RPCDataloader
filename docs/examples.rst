Examples
========

.. _ImageNet training example:

ImageNet training example
-------------------------

This example uses rpcdataloader for the training of a ResNet50 on ImageNet.
It supports for distributed training and mixed-precision.
Modifications to make use of rpcdataloader are highlighted and evaluation routines are ommitted for readability.

Prior to running this script, you should :ref:`spawn workers <Usage>`.

.. literalinclude:: example_rpc.py
    :linenos:
    :emphasize-lines: 13,20,48-50,71-73,84


.. _Slurm integration example:

Slurm integration example
-------------------------

To use rpcdataloader on a `Slurm <https://slurm.schedmd.com/>`_ cluster, the `heterogeneous jobs <https://slurm.schedmd.com/heterogeneous_jobs.html>`_ functionality will let you reserve two groups of resources: one for workers and one for training scripts.
The sample script below demonstrates how to do this.

Note that you might need to adjust port numbers to avoid collisions between jobs.
You might also need to adjust resource specifications depending on the slurm configuration.

.. literalinclude:: example_rpc.slurm
    :linenos:
    :language: shell
