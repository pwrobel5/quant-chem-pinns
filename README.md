# Quant Chem PINNs

This repository contains source codes prepared for Master Thesis done at the AGH UST in Cracow.
The code is under refactoring process now.

Topic of the Thesis: Neural networks in solving differential equations

Text of the Thesis is available [here](msc-thesis.pdf)

Two approaches were used:
* Physics Informed deep Neural Networks (PINNs) for solving Schrödinger equations for model problems in quantum chemistry, used library: [DeepXDE](https://github.com/lululxvi/deepxde)
* Acceleration of DFTB based molecular dynamics (MD) simulation by prediction of atomic charges by SchNet, used library: [Schnetpack](https://github.com/atomistic-machine-learning/schnetpack)

<!---
Needs CUDA 11.8, for 12.1 there are problems with Tensorflow

Add TensorRT path to LD_LIBRARY_PATH (e.g. venv/lib/python3.10/site-packages/tensorrt) as well as the path to .so files from CUDA and CuDNN

Use tensorflow.compat.v1 backend (defined in ~/.deepxde/config.json), as tensorflow in version 2+ has some problems with 2D functions.

For DFTB+ API it is neccessary to have properly set LD_LIBRARY_PATH (to point to dftb+/lib directory)
It threw some problems with not found GOMP_5.0 version, replacement of pointed file by the library libgomp.so from the system fixed everything
-->
