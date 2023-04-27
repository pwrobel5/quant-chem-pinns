# quant-chem-pinns

Needs CUDA 11.8, for 12.1 there are problems with Tensorflow

Add TensorRT path to LD_LIBRARY_PATH (e.g. venv/lib/python3.10/site-packages/tensorrt) as well as the path to .so files from CUDA and CuDNN

Use tensorflow.compat.v1 backend (defined in ~/.deepxde/config.json), as tensorflow in version 2+ has some problems with 2D functions.
