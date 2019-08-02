# TF Seal

TF Seal provides a bridge between [TF Encrypted](https://github.com/tf-encrypted/tf-encrypted) and the [Microsoft SEAL](https://github.com/microsoft/SEAL) homomorphic encryption library, making it easier than ever to use this library to compute on encrypted data directly from TensorFlow.

[![PyPI](https://img.shields.io/pypi/v/tf-seal.svg)](https://pypi.org/project/tf-seal/) [![CircleCI Badge](https://circleci.com/gh/dropoutlabs/tf-seal/tree/master.svg?style=svg)](https://circleci.com/gh/dropoutlabs/tf-seal/tree/master)

## Usage

The following demonstrates how to perform a matrix multiplication using homomorphic encryption inside of TensorFlow.

```
import numpy as np
import tensorflow as tf

import tf_seal as tfs

public_keys, secret_key = tfs.seal_key_gen(gen_relin=True, gen_galois=True)

# encrypted input -> tf_seal.Tensor
a_plain = np.random.normal(size=(2, 2)).astype(np.float32)
a = tfs.constant(a_plain, secret_key, public_keys)

# public weights
b = np.random.normal(size=(2, 2)).astype(np.float32)

# because of how the data is laid out in memory tfs.matmul expects
# the b matrix to be order column-major wise
c = tfs.matmul(a, b.transpose())

with tf.Session() as sess:
    print(sess.run(c))
```

## Installation

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/distribution/) to set up and use a Python 3.7 environment for all instructions below:

```
conda create -n tfseal python=3.7 -y
source activate tfseal
```

### Custom TensorFlow

A custom build of TensorFlow is currently needed to run TF Seal due to a mismatch between the C++ version used by the official TensorFlow build (C++11) and the one needed by Microsoft SEAL (C++17). A [patched version of TensorFlow](https://github.com/dropoutlabs/tensorflow) built with C++17 can be installed as shown below.

#### Ubuntu

```
wget https://storage.googleapis.com/tf-pips/tf-c++17-support/tf_nightly-1.14.0-cp37-cp37m-linux_x86_64.whl
pip install tf_nightly-1.14.0-cp37-cp37m-linux_x86_64.whl
```

#### macOS

```
wget https://storage.googleapis.com/tf-pips/tf-c++17-support/tf_nightly-1.14.0-cp37-cp37m-macosx_10_7_x86_64.whl
pip install tf_nightly-1.14.0-cp37-cp37m-macosx_10_7_x86_64.whl
```

After installing the custom build of TensorFlow you can install TF Seal from [PyPi]() using pip:

```
pip install tf-seal
```

## Development

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/distribution/) to set up and use a Python 3.7 environment for all instructions below:

```
conda create -n tfseal-dev python=3.7 -y
source activate tfseal-dev
```

### Requirements

#### Ubuntu

- Python (== 3.7)
- [Bazel](https://www.bazel.build/) (>= 0.26.1)
- CMake
- [TensorFlow built with C++17](#custom-tensorflow)

CMake can be installed simply with apt:

```
sudo apt install cmake
```

Bazel is a little more involved, the following instructions can be installed, recommend installing Bazel 0.26.1: https://docs.bazel.build/versions/master/install-ubuntu.html#install-with-installer-ubuntu

The remaining PyPI packages can then be installed using:

```
pip install -r requirements-dev.txt
```

Once the custom TensorFlow is installed you will be able to start development.

#### macOS

We need the following items:

- Python (== 3.7)
- [Bazel](https://www.bazel.build/) (>= 0.26.1)
- CMake
- [TensorFlow built with C++17](#custom-tensorflow)

Using [Homebrew](https://brew.sh/) we make sure that both [Bazel](https://docs.bazel.build/versions/master/install-os-x.html#install-with-installer-mac-os-x) and CMake are installed:

```
brew tap bazelbuild/tap
brew install bazelbuild/tap/bazel
brew install cmake
```

The remaining PyPI packages can then be installed using:

```
pip install -r requirements-dev.txt
```

Once the custom TensorFlow is installed you will be able to start development.

### Testing

Once the development environment is set up you can run:

```
make test
```

