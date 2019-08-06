# TF SEAL

TF SEAL provides a bridge between [TensorFlow](https://tensorflow.org) and the [Microsoft SEAL](https://github.com/microsoft/SEAL) homomorphic encryption library, making it easier than ever to use this library to compute on encrypted data. It currently offers low-level operations for interacting with Microsoft SEAL via TensorFlow with [work in progress](#road-map) on a high-level integration into [TF Encrypted](https://tf-encrypted.io).

[![PyPI](https://img.shields.io/pypi/v/tf-seal.svg)](https://pypi.org/project/tf-seal/) [![CircleCI Badge](https://circleci.com/gh/tf-encrypted/tf-seal/tree/master.svg?style=svg)](https://circleci.com/gh/tf-encrypted/tf-seal/tree/master)

## Usage

The following demonstrates how the low-level interface can be used to perform a matrix multiplication using homomorphic encryption inside of TensorFlow.

```python
import numpy as np
import tensorflow as tf
import tf_seal as tfs

public_keys, secret_key = tfs.seal_key_gen(gen_relin=True, gen_galois=True)

# sample inputs in the form of tf.Tensors
a = tf.random.normal(shape=(2, 3), dtype=tf.float32)
b = tf.random.normal(shape=(2, 3), dtype=tf.float32)

# the plaintext equivalent of our computation
c = tf.matmul(a, tf.transpose(b))

# encrypt inputs, yielding tf_seal.Tensors
a_encrypted = tfs.convert_to_tensor(a, secret_key, public_keys)
b_encrypted = tfs.convert_to_tensor(b, secret_key, public_keys)

# perform computation on encrypted data
# - note that because of how the data is laid out in memory,
#   tf_seal.matmul expects the right-hand matrix to be ordered
#   column-major wise, i.e. already transposed
c_encrypted = tfs.matmul(a_encrypted, b_encrypted)

with tf.Session() as sess:
    expected, actual = sess.run([c, c_encrypted])
    np.testing.assert_almost_equal(actual, expected, decimal=3)
```

## Road map

We are currently working on integrating TF SEAL into [TF Encrypted](https://tf-encrypted.io) such that privacy-preversing machine learning applications can instead access the library through a high-level interface and take advantage of e.g. the Keras API. This includes adding logic that helps optimize homomorphic encryption for a perticular computation, making use even easier.

<img src="https://raw.githubusercontent.com/tf-encrypted/assets/master/tf-seal/app-stack.png" width="45%" />

## Installation

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/distribution/) to set up and use a Python 3.7 environment for all instructions below:

```
conda create -n tfseal python=3.7 -y
source activate tfseal
```

After installing the [custom build of TensorFlow](#custom-tensorflow) you can install [TF SEAL from PyPI](https://pypi.org/project/tf-seal/) using pip:

```
pip install tf-seal
```

## Examples

There is currently one example displaying how we can run a simple logistic regression prediction with TF SEAL.

Once TF SEAL is installed we can run the example by simplying running:

```
python logistic_regression.py
```

## Development

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/distribution/) to set up and use a Python 3.7 environment for all instructions below:

```
conda create -n tfseal-dev python=3.7 -y
source activate tfseal-dev
```

The basic requirements are:

- Python (== 3.7)
- [Bazel](https://docs.bazel.build/versions/master/install.html) (>= 0.26.1)
- CMake
- [TensorFlow built with C++17](#custom-tensorflow)

The remaining PyPI packages can then be installed using:

```
pip install -r requirements-dev.txt
```

### Testing

All tests can be run via Bazel with:

```
make test
```

### Building

The pip package can be built using:

```
make build
```

with the resulting wheel file placed in `./artifacts`.

## Custom TensorFlow

A custom build of TensorFlow is currently needed to run TF SEAL due to a mismatch between the C++ version used by the official TensorFlow build (C++11) and the one needed by Microsoft SEAL (C++17). A [patched version of TensorFlow](https://github.com/dropoutlabs/tensorflow) built with C++17 can be installed as shown below.

#### Ubuntu binary

```
wget https://storage.googleapis.com/tf-pips/tf-c++17-support/tf_nightly-1.14.0-cp37-cp37m-linux_x86_64.whl
pip install tf_nightly-1.14.0-cp37-cp37m-linux_x86_64.whl
```

#### macOS binary

```
wget https://storage.googleapis.com/tf-pips/tf-c++17-support/tf_nightly-1.14.0-cp37-cp37m-macosx_10_7_x86_64.whl
pip install tf_nightly-1.14.0-cp37-cp37m-macosx_10_7_x86_64.whl
```

#### From source

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/distribution/) to first set up and use a Python 3.7 environment:

```
conda create -n customtf python=3.7 -y
source activate customtf
```

This requires that [Bazel](https://docs.bazel.build/versions/master/install.html) (== 0.26.1) has been installed. The patched version of TensorFlow may then be built using:

```
git clone https://github.com/tf-encrypted/tf-seal.git
cd tf-seal
pip install -r requirements-customtf.txt

make tensorflow
pip install -U tf_nightly-1.14.0-cp37-cp37m-*
```
