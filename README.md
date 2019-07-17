# TF Seal

TF Seal provides a bridge between [TF Encrypted](https://github.com/tf-encrypted/tf-encrypted) and the [Microsoft SEAL](https://github.com/microsoft/SEAL) homomorphic encryption library, making it easier than ever to use this library to compute on encrypted data directly from TensorFlow.

[![PyPI](https://img.shields.io/pypi/v/tf-seal.svg)](https://pypi.org/project/tf-seal/) [![CircleCI Badge](https://circleci.com/gh/dropoutlabs/tf-seal/tree/master.svg?style=svg)](https://circleci.com/gh/dropoutlabs/tf-seal/tree/master)

## Usage

*TODO*

## Installation

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/distribution/) to set up and use a Python 3.7 environment for all instructions below:

```
conda create -n tfseal python=3.7 -y
source activate tfseal
```

TF Seal can then be installed from [PyPI]():

```
pip install tf-seal
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

## Development

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/distribution/) to set up and use a Python 3.7 environment for all instructions below:

```
conda create -n tfseal-dev python=3.7 -y
source activate tfseal-dev
```

### Requirements

#### Ubuntu

*TODO*

<!--
The only requirement for Ubuntu is to have [docker installed](https://docs.docker.com/install/linux/docker-ce/ubuntu/). This is the recommended way to [build custom operations for TensorFlow](https://github.com/tensorflow/custom-op). We provide a custom development container for TF Big with all dependencies already installed.


```
wget https://storage.googleapis.com/tf-pips/tf-c++17-support/tf_nightly-1.14.0-cp37-cp37m-linux_x86_64.whl
pip install tf_nightly-1.14.0-cp37-cp37m-linux_x86_64.whl
```
-->

#### macOS

Setting up a development environment on macOS is a little more involved since we cannot use a docker container. We need the following items:

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

Note that Microsoft SEAL will be downloaded as part of the build process but the Bazel build files also include instructions for using a local version instead.

### Testing

#### Ubuntu

*TODO*

<!--
Run the tests on Ubuntu by running the `make test` command inside of a docker container. Right now, the docker container doesn't exist on docker hub yet so we must first build it:

```
docker build -t tf-encrypted/tf-big:0.1.0 .
```

Then we can run `make test`:

```
sudo docker run -it \
  -v `pwd`:/opt/my-project -w /opt/my-project \
  tf-encrypted/tf-big:0.1.0 /bin/bash -c "make test"
```
-->

#### macOS

Once the development environment is set up we can simply run:

```
make test
```

<!--
### Building Pip Package

#### macOS

```
make build
```
-->
