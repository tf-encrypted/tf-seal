# TF Seal

TF Seal implements a bridge between Tensorflow and [Microsoft SEAL](https://github.com/microsoft/SEAL) library.

## Developer Requirements

TODO Ubuntu instructions

**MacOS**

TODO simplify this, add a bootstrap script to Makefile

Since we can't use a MacOS docker container, setting up a development environment is a little more involved. We need four things:

- Python 3.7
- Homebrew
- Bazel 0.15.0 or greater
- cmake
- [Custom Tensorflow build]()
- Microsoft SEAL

We recommend using [Anaconda](https://www.anaconda.com/distribution/) to set up a Python 3.7 environment. Once Anaconda is installed this can be done with:

```
$ conda create -n py37 python=3.7
$ source activate py37
```

We recommed using [Homebrew](https://brew.sh/) to install the next couple of dependencies. This can be installed easily with:

```
$ /usr/bin/ruby -e "$(curl -fsSL \
    https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

Bazel recommends installing with their binary installed. The documentation for this can be found [here](https://docs.bazel.build/versions/master/install-os-x.html#install-with-installer-mac-os-x). But if you have Homebrew already installed you can install bazel with a couple of simple commands:

```
$ brew tap bazelbuild/tap
$ brew install bazelbuild/tap/bazel
```

Install the custom Tensorflow:

```
pip install tf_nightly-1.14.0-cp37-cp37m-macosx_10_7_x86_64.whl
```

Install cmake:

```
brew install cmake
```

We'll have to build and install Microsoft SEAL.

Download and extract:

```
wget https://github.com/microsoft/SEAL/tree/3.3.0.tar.gz
tar -xf SEAL-3.3.0.tar.gz
```

Build and install:

```
cd SEAL-3.3.0/native/src
cmake .
make
sudo make install
```

## Building

### Tests

**MacOS**

Once the environment is set up we can simply run:

```
make test
```

### Pip Package

TODO