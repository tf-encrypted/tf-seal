# TF Seal

TF Seal implements a bridge between Tensorflow and [Microsoft SEAL](https://github.com/microsoft/SEAL) library.

## Developer Requirements

**Ubuntu**

TODO simplify this, add a bootstrap script to Makefile

We need to install the following software to be able to use TF SEAL.

- Python 3.7
- Bazel 0.15.0 or greater
- cmake
- [Custom Tensorflow build](https://storage.googleapis.com/tf-pips/tf-c%2B%2B17-support/tf_nightly-1.14.0-cp37-cp37m-linux_x86_64.whl)
- Microsoft SEAL

We recommend using [Anaconda](https://www.anaconda.com/distribution/) to set up a Python 3.7 environment. Once Anaconda is installed this can be done with:

```
$ conda create -n py37 python=3.7
$ conda activate py37
```


**MacOS**

TODO simplify this, add a bootstrap script to Makefile

We need to install the following software to be able to use TF SEAL.

- Python 3.7
- Homebrew
- Bazel 0.15.0 or greater
- cmake
- [Custom Tensorflow build](https://storage.googleapis.com/tf-pips/tf-c%2B%2B17-support/tf_nightly-1.14.0-cp37-cp37m-macosx_10_7_x86_64.whl)
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
wget https://storage.googleapis.com/tf-pips/tf-c%2B%2B17-support/tf_nightly-1.14.0-cp37-cp37m-macosx_10_7_x86_64.whl
pip install tf_nightly-1.14.0-cp37-cp37m-macosx_10_7_x86_64.whl
```

Install cmake:

```
brew install cmake
```

We'll have to build and install Microsoft SEAL.

Download and extract:

```
wget https://github.com/microsoft/SEAL/tree/3.3.1.tar.gz
tar -xf SEAL-3.3.1.tar.gz
```

Build and install:

```
cd SEAL-3.3.1/native/src
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