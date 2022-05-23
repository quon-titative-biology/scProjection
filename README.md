# scProjection

Projection and Deconvolution using deep hierarchical and generative neural network. Refer to our preprint: https://www.biorxiv.org/content/10.1101/2022.04.26.489628v1

## Tutorials

First follow the install instructions below, at the bottom of the page, before following the tutorials.

[CellBench](https://github.com/quon-titative-biology/examples/blob/master/scProjection_cellbench/scProjection_cellbench.md)


## Install scProjection
```shell
pip3 install scProjection
```

## Package requirements

scProjection requires: Python 3. This is a guide to installing python on different operating systems.

### (Python)
  #### All platforms:
  1. [Download install binaries for Python 3 here](https://www.python.org/downloads/release/)
  #### Alternative (On Windows):
  1. Download Python 3
  2. Make sure pip is included in the installation.

  #### Alternative (On Ubuntu):
  1. sudo apt update
  2. sudo apt install python3-dev python3-pip

  #### Alternative (On MacOS):
  1. /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
  2. export PATH="/usr/local/bin:/usr/local/sbin:$PATH"
  3. brew update
  4. brew install python  # Python 3

## Setup of virtualenv

scProjection also requires: tensorflow, tensorflow-probability, sklearn and numpy. It is generally easier to setup the dependencies using a virtual environment which can be done as follows:

```shell
## Create the virtual environment
virtualenv -p python3 pyvTf2

## Launch the virtual environment
source ./pyvTf2/bin/activate

## Setup dependencies
pip3 install tensorflow
pip3 install tensorflow-probability
pip3 install sklearn
pip3 install numpy

## Install scProjection
pip3 install scProjection
```

## Updates

#### (5/23/2022) Codebase from publication made public. Need to improve user interface with method.
