# scProjection

Projecting RNA measurements onto single cell atlases to extract cell type-specific expression profiles using scProjection. Refer to our paper:https://www.nature.com/articles/s41467-023-40744-6

## Tutorials

First follow the install instructions below, at the bottom of the page, before following the tutorials.

[Tutorial 1: Deconvolution of CellBench mixtures](https://github.com/quon-titative-biology/examples/blob/master/scProjection_cellbench/scProjection_cellbench.md)

[Tutorial 2: Deconvolution of spatial MERFISH data](https://github.com/quon-titative-biology/examples/blob/master/scProjection_spatial/MERFISH_deconv_example.md)

[Tutorial 3: Projection of pseudo bulk data](https://github.com/quon-titative-biology/examples/tree/master/scProjection_pseudobulk/readme.md)

[Tutorial 4: Imputation of gene expression patterns of spatial osmFISH data](https://github.com/quon-titative-biology/examples/blob/master/scProjection_imputation/readme.md)

## Install scProjection
```shell
pip3 install scProjection
```
The install time should be less than 30 min.
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
pip3 install scikit-learn
pip3 install numpy

## Install scProjection
pip3 install scProjection
```

## Updates
#### (3/16/2023) More tutorials have been added.
#### (5/23/2022) Codebase from publication made public. Need to improve user interface with method.
#### (11/9/2022) Added more tutorials with examples running scProjection in both R and Python
