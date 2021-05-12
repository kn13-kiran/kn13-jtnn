# Junction Tree Variational Autoencoder for Molecular Graph Generation

Partial implementation (molecule generation only) for Junction Tree Variational Autoencoder [https://arxiv.org/abs/1802.04364](https://arxiv.org/abs/1802.04364). Some of the reference code is borrowed from https://github.com/wengong-jin/icml18-jtnn


# Requirements
* Mac OS Cataliina (I've tested this on Mac)
* RDKit (version >= 2017.09)
* Python (version == 2.7)
* PyTorch (version >= 0.2)

To install RDKit, please follow the instructions here [http://www.rdkit.org/docs/Install.html](http://www.rdkit.org/docs/Install.html)

# Quick Start
The following directories contains the most up-to-date implementations of our model:
* `model/` contains code for model implementation.
* `train/` contains code for  training. Please refer to `train/README.md` for details.

The following directories provides scripts for the experiments in our original ICML paper:

