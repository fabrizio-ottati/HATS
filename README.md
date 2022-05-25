# HATS: Histograms of Averaged Time Surfaces

HATS is a novel data representation for event-based computer vision, proposed by [Sironi et al.](https://arxiv.org/pdf/1803.07913.pdf). It consists in creating an histogram of time surfaces: for each event, the corresponding time surface is generated and accumulated. More on this later :)

Here, a Python implementation of the algorithm is proposed. The DVS datasets are taken from [Tonic](https://tonic.readthedocs.io/en/latest/index.html) (check it out!), in which the HATS transform source code proposed here is embedded as a transform.

## What is a DVS camera?

Well, take a look to [this introduction](https://tonic.readthedocs.io/en/latest/getting_started/intro-event-cameras.html) and [this tutorial](https://tonic.readthedocs.io/en/latest/tutorials/nmnist.html) from Tonic documentation.

## What is a time surface?

More on this in the future :)

## The code

The code consists of a couple of scripts for the dataset encoding and SVM training. The tested datasets are CIFAR10DVS and NMNIST.

In `encode_NMNIST.py`, the dataset NMNIST is downloaded from Tonic and encoded using the HATS code in `hats.py`; the results is saved to an HDF5 file (size is ~1GB), in order to play with the SVM without having to wait for the HATS encoding to be performed on the whole dataset. The HATS parameters are taken from the [supplementary material](https://openaccess.thecvf.com/content_cvpr_2018/Supplemental/1083-supp.pdf) of the HATS paper.

In `test_SVM.py`, a LinearSVM is trained on the dataset using `SGDClassifier` from `sklearn`. 80% of the training dataset is used for training and 20% for validation, in order to choose the classifier hyperparameters. Then, the classifier is retrained on the whole dataset and tested on the test one. The resulting classification accuracies on NMNIST and CIFAR10DVS are 98.41% and 46.41%, respectively.

## Usage

To run the code on NMNIST, write the following in the terminal. 
```bash
$ cd NMNIST
$ python encode_NMNIST.py "./NMNIST.hdf5"
$ cd ..
$ python test_SVM.py "./NMNIST/NMNIST.hdf5"
```

The same applies for CIFAR10DVS.

## Requirements

```python
tonic
scikit-learn
numpy
tqdm # Optional but the progress bars are nice :)
h5py
```

## Possible improvements and contributions

I would love to add NCARS and NCALTECH101 to fully re-implement the HATS paper results, but I am lacking free time :) Any contribution would be highly appreciated!

I would also really appreciate if someone could help me reach the accuracies shown in the original article (i.e. 99.2% for NMNIST and 52% for CIFAR10DVS).

In the future, I will probably add some documentation on time surfaces.
