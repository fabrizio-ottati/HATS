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

In `test_SVM.py`, a LinearSVM is trained on the dataset using `SGDClassifier` from `sklearn`. 80% of the training dataset is used for training and 20% for validation, in order to choose the classifier hyperparameters. Then, the classifier is retrained on the whole dataset and tested on the test one.

## Some details on the classifier

The linear SVM is implemented using `sklearn.linear__model.SGDClassifier` with the _hinge_ loss. I did not use `sklearn.SVM.LinearSVC` because it does not allow to take advantage of multi-core processing and it does not allow batch-based learning in case of large datasets. 

I use an L2 normalization (i.e. `sklearn.preprocessing.Normalizer`) on the data for the classifier, since it leads to faster training convergence and better classification accuracy. I do not know why conventional zero-mean normalization (i.e. `sklearn.preprocessing.StandardScaler`) does not work properly. This may be due to the fact the an histogram is in the range _[0, something]_. Performance using zero-mean normalization is reported for clarity. 

If you want to play with the code, notice that `fit_intercept` is set to `False` for L2 normalization, as it allows for faster convergence and slightly improved accuracy, while it has to be set to `True` for zero-mean normalization. 

## Performance 

| Normalization               | Dataset      | Tuned SVM C | Accuracy |
|-----------------------------|--------------|-------------|----------|
| Zero-mean  (StandardScaler) | NMNIST       | 10^7        | 98.13%   |
|                             | CIFAR10DVS   | 10^6        | 43.15%   |
|                             | NCARS        | 10^8        | 86.65%   |
|                             | NCALTECH101  |             |          |
| L2 norm  (Normalizer)       | NMNIST       | 10^8        | 98.34%   |
|                             | CIFAR10DVS   | 10^6        | 45.1%    |
|                             | NCARS        | 10^5        | 87.20%   |
|                             | NCALTECH101  |             |          |

## Usage

To run the code on NMNIST, write the following in the terminal. 
```bash
$ cd NMNIST
$ python encode_NMNIST.py "./NMNIST.hdf5"
$ cd ..
$ python test_SVM.py ./NMNIST/NMNIST.hdf5 --c-min 1e4 --c-min 1e8
```

Minimum and maximum C are set to `1e4` and `1e8` by default in the code.

## Requirements

```python
tonic
scikit-learn
numpy
tqdm # Optional but the progress bars are nice :)
h5py
```

## Possible improvements and contributions

I did not add the NCARS download code because the dataset is not available on Tonic right now. I packaged it my self but the HDF5 file is too large for GitHub. As soon as we will be able to add NCARS to Tonic, I will upload the corresponding code here. If you need the dataset, get in touch with me and I will send you the code to download it and package it.

I would love to add also NCALTECH101 to fully re-implement the HATS paper results, but I am lacking free time :) Any contribution would be highly appreciated!

I would also really appreciate if someone could help me reach the accuracies shown in the original article (i.e. 99.2% for NMNIST and 52% for CIFAR10DVS).

In the future, I will probably add some documentation on time surfaces.
