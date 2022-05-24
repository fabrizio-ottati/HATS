# HATS: Histograms of Averaged Time Surfaces

HATS is a novel data representation for event-based computer vision, proposed by [Sironi et al.](https://arxiv.org/pdf/1803.07913.pdf). It consists in creating an histogram of time surfaces: for each event, the corresponding time surface is generated and accumulated. More on this later :)

Here, a Python implementation of the algorithm is proposed. The DVS dataset is taken from [Tonic](https://tonic.readthedocs.io/en/latest/index.html) (check it out!), in which the HATS transform source code proposed here is embedded as a transform.

## The code

The code consists of a couple of scripts for the dataset encoding and SVM training.

In `encode_NMNIST.py`, the dataset NMNIST is downloaded from Tonic and encoded using the HATS code in `hats.py`; the results is saved to an HDF5 file (size is ~1GB). The HATS parameters are taken from the [supplementary material](https://openaccess.thecvf.com/content_cvpr_2018/Supplemental/1083-supp.pdf) of the HATS paper.

In `test_SVM.py`, a LinearSVM is trained on the dataset using `SGDClassifier` from `sklearn`. 80% of the training dataset is used for training and 20% for validation, in order to choose the classifier hyperparameters. Then, the classifier is retrained on the whole dataset and tested on the test one. The resulting classification accuracy is 98.41%.

## Usage

To run the code on NMNIST, write the following in the terminal. 
```bash
$ python encode_NMNIST.py
$ python test_SVM.py
```

## Requirements

```python
tonic
scikit-learn
numpy
tqdm # Optional but the progress bars are nice :)
h5py
```
