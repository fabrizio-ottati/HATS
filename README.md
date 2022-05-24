# HATS: Histograms of Averaged Time Surfaces

HATS is a novel data representation for event-based computer vision, proposed by [Sironi et al.](https://arxiv.org/pdf/1803.07913.pdf). It consists in creating an histogram of time surfaces: for each event, the corresponding time surface is generated and accumulated. 

Here, a Python implementation of the algorithm is proposed. The datasets are taken from [Tonic](https://tonic.readthedocs.io/en/latest/index.html) (check it out!), in which the HATS transform source code proposed here is embedded as a transform.

Here, a couple of scripts for the dataset encoding and SVM training are proposed.
In `encode_NMNIST.py`, the dataset NMNIST is downloaded from Tonic and encoded using the HATS code in `hats.py`; the results is saved to an HDF5 file (size is ~1GB).
In `test_SVM.py`, a LinearSVM is trained on the dataset using `SGDClassifier` from `sklearn`. The resulting classification accuracy is 98.41%. 80% of the training dataset is used for training and 20% for validation, in order to choose the classifier hyperparameters. Then, the classifier is retrained on the whole dataset and tested on the test one.