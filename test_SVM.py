#! /usr/bin/env python3

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
import tonic
import numpy as np
import h5py
from tqdm import tqdm

# Generation of a parametric SVM.
gen_SVM = lambda C, seed: Pipeline(
    [
        ("scaler", Normalizer(norm="l2")),
        (
            "SVC",
            SGDClassifier(
                loss="hinge",
                alpha=1 / C,
                fit_intercept=False,
                max_iter=1e6,
                tol=1e-5,
                n_jobs=-1,
                random_state=seed
            ),
        ),
    ]
)

# Flattening of the histogram for the SVM.
flat_hist = lambda hist: hist.reshape((hist.shape[0], np.prod(hist.shape[1:])))

f = h5py.File("NMNIST.hdf5", "r")
train, test = f["train"], f["test"]
ds = tonic.datasets.NMNIST(save_to="./data_tr", train=True)
SEED, VAL_SIZE = 32, 0.2
tr_idxs, val_idxs, _, _ = train_test_split(
  range(len(ds)),
  ds.targets,
  stratify=ds.targets,
  test_size=VAL_SIZE,
  random_state=SEED
)


# For HDF5 arrays, the indices need to be sorted.
tr_idxs.sort()
val_idxs.sort()
print("="*50+"\nDividing dataset in training an validation.")
print("-"*50+"\nTraining set.")
X_tr, y_tr = flat_hist(train['histograms'][tr_idxs]), train['labels'][tr_idxs]
print("-"*50+"\nValidation set.")
X_val, y_val = flat_hist(train['histograms'][val_idxs]), train['labels'][val_idxs]

print("="*50+"\nTuning the hyperparameters.")
# Training and validation.
best_C, best_acc = 1, 0
for C_exp in tqdm(range(0, 10)):
    C = 10**C_exp
    SVM = gen_SVM(C, SEED)
    SVM.fit(X_tr, y_tr)
    acc = SVM.score(X_val, y_val)
    if acc > best_acc:
        best_acc, best_C = acc, C
    print("-"*50+f"\nValidation accuracy with C={C:.0e}: {acc*100:.2f}%.")
SVM = gen_SVM(best_C, SEED)
print("-"*50+f"\nBest validation accuracy with C={best_C:.0e}: {best_acc*100:.2f}%.")

# Testing.
print("="*50+"\nTraining the tuned model on whole dataset.")
SVM.fit(flat_hist(np.concatenate([X_tr, X_val])), np.concatenate([y_tr, y_val]))
print("="*50+"\nTesting the model.")
acc = SVM.score(flat_hist(test['histograms'][:]), test['labels'][:])
print(f"Test accuracy: {acc*100:.2f}%.")
f.close()
