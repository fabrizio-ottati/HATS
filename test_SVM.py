#! /usr/bin/env python3

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer as Scaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import numpy as np
import h5py
from tqdm import tqdm
import math
import argparse

#----------------------------------------------

# Generation of a parametric SVM.
gen_SVM = lambda C, seed: Pipeline(
    [
        (
            "scaler",
            Scaler()),
        (
            "classifier",
            SGDClassifier(
                loss="hinge",
                alpha=1 / C,
                fit_intercept=False,
                max_iter=1e6,
                tol=1e-3,
                n_jobs=-1,
                random_state=seed,
            ),
        ),
    ]
)

# Flattening of the histogram for the SVM.
flat = lambda X: X.reshape((X.shape[0], np.prod(X.shape[1:])))

#----------------------------------------------

parser = argparse.ArgumentParser(description="Training of an SVM on DVS datasets encoded with the HATS algorithm.")
parser.add_argument("dataset", type=str, help="The HDF5 file containing the encoded dataset.")
parser.add_argument("--c-min", metavar="C_MIN", type=float, default=1e3, help="The minimum value of C for the linear SVM.")
parser.add_argument("--c-max", metavar="C_MAX", type=float, default=1e8, help="The maximum value of C for the linear SVM.")
args = parser.parse_args()

f = h5py.File(args.dataset, "r")
X_tr, y_tr = f["train"]["histograms"], f["train"]["labels"]
X_ts, y_ts = f["test"]["histograms"], f["test"]["labels"]

SEED, VAL_SIZE = 32, 0.2
tr_idxs, val_idxs, _, _ = train_test_split(
    range(len(y_tr)),
    y_tr[:],
    stratify=y_tr[:],
    test_size=VAL_SIZE,
    random_state=SEED
)

# For HDF5 arrays, the indices need to be sorted.
tr_idxs.sort()
val_idxs.sort()

print("="*50+"\nTuning the hyperparameters.")
# Training and validation.
best_C, best_acc = 0, 0
EXP_MIN, EXP_MAX = math.ceil(math.log10(args.c_min)), math.ceil(math.log10(args.c_max))
for C_exp in tqdm(range(EXP_MIN, EXP_MAX+1)):
    C = 10**C_exp
    SVM = gen_SVM(C, SEED)
    SVM.fit(flat(X_tr[tr_idxs]), y_tr[tr_idxs])
    acc = SVM.score(flat(X_tr[val_idxs]), y_tr[val_idxs])
    if acc > best_acc:
        best_acc, best_C = acc, C
    print("-"*50+f"\nValidation accuracy with C={C:.0e}: {acc*100:.2f}%.")
SVM = gen_SVM(best_C, SEED)
print("-"*50+f"\nBest validation accuracy with C={best_C:.0e}: {best_acc*100:.2f}%.")

# Testing.
print("="*50+"\nTraining the tuned model on whole dataset.")
SVM.fit(flat(X_tr[:]), y_tr[:])
print("="*50+"\nTesting the model.")
acc = SVM.score(flat(X_ts[:]), y_ts[:])
print(f"Test accuracy: {acc*100:.2f}%.")
f.close()
