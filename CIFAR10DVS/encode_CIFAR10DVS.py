#! /usr/bin/env python3

import tonic
from sklearn.model_selection import train_test_split
import numpy as np
import h5py
import sys; sys.path.append("../")
from hats import HATS
from joblib import Parallel, delayed
from tqdm import tqdm
from tonic.transforms import Denoise

assert len(sys.argv)>=2, "Error: too few arguments (HDF5 file name is probably missing."

ds = tonic.datasets.CIFAR10DVS(save_to="./data", transform=Denoise())

cell_size, surface_size, temporal_window, tau = 12, 9, 1e4, 2e5
HATS_instance = HATS(sensor_size=ds.sensor_size, cell_size=cell_size, surface_size=surface_size, tau=tau, tw=temporal_window)
HATS_wrapper= lambda events, label: (HATS_instance.get_histogram(events), label)

# Dividing the dataset in training and testing ones.
TS_PERC, SEED = 0.2, 32
train_indices, test_indices, _, _ = train_test_split(
  range(len(ds)),
  ds.targets,
  stratify=ds.targets,
  test_size=TS_PERC,
  random_state=SEED
)
BATCH_SIZE, TR_SIZE, TS_SIZE = 64, len(train_indices), len(test_indices)

f = h5py.File(sys.argv[1], 'w')
hist_shape = HATS_wrapper(*ds[0])[0].shape
train, test = f.create_group("train"), f.create_group("test")

hists = train.create_dataset("histograms", (TR_SIZE, *hist_shape), dtype=np.float32)
labels = train.create_dataset("labels", (TR_SIZE,), dtype=np.int32)
for i in tqdm(range(0, TR_SIZE, BATCH_SIZE)):
  X, y = tuple(zip(*Parallel(n_jobs=-1)(delayed(HATS_wrapper)(*ds[train_indices[j]]) for j in range(i, min(TR_SIZE, i+BATCH_SIZE)))))
  hists[i:min(TR_SIZE, i+BATCH_SIZE), :, :, :], labels[i:min(TR_SIZE, i+BATCH_SIZE)] = np.stack(X), np.stack(y).astype(dtype=np.int32)

hists = test.create_dataset("histograms", (TS_SIZE, *hist_shape), dtype=np.float32)
labels = test.create_dataset("labels", (TS_SIZE,), dtype=np.int32)
for i in tqdm(range(0, TS_SIZE, BATCH_SIZE)):
  X, y = tuple(zip(*Parallel(n_jobs=-1)(delayed(HATS_wrapper)(*ds[test_indices[j]]) for j in range(i, min(TS_SIZE, i+BATCH_SIZE)))))
  hists[i:min(TS_SIZE, i+BATCH_SIZE), :, :, :], labels[i:min(TS_SIZE, i+BATCH_SIZE)] = np.stack(X), np.stack(y).astype(dtype=np.int32)

f.close()
  
