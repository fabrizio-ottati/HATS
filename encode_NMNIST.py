#! /usr/bin/env python3

import tonic
import numpy as np
import h5py
from tonic.transforms import Denoise
from hats import HATS
from joblib import Parallel, delayed
from tqdm import tqdm

FILTER_TIME=1e4
ds_tr, ds_ts = tonic.datasets.NMNIST(save_to="./data_tr", transform=Denoise(filter_time=FILTER_TIME), train=True), tonic.datasets.NMNIST(save_to="./data_ts", transform=Denoise(filter_time=FILTER_TIME), train=False)

cell_size, surface_size, temporal_window, tau = 5, 5, 100e-3*1e6, 1e11
hats_encoder = HATS(sensor_size=ds_tr.sensor_size, cell_size=cell_size, surface_size=surface_size, tw=temporal_window, tau=tau)
HATS_wrapper = lambda events, label: (hats_encoder.get_histogram(events), label)

BATCH_SIZE, TR_SIZE, TS_SIZE = 64, len(ds_tr), len(ds_ts)

f = h5py.File('NMNIST.hdf5', 'w')
hist_shape = HATS_wrapper(*ds_tr[0])[0].shape
train, test = f.create_group("train"), f.create_group("test")

hists = train.create_dataset("histograms", (TR_SIZE, *hist_shape), dtype=np.float32)
labels = train.create_dataset("labels", (TR_SIZE,), dtype=np.int32)
for i in tqdm(range(0, TR_SIZE, BATCH_SIZE)):
  X, y = tuple(zip(*Parallel(n_jobs=-1)(delayed(HATS_wrapper)(*ds_tr[j]) for j in range(i, min(TR_SIZE, i+BATCH_SIZE)))))
  hists[i:min(TR_SIZE, i+BATCH_SIZE), :, :, :], labels[i:min(TR_SIZE, i+BATCH_SIZE)] = np.stack(X), np.stack(y).astype(dtype=np.int32)

hists = test.create_dataset("histograms", (TS_SIZE, *hist_shape), dtype=np.float32)
labels = test.create_dataset("labels", (TS_SIZE,), dtype=np.int32)
for i in tqdm(range(0, TS_SIZE, BATCH_SIZE)):
  X, y = tuple(zip(*Parallel(n_jobs=-1)(delayed(HATS_wrapper)(*ds_ts[j]) for j in range(i, min(TS_SIZE, i+BATCH_SIZE)))))
  hists[i:min(TS_SIZE, i+BATCH_SIZE), :, :, :], labels[i:min(TS_SIZE, i+BATCH_SIZE)] = np.stack(X), np.stack(y).astype(dtype=np.int32)

f.close()
  
