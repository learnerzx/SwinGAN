
import h5py
import os
import numpy as np

def save_data(out_file_name, rec_imgs, F_rec_Kspaces, fully_sampled_img, ZF_img, rec_Kspaces):
    data_shape = rec_imgs.shape
    with h5py.File(out_file_name, 'w') as f:
        dset = f.create_dataset('rec_imgs', data_shape, data=rec_imgs, compression="gzip", compression_opts=9)
        dset = f.create_dataset('F_rec_Kspaces', data_shape, data=F_rec_Kspaces, compression="gzip", compression_opts=9)
        dset = f.create_dataset('fully_sampled_img', data_shape, data=fully_sampled_img, compression="gzip", compression_opts=9)
        dset = f.create_dataset('ZF_img', data_shape, data=ZF_img, compression="gzip", compression_opts=9)
        dset = f.create_dataset('rec_Kspaces', data_shape, data=rec_Kspaces, compression="gzip", compression_opts=9)
