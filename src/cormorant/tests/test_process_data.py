import torch
from process_data import process_data_files, process_xyz_md17, process_xyz_gdb9

datadir = '/users/branderson/datasets/raw'
# datafile = 'uracil.tar.bz2'
# proc_func = process_xyz_md17

datafile = 'dsgdb9nsd.xyz.tar.bz2'
proc_func = process_xyz_gdb9

molecules = process_data_files('/'.join([datadir, datafile]), proc_func, file_idx_list=[1, 2, 3])

# print(len(molecules))



breakpoint()
