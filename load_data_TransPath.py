'''
This code file is to read data files 
and store them in numpy array file npy

input: multiple np arrays for each casename, such as: TCGA-95.npy, ..., TCGA-NJ.npy
output: one np array for each class, such as x_adc.npy, x_scc.npy, x_normal.npy

the same process in repeated for train,val, test1-3
'''

import os
import h5py
import argparse
import pickle
import numpy as np
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--c', type=str, default='adc') #adc/scc/normal
parser.add_argument('--phase', type=str, default='train') #train/val/test
args = parser.parse_args()
print('args: ', args)

base_path = '/work/deogun/alali/TransPath/features/'
print('loading data from: ', base_path)
phase_path = os.path.join(base_path, args.phase)
print('phase path: ', phase_path)
class_path = os.path.join(phase_path, args.c)
x_shapes_list = []
max_shape0 = 40911
##### source: https://stackoverflow.com/a/59093534/4368684 #######
def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    if pad_width[1] != 0:                      # <-- the only change (0 indicates no padding)
        vector[-pad_width[1]:] = pad_value
##################################################################

# start from here, remove the above code 
print('padding all to shape: ', max_shape0)
padded_class_array = []
case_paths = glob(os.path.join(class_path, '*.npy'))
print('There are {} numpy arrays in {}'.format(len(case_paths), class_path))
for a_path in case_paths:
    a = np.load(a_path)
    print('before padding, shape = ', a.shape)
    temp = np.pad(a, ((0, max_shape0 - a.shape[0]), (0, 0)), mode='constant', constant_values=0)
    print('after padding, shape = ', temp.shape)
    padded_class_array.append(temp)
print('final padded_class_array of length = ', len(padded_class_array))
print('converting list to array')
np_class_array = np.array(padded_class_array)
print('np class array shape: ', np_class_array.shape)
np_filename = os.path.join(class_path, 'x_{}'.format(args.c))
np.save(np_filename, np_class_array)
print(np_filename, ' SAVED')
#for i in range(1, len(class_array)):
#    print('i = ', i)
#    print('shape of class_array[i]: ', class_array[i].shape)
#    np_class_array = np.concatenate((np_class_array, class_array[i]))
#    print('np class array shape: ', np_class_array.shape)
#print('done making np array ')
print('done')

