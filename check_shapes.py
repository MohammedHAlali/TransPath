import numpy as np
import os
from glob import glob

inpath = 'features/test3/normal'
np_array_list = glob(os.path.join(inpath, '*.npy'))
# Find the maximum x (number of rows)
#max_x = max(a.shape[0] for a=np.load(arr) in array_list)
#print(f"The maximum x is: {max_x}")

print('there are {} array files'.format(len(np_array_list)))
my_max = 0
for a in np_array_list:
    np_array = np.load(a)
    print('array shape: ', np_array.shape)
    my_max = max(my_max, np_array.shape[0])

print(f"The maximum x (rows) among the arrays is: {my_max}")


