'''
this code generates a csv file that contains all filename paths for the tile images in the dataset
'''

import glob
import csv
import os
import argparse
parser = argparse.ArgumentParser(description='Training a CNN model with variable batch size.')
parser.add_argument('--c', type=str, default='adc', help='Size of each batch for training.')
parser.add_argument('--group', type=str, default='05', help='Size of each batch for training.')
args = parser.parse_args()

base_path = '/common/deogun/alali/data/lung_png20x/'
phase = 'train'
phase_path = os.path.join(base_path, phase)
class_path = os.path.join(phase_path, 'adc')
case_path = glob.glob(os.path.join(class_path, 'TCGA-{}*'.format(args.group)))
#one_case = case_path[0]
#print('one path: ', one_case)
with open('file_{}_{}.csv'.format(args.c, args.group), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename'])  # Header row
    for case in case_path:
        print('case path = ', case)
        filenames = glob.glob(os.path.join(case, '*'))
        print('number of images found = ', len(filenames))
        for path in filenames:
            writer.writerow([path])
print('done creating file_{}_{}.csv'.format(args.c, args.group))
