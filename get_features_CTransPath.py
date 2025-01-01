import os
import csv
import sys
import glob
import pandas as pd
import argparse
import torch
print('Torch version = ', torch.__version__)
import torchvision
print('Torchvision version = ', torchvision.__version__)
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from ctran import ctranspath
import numpy as np

parser = argparse.ArgumentParser(description='Getting features from CTransPath.')
parser.add_argument('--phase', type=str, default='train', help='name.')
parser.add_argument('--c', type=str, default='adc', help='name of the file that contains image paths.')
parser.add_argument('--group', type=str, default='05', help='name.')
args = parser.parse_args()

print('args: ', args)

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
trnsfrms_val = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std)
    ]
)
class roi_dataset(Dataset):
    def __init__(self, img_csv,
                 ):
        super().__init__()
        self.transform = trnsfrms_val

        self.images_lst = img_csv

    def __len__(self):
        return len(self.images_lst)

    def __getitem__(self, idx):
        path = self.images_lst.filename[idx]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)


        return image


model = ctranspath()
model.head = nn.Identity()
td = torch.load(r'./ctranspath.pth')
model.load_state_dict(td['model'], strict=True)
print('loaded model successfully')

# get filenames here instead of saving them into csv file
base_path = '/common/deogun/alali/data/lung_png20x/'
out_path = 'features/'
if(not os.path.exists(out_path)):
    os.mkdir(out_path)
out_phase_path = os.path.join(out_path, args.phase)
if(not os.path.exists(out_phase_path)):
    os.mkdir(out_phase_path)
out_class_path = os.path.join(out_phase_path, args.c)
if(not os.path.exists(out_class_path)):
    os.mkdir(out_class_path)
print('output path: ', out_class_path)

phase_path = os.path.join(base_path, args.phase)
class_path = os.path.join(phase_path, args.c)
case_path = glob.glob(os.path.join(class_path, 'TCGA-{}*'.format(args.group)))
for k, case in enumerate(case_path):
    print('case [{}/{}]= {}'.format(k, len(case_path), case))
    case_list = case.split('/')
    case_name = case_list[-1]
    with open('{}_{}_{}.csv'.format(args.phase, args.c, case_name), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename'])  # Header row
        filenames = glob.glob(os.path.join(case, '*'))
        print('number of images found = ', len(filenames))
        for path in filenames:
            writer.writerow([path])
    print('done creating {}_{}_{}.csv'.format(args.phase, args.c, case_name))
    img_csv = pd.read_csv('{}_{}_{}.csv'.format(args.phase, args.c, case_name))
    #print('img_csv of type: ', type(img_csv))
    #print('img_csv = ', img_csv)
    test_datat=roi_dataset(img_csv)
    database_loader = torch.utils.data.DataLoader(test_datat, batch_size=1, shuffle=False)

    #model = ctranspath()
    #model.head = nn.Identity()
    #td = torch.load(r'./ctranspath.pth')
    #model.load_state_dict(td['model'], strict=True)
    #print('loaded model successfully')
    feat_list = []
    model.eval()
    sample_size = 0
    with torch.no_grad():
        for i, batch in enumerate(database_loader):
            #print('== loading batch [{}/{}] batch of shape: {}'.format(i,  batch.shape))
            features = model(batch)
            features = features.cpu().numpy()
            features = features.squeeze()
            # Get the size of one object (average size)
            sample_size = sys.getsizeof(features)
            if(i % 100 == 0):
                print('== loading batch [{}/{}] batch of shape: {}'.format(i, len(database_loader), batch.shape))
                print('=== extracted features of shape = ', features.shape)
                print('=== size of one feature vector in bytes = ', sample_size)
            feat_list.append(features)

    print('done array of length = ', len(feat_list))
    # Estimate the total memory required
    print('size of one feature vector = ', sample_size)
    estimated_memory = sample_size * len(feat_list)
    # Convert to megabytes for easier interpretation
    estimated_memory_gb = estimated_memory / (1024 ** 2)
    print(f"Estimated memory required: {estimated_memory_gb:.4f} MB")
    print('Trying to convert the list to numpy array...')
    feat_np = np.array(feat_list)
    print('done np array of shape: ', feat_np.shape)
    save_path = os.path.join(out_class_path, case_name)
    print('trying to save array in path: ', save_path)
    np.save(arr=feat_np, file=save_path)
    print('np array succ saved')
print('============== done =====================')
