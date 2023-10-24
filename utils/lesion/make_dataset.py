import sys
import os
import numpy as np
from shutil import copyfile

from torch.utils.data import random_split
import albumentations.augmentations.functional as F
import cv2 as cv

#sys.path.append('../../')
import helpers as h

VALID_INPUT_FOLDER = '../../data/ISIC2018/ISIC2018_Task1-2_Validation_Input'
TRAIN_INPUT_FOLDER = '../../data/ISIC2018/ISIC2018_Task1-2_Training_Input'
VALID_GT_FOLDER = '../../data/ISIC2018/ISIC2018_Task1_Validation_GroundTruth'
TRAIN_GT_FOLDER = '../../data/ISIC2018/ISIC2018_Task1_Training_GroundTruth'

def get_files(folder):
  files = h.listdir(folder)
  files.sort()
  files = [f for f in files if not '.txt' in f]
  files = [os.path.join(folder, f) for f in files]
  return files

valid_input = get_files(VALID_INPUT_FOLDER)
train_input = get_files(TRAIN_INPUT_FOLDER)

valid_gt = get_files(VALID_GT_FOLDER)
train_gt = get_files(TRAIN_GT_FOLDER)

inputs = valid_input + train_input
gts = valid_gt + train_gt

# split same as in Double U-Net paper: https://arxiv.org/pdf/2006.04868v2.pdf
train_valid_test_split = (0.8, 0.1, 0.1)

test_count = int(train_valid_test_split[2] * len(inputs))
valid_count = test_count
train_count = len(inputs) - test_count * 2

print(train_count, valid_count, test_count)
assert(test_count + valid_count + train_count == len(inputs))

all_files = np.array(list(zip(inputs, gts)))
np.random.seed(42)
np.random.shuffle(all_files)

train_files = all_files[:train_count]
valid_files = all_files[train_count : train_count + valid_count]
test_files = all_files[-test_count:]

def save_files(files, folder):
  h.mkdir(folder)
  h.mkdir(os.path.join(folder, 'images'))
  h.mkdir(os.path.join(folder, 'masks'))

  for input_file, gt_file in files:
    file_name = input_file.split('/')[-1]
    input_destination = os.path.join(folder, 'images', file_name)

    input_img = cv.imread(input_file)
    input_img = F.resize(input_img, 384, 512)

    gt_img = cv.imread(gt_file, cv.IMREAD_GRAYSCALE)
    gt_img = F.resize(gt_img, 384, 512)

    cv.imwrite(input_destination, input_img)

    gt_destionation = os.path.join(folder, 'masks', file_name.replace('.jpg', '.png'))
    cv.imwrite(gt_destionation, gt_img)

save_files(train_files, '../../data/ISIC2018/train')
save_files(valid_files, '../../data/ISIC2018/valid')
save_files(test_files, '../../data/ISIC2018/test')
