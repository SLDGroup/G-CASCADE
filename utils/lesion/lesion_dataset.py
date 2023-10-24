import sys
import os.path as p

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import albumentations.augmentations.functional as F

sys.path.append('..')
import helpers as h
import polar_transformations

class LesionDataset(Dataset):

  width = 512
  height = 384

  in_channels = 3
  out_channels = 1

  def __init__(self, directory, polar=True, manual_centers=None, center_augmentation=False, percent=None):
    self.directory = p.join('datasets/lesion', directory)
    self.polar = polar
    self.manual_centers = manual_centers
    self.center_augmentation = center_augmentation
    self.percent = percent

    self.file_names = h.listdir(p.join(self.directory, 'label'))
    self.file_names.sort()
    
  def __len__(self):
    length = len(self.file_names)
    if self.percent is not None:
      length = int(length * self.percent)
    return length

  def __getitem__(self, idx):
    file_name = self.file_names[idx]
    label_file = p.join(self.directory, 'label', file_name)
    input_file = p.join(self.directory, 'input', file_name.replace('.png', '.jpg'))

    label = cv.imread(label_file, cv.IMREAD_GRAYSCALE)
    label = label.astype(np.float32)
    label /= 255.0

    input = cv.imread(input_file)
    input = cv.cvtColor(input, cv.COLOR_BGR2RGB)
    input = input.astype(np.float32)
    input /= 255.0
    input -= 0.5 
    
    # convert to polar
    if self.polar:
      if self.manual_centers is not None:
        center = self.manual_centers[idx]
      else:
        center = polar_transformations.centroid(label)

      if self.center_augmentation and np.random.uniform() < 0.3:
        center_max_shift = 0.05 * LesionDataset.height
        center = np.array(center)
        center = (
          center[0] + np.random.uniform(-center_max_shift, center_max_shift),
          center[1] + np.random.uniform(-center_max_shift, center_max_shift))
      
      input = polar_transformations.to_polar(input, center)
      label = polar_transformations.to_polar(label, center)

    # to PyTorch expected format
    input = input.transpose(2, 0, 1)
    label = np.expand_dims(label, axis=-1)
    label = label.transpose(2, 0, 1)

    input_tensor = torch.from_numpy(input)
    label_tensor = torch.from_numpy(label)

    return input_tensor, label_tensor
