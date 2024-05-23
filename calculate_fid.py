import os
import csv
from pytorch_fid import fid_score
from natsort import natsorted

# Paths to the directories containing real and generated images
directory = '/samples/path'
cifar10_path = '/cifar/10/path'

# Compute FID score
fid_value = fid_score.calculate_fid_given_paths([directory, cifar10_path], batch_size=500, device='cuda', dims=2048)