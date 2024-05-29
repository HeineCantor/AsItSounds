import os
import shutil
import glob
import numpy as np
import pandas as pd
import pretty_midi
import pypianoroll
import tables
from music21 import converter, instrument, note, chord, stream
import music21
import librosa
import librosa.display
import matplotlib.pyplot as plt
import json
from datetime import datetime
import random

import torch
import torch.nn as nn
from torch.nn import functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

from tqdm.notebook import tqdm, trange

import random
import itertools

root_dir = '/home/heinecantor/Desktop/MIDI Dataset/'
data_dir = root_dir + '/lpd_5/lpd_5_cleansed'
music_dataset_lpd_dir = root_dir + '/lmd_matched'

# Loading
combined_pianorolls = torch.load(os.path.join(root_dir, 'Lakh Piano Dataset', 'full_pianorolls_pt1.pt'))/ 127.0 # perchè? / 127.0
#pianoroll_lengths = torch.load(os.path.join(root_dir, 'Lakh Piano Dataset', 'full_pianorolls_lengths_pt1.pt'))

print(combined_pianorolls.shape)
#print(pianoroll_lengths.shape)