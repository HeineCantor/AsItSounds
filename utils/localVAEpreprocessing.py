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

RESULTS_PATH = os.path.join(root_dir, 'Lakh Piano Dataset', 'Metadata')

# Utility functions for retrieving paths from a msd_id (milion song dataset id)
def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

# function for retrieving path of file h5 (metadata) from a msd_id
def msd_id_to_h5(msd_id):
    """Given an MSD ID, return the path to the corresponding h5"""
    return os.path.join(RESULTS_PATH, 'lmd_matched_h5',
                        msd_id_to_dirs(msd_id) + '.h5')

# Load the midi npz file from the LMD cleansed folder, given the msd_id and the md5
def get_midi_npz_path(msd_id, midi_md5):
    return os.path.join(data_dir,
                        msd_id_to_dirs(msd_id), midi_md5 + '.npz')

# Load the midi file from the Music Dataset folder
def get_midi_path(msd_id, midi_md5):
    return os.path.join(music_dataset_lpd_dir,
                        msd_id_to_dirs(msd_id), midi_md5 + '.mid')

# Open the cleansed ids - cleansed file ids : msd ids
cleansed_ids = pd.read_csv(os.path.join(root_dir, 'cleansed_ids.txt'), delimiter = '    ', header = None, engine ='python')
lpd_to_msd_ids = {a:b for a, b in zip(cleansed_ids[0], cleansed_ids[1])}
msd_to_lpd_ids = {a:b for a, b in zip(cleansed_ids[1], cleansed_ids[0])}

# Reading the genre annotations
genre_file_dir = os.path.join(root_dir, 'msd_tagtraum_cd1.cls')
ids = []
genres = []

with open(genre_file_dir) as f:
    line = f.readline()
    while line:

        # Avoid the initial lines of the file
        if line[0] != '#':
          split = line.strip().split("\t")

          # Single genre case
          if len(split) == 2:
            ids.append(split[0])
            genres.append(split[1])
          # Sub-genre case
          elif len(split) == 3:
            ids.append(split[0])
            ids.append(split[0])
            genres.append(split[1])
            genres.append(split[2])
        line = f.readline()

# Dataframe and dictionary
genre_df = pd.DataFrame(data={"TrackID": ids, "Genre": genres})
genre_dict = genre_df.groupby('TrackID')['Genre'].apply(lambda x: x.tolist()).to_dict()

print(f"Total number of samples: {len(msd_to_lpd_ids.keys())}")

train_ids = np.random.choice(list(msd_to_lpd_ids.keys()), size=5000, replace=False)
# train_ids = [10000:] # full dataset loading (~21.000 songs)

from tqdm import tqdm

combined_pianorolls = []
i = 0
for msd_file_name in tqdm(train_ids):

  lpd_file_name = msd_to_lpd_ids[msd_file_name]
  # Get the NPZ path
  npz_path = get_midi_npz_path(msd_file_name, lpd_file_name)
  #print(npz_path)
  multitrack = pypianoroll.load(npz_path)
  #print(multitrack)
  multitrack.set_resolution(2).pad_to_same()
  #print(multitrack)

  # Piano, Guitar, Bass, Strings, Drums
  # Splitting into different parts

  parts = {'piano_part': None, 'guitar_part': None, 'bass_part': None, 'strings_part': None, 'drums_part': None}
  song_length = None
  empty_array = None
  has_empty_parts = False
  for track in multitrack.tracks:
    #print(track.pianoroll.shape)
    #print(track.pianoroll)
    if track.name == 'Drums':
      parts['drums_part'] = track.pianoroll
    if track.name == 'Piano':
      parts['piano_part'] = track.pianoroll
    if track.name == 'Guitar':
      parts['guitar_part'] = track.pianoroll
    if track.name == 'Bass':
      parts['bass_part'] = track.pianoroll
    if track.name == 'Strings':
      parts['strings_part'] = track.pianoroll
    if track.pianoroll.shape[0] > 0:
      empty_array = np.zeros_like(track.pianoroll)
      #print(empty_array)
      #print(track.pianoroll)

  for k,v in parts.items():
    if v.shape[0] == 0:
      parts[k] = empty_array.copy()
      has_empty_parts = True

  # Stack all together - Piano, Guitar, Bass, Strings, Drums
  combined_pianoroll = torch.tensor([parts['piano_part'], parts['guitar_part'], parts['bass_part'], parts['strings_part'], parts['drums_part']])
  #print(combined_pianoroll.shape)

  # These contain velocity information - the force with which the notes are hit - which can be standardized to 0/1 if we want (to compress)
  if has_empty_parts == False:
    combined_pianorolls.append(combined_pianoroll)
    #print(combined_pianorolls.size())
    i+=1
    #print(i)

# Stack of the pianorolls and list of lengths

pianoroll_lengths = [e.size()[1] for e in combined_pianorolls]
#print(combined_pianorolls.size()[1])
#print(pianoroll_lengths)
combined_pianorolls = torch.hstack(combined_pianorolls)
#print(combined_pianorolls)

# Saving files

torch.save(combined_pianorolls, os.path.join(root_dir, 'Lakh Piano Dataset', '5000_pianorolls.pt'))
pianoroll_lengths = torch.tensor(pianoroll_lengths)
#print(pianoroll_lengths)
torch.save(pianoroll_lengths, os.path.join(root_dir, 'Lakh Piano Dataset', '5000_pianorolls_lengths.pt'))