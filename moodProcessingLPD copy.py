import os
import pandas as pd
import pickle
from sklearn.preprocessing import normalize
import numpy as np

# Specify the directory you want to crawl
directory = '/home/heinecantor/Desktop/MIDI Dataset/lpd_5/lpd_5_cleansed/'
moodDataframe = pd.read_csv('/home/heinecantor/Desktop/MIDI Dataset/7labels.csv')[:8386] # noi non approviamo reddit siamo contro e ci dissociamo pesantemente

# Reading the genre annotations
genre_file_dir = '/home/heinecantor/Desktop/MIDI Dataset/msd_tagtraum_cd1.cls'
ids = []
genres = []
with open(genre_file_dir) as f:
    line = f.readline()
    while line:
        if line[0] != '#':
          split = line.strip().split("\t")
          if len(split) == 2:
            ids.append(split[0])
            genres.append(split[1])
          elif len(split) == 3:
            ids.append(split[0])
            ids.append(split[0])
            genres.append(split[1])
            genres.append(split[2])
        line = f.readline()
genre_df = pd.DataFrame(data={"TrackID": ids, "Genre": genres})

genre_dict = genre_df.groupby('TrackID')['Genre'].apply(lambda x: x.tolist()).to_dict()

genre_dict = {k:v[0] for k, v in genre_dict.items()}
#print(genre_dict)

processed_genre_dict = {}

# Iterate over all directories and subdirectories
for root, dirs, files in os.walk(directory):
    # Process each directory
    for file in files:
        # Do something with the directory
        fullPath = os.path.join(root, file)
        fileName = fullPath.split('/')[-1]
        dirName = fullPath.split('/')[-2]
        #print(dirName + " - " + fileName.split('.')[0])
        try:
            processed_genre_dict[dirName] = genre_dict[dirName]
        except:
            pass

distinct_genres = list(genre_dict.values())
distinct_genres = list(set([element for element in distinct_genres]))

for i in range(len(distinct_genres)):
    print(list(genre_dict.values()).count(distinct_genres[i]))

# Save the moodDictProcessed dictionary in a file
with open('/home/heinecantor/Desktop/git/AsItSounds/p_genre_dict.pkl', 'wb') as file:
    pickle.dump(processed_genre_dict, file)