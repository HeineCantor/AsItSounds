import os
import pandas as pd
import pickle
from sklearn.preprocessing import normalize
import numpy as np

# Specify the directory you want to crawl
directory = '/home/heinecantor/Desktop/MIDI Dataset/lpd_5/lpd_5_cleansed/'
moodDataframe = pd.read_csv('/home/heinecantor/Desktop/MIDI Dataset/7labels.csv')[:8386] # noi non approviamo reddit siamo contro e ci dissociamo pesantemente

# Transform pandas dataframe into a dictionary
moodDict = moodDataframe.set_index(moodDataframe.columns[0]).T.to_dict('list')

# Change every key in the dictionary
moodDict = {new_key: value for old_key, value in moodDict.items() for new_key in [old_key.split('/')[-1].split('.')[0]]}

print(moodDict)

moodDictProcessed = {}

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
            numpyArray = np.asarray(moodDict[fileName.split('.')[0]])
            moodDictProcessed[dirName] = np.argmax(numpyArray / np.sum(numpyArray))
        except:
            pass

print(moodDictProcessed)

for i in range(7):
    print(list(moodDictProcessed.values()).count(i))

# Save the moodDictProcessed dictionary in a file
with open('/home/heinecantor/Desktop/git/AsItSounds/mood_dict.pkl', 'wb') as file:
    pickle.dump(moodDictProcessed, file)