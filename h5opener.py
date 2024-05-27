import h5py

PATH = '/home/heinecantor/Desktop/MIDI Dataset/lmd_matched_h5/A/A/A/TRAAAGR128F425B14B.h5'
#PATH = '/home/heinecantor/Desktop/MIDI Dataset/msd_summary_file.h5'

file = h5py.File(PATH, 'r')
print(file.keys())
print(list(file['metadata']))