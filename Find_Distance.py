import os
os.chdir(/home/donghan/DeepLabCut)

import h5py
filename = '1034 SI_B, Aug 15, 13 7 49DeepCut_resnet50_ReachingMar12shuffle1_800.h5'
f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]

# Get the data
data = list(f[a_group_key])
