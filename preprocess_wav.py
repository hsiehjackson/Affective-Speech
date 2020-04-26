import csv
import os
import sys
import numpy as np

# out_folder = './data/wav'
# os.makedirs(out_folder,exist_ok=True)

# def file_search(dirname, ret, list_avoid_dir=[]):
    
#     filenames = os.listdir(dirname)
    
#     for filename in filenames:
#         full_filename = os.path.join(dirname, filename)

#         if os.path.isdir(full_filename) :
#             if full_filename.split('/')[-1] in list_avoid_dir:
#                 continue
#             else:
#                 file_search(full_filename, ret, list_avoid_dir)
            
#         else:
#             if full_filename.split('/')[-1][0] != '.':
#                 ret.append( full_filename )        

# list_files = []

# for x in range(5):
#     sess_name = 'Session' + str(x+1)
#     path = sess_name + '/sentences/wav'
#     file_search(path, list_files)
#     list_files = sorted(list_files)

# print(len(list_files))
# for f in list_files:
#     cmd = 'cp {} {}'.format(f,out_folder)
#     os.system(cmd)

filenames = os.listdir('./data/wav')
print(len(filenames))
filenames = [n[:-4] for n in filenames ]

labelnames = []
with open('./data/label.csv') as f:
    for l in f.readlines():
        labelnames.append(l.split(',')[0])
print(len(labelnames))

print(filenames[:5])
print(labelnames[:5])

filenames = set(filenames)
labelnames = set(labelnames)
print(filenames - labelnames)