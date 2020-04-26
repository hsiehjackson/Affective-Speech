import csv
import os
import sys
import numpy as np

out_file = './data/label.csv'

list_category = [
                'ang',
                'hap',
                'sad',
                'neu',
                'fru',
                'exc',
                'fea',
                'sur',
                'dis',
                'oth',
                'xxx'
                ]

category = {}
for c_type in list_category:
    category[c_type] = len(category)

def file_search(dirname, ret, list_avoid_dir=[]):
    
    filenames = os.listdir(dirname)
    
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)

        if os.path.isdir(full_filename) :
            if full_filename.split('/')[-1] in list_avoid_dir:
                continue
            else:
                file_search(full_filename, ret, list_avoid_dir)
            
        else:
            ret.append( full_filename )          


def find_category(lines):
    is_target = True
    
    id = ''
    c_label = ''
    list_ret = []
    
    for line in lines:
        
        if is_target == True:
            
            try:
                id          = line.split('\t')[1].strip()  #  extract ID
                c_label  = line.split('\t')[2].strip()  #  extract category
                if c_label not in category:
                    print("ERROR nokey" + c_label)
                    sys.exit()
                
                list_ret.append( [id, c_label] )
                is_target = False

            except:
                print("ERROR " + line)
                sys.exit()
        
        else:
            if line == '\n':
                is_target = True
        
    return list_ret

def extract_labels( list_in_file, out_file ) :
    id = ''
    lines = []
    list_ret = []
    
    for in_file in list_in_file:
        
        if in_file.split('/')[-1][0]=='.':
            continue

        with open(in_file, 'r') as f:
            lines = f.readlines()
            lines = lines[2:]                           # remove head
            list_ret = find_category(lines)
            
        list_ret = sorted(list_ret)                   # sort based on first element
    
        with open(out_file, 'a') as f:
            csv_writer = csv.writer( f )
            csv_writer.writerows( list_ret )

list_files = []
list_avoid_dir = ['Attribute', 'Categorical', 'Self-evaluation']

for x in range(5):
    sess_name = 'Session' + str(x+1)
    path = sess_name + '/dialog/EmoEvaluation/'
    file_search(path, list_files, list_avoid_dir)
    list_files = sorted(list_files)

    print(sess_name + ", #sum files: " + str(len(list_files)))

extract_labels(list_files, out_file)