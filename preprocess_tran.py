import csv
import os




os.makedirs('./data',exist_ok=True)
out_file = './data/processed_tran.csv'

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

def extract_trans( list_in_file, out_file ) :
    
    lines = []
    
    for in_file in list_in_file:
        cnt = 0

        with open(in_file, 'r') as f:
            lines = f.readlines()

        with open(out_file, 'a') as f:

            csv_writer = csv.writer( f )
            lines = sorted(lines)                  # sort based on first element
            
            for line in lines:

                name = line.split(':')[0].split(' ')[0].strip()
                
                # unwanted case 
                if name[:3] != 'Ses':             # noise transcription such as reply  M: sorry
                    continue
                elif name[-3:-1] == 'XX':        # we don't have matching pair in label
                    continue
                trans = line.split(':')[1].strip()
                
                cnt += 1
                csv_writer.writerow([name, trans])


list_files = []

for x in range(5):
    sess_name = 'Session' + str(x+1)
    path = sess_name + '/dialog/transcriptions/'
    file_search(path, list_files)
    list_files = sorted(list_files)

    print(sess_name + ", #sum files: " + str(len(list_files)))


extract_trans(list_files, out_file)