import json
import h5py
import numpy as np
import sys
import os
import glob
import random
import re
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool
import itertools

from tacotron.utils import get_spectrograms
from pathlib import Path

TEST_SPEAKERS = 20
TEST_UTTERANCE_PROPORTION = 0.1
SEGMENT_SIZE = 128
WORKERS = 30
finish = 0

def read_speaker_info(speaker_info_path):
    speaker_infos = {}
    emotion = {'ang':[], 'hap':[], 'sad':[], 'neu':[]}
    speaker2filenames = {}
    with open(speaker_info_path, 'r') as f:
        for i, line in enumerate(f):
            speaker_id = line.strip().split(',')[0]
            speaker_info = line.strip().split(',')[1]
            speaker_infos[speaker_id] = line.strip().split(',')[1:]
            emotion[speaker_info].append(speaker_id)
    return speaker_infos, emotion

def universal_worker(input_pair):
    function, args = input_pair
    return function(*args)

def pool_args(function, *args):
    return zip(itertools.repeat(function), zip(*args))

def make_one_dataset(filename,total,display=False):

    global finish
    speaker_id = filename.strip().split('/')[-1][:-4]
    mel_spec, lin_spec = get_spectrograms(filename)

    # wav = preprocess_wav(Path(filename))
    # d_mel, d_mel_slices = d_wav2spec(wav)

    print('[Processor] - processing {}/{} {} | mel: {} '.format(
       finish*WORKERS, total, speaker_id, mel_spec.shape), end='\r')
    result = {}
    result['speaker_id'] = speaker_id
    result['mel_spec'] = mel_spec
    result['lin_spec'] = lin_spec
    finish += 1
    return result

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 make_dataset.py [folder_name]')
        exit(0)

    folder_name=sys.argv[1]
    root_dir='/home/b04020/2018_autumn/IEMOCAP_full_release/data/'
    save_dir=os.path.join('/home/b04020/2018_autumn/IEMOCAP_full_release/data/h5py/',folder_name)
    os.makedirs(save_dir,exist_ok=True)

    audio_dir=os.path.join(root_dir, 'wav')
    speaker_info_path=os.path.join(root_dir,'label_process.csv')
    h5py_path=os.path.join(save_dir,'dataset.hdf5')

    random.seed(42)
    # Get Speaker Info
    speaker_infos, emotion = read_speaker_info(speaker_info_path)
    speaker_ids = set(list(speaker_infos.keys()))
    test_speaker_ids = []
    for k, v in emotion.items():
        random.shuffle(v)
        test_speaker_ids += v[:int(len(v)*0.1)]
    train_speaker_ids = list(speaker_ids - set(test_speaker_ids))

    print('Train Datas: {}'.format(len(test_speaker_ids)))
    print('Test Datas: {}'.format(len(train_speaker_ids)))

    test_speaker_ids.sort()
    train_speaker_ids.sort()
    train_speaker_ids = [str(i) for i in train_speaker_ids]
    test_speaker_ids = [str(i) for i in test_speaker_ids]

    # Prepare Split List
    train_path_list, test_path_list = [], []

    for speaker in train_speaker_ids:
        train_path_list.append(os.path.join(audio_dir, '{}.wav'.format(speaker)))
    for speaker in test_speaker_ids:
        test_path_list.append(os.path.join(audio_dir, '{}.wav'.format(speaker)))

    with open(os.path.join(save_dir, 'test_speakers.txt'), 'w') as f:
        for speaker in test_speaker_ids:
            line = [speaker]
            line += speaker_infos[speaker]
            line = '    '.join(line)
            f.write(f'{line}\n')

    with open(os.path.join(save_dir, 'train_speakers.txt'), 'w') as f:
        for speaker in train_speaker_ids:
            line = [speaker]
            line += speaker_infos[speaker]
            line = '    '.join(line)
            f.write(f'{line}\n')
    input()
    all_path_list = train_path_list + test_path_list

    with h5py.File(h5py_path, 'w') as f_h5:

        P = Pool(processes=WORKERS) 
        results = P.map(universal_worker, pool_args(make_one_dataset, 
                                                all_path_list, 
                                                [len(all_path_list)]*len(all_path_list)
                                                ))
        P.close()
        P.join()

        train_path_result = results[:len(train_path_list)]
        test_path_result = results[-len(test_path_list):]

        for datatype, results in zip(['train', 'test'], [train_path_result, test_path_result]):
            total_segment = 0
            savenames = []
            for i in tqdm(range(len(results))):
                if len(results[i]['mel_spec']) >= SEGMENT_SIZE:
                    speaker_id = results[i]['speaker_id']
                    f_h5.create_dataset(f'{speaker_id}/mel', data=results[i]['mel_spec'], dtype=np.float32)
                    f_h5.create_dataset(f'{speaker_id}/lin', data=results[i]['lin_spec'], dtype=np.float32)
                    savenames.append(speaker_id)
                    total_segment += 1

            with open(os.path.join(save_dir, '{}_data.json'.format(datatype)), 'w') as f:
                json.dump(savenames,f,indent=4)

            print('{} sets have {} segments'.format(datatype,total_segment))