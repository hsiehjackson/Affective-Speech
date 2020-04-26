from tacotron.utils import get_spectrograms
import matplotlib as mpl
from matplotlib import pyplot as plt
import os
import librosa.display as lds

folders = '/home/b04020/2018_autumn/IEMOCAP_full_release/data/wav/'
test = 'test.wav'
files = os.listdir(folders)
# mel_spec, lin_spec = get_spectrograms(test)
# lds.specshow(mel_spec.T, y_axis='mel', x_axis='time', cmap=plt.cm.Blues)
# plt.show()

for f in files:
    mel_spec, lin_spec = get_spectrograms(os.path.join(folders,f))
    print(lin_spec[:,-1:])
    print(lin_spec[:,:1])
    lds.specshow(lin_spec.T, y_axis='mel', x_axis='time', cmap=plt.cm.Blues)
    plt.show()

