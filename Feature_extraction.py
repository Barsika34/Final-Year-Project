import dcase_util
import numpy as np
import os
import librosa

# from scipy.misc import imread
cl = 0.
q = 0
j=0
labels=[]
ex4=[]
p=0
h=0
#%%
def load_audio(audio_path, sr=None):
    # By default, librosa will resample the signal to 22050Hz(sr=None). And range in (-1., 1.)
    print(audio_path)
    sound_sample, sr = librosa.load(audio_path, sr=44100, mono=True,duration=1.0)
    return sound_sample
            
#%%

data_path='C:/Users/Bars/Desktop/ESC-10/audio'
save_path='C:/Users/Bars/Desktop/ESC-10/Features2/'
os.chdir(save_path)

mel_features=dcase_util.features.MelExtractor()  # Class mel feature

for root, dirs, files in os.walk(data_path, topdown=False):
    print('Root:', root)
    for name in dirs:
        print('name:', name)
        parts = []
        parts += [each for each in os.listdir(os.path.join(root,name)) if each.endswith('.ogg')]
        print(parts)
        for part in parts:
            y=load_audio(os.path.join(root,name,part))
            feat_mel=mel_features.extract(y)
            filename =save_path +  name + '_feat' +'/'+ part[0:-4]+'.npy'
            np.save(filename,feat_mel)
#%%

