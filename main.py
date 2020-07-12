import os
import glob
import librosa
from tqdm import tqdm
import pickle
import numpy as np
from python_speech_features import mfcc, fbank, logfbank
from annoy import AnnoyIndex
from collections import Counter


def extract_features(y, sr=16000, nfilt=10, winsteps=0.02):
    try:
        feat = mfcc(y, sr, nfilt=nfilt, winstep=winsteps)
        return feat
    except:
        raise Exception("Extraction feature error")


def crop_feature(feat, i=0, nb_step=10, maxlen=100):
    crop_feat = np.array(feat[i: i + nb_step]).flatten()
    crop_feat = np.pad(crop_feat, (0, maxlen - len(crop_feat)), mode='constant')
    return crop_feat


features = []
songs = []
for song in tqdm(os.listdir('data')):
    song = os.path.join('data', song)
    y, sr = librosa.load(song, sr=16000)
    feat = extract_features(y)
    for i in range(0, feat.shape[0] - 10, 5):
        features.append(crop_feature(feat, i, nb_step=10))
        songs.append(song)

f = 100
t = AnnoyIndex(f, metric='euclidean')

for i in range(len(features)):
    v = features[i]
    t.add_item(i, v)

t.build(100)

song = os.path.join('', 'test.wav')
y, sr = librosa.load(song, sr=16000)
feat = extract_features(y)

results = []
for i in range(0, feat.shape[0], 10):
    crop_feat = crop_feature(feat, i, nb_step=10)
    result = t.get_nns_by_vector(crop_feat, n=5)
    result_songs = [songs[k] for k in result]
    results.append(result_songs)

results = np.array(results).flatten()

most_song = Counter(results)
for item in most_song.most_common():
    print(item[0])
