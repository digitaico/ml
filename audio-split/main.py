# https://dev.to/highcenburg/separate-vocals-from-a-track-using-python-4lb5

import os
import librosa
from librosa import display
import numpy as np 
import IPython.display as ipd 
import matplotlib as plt

y, sr = librosa.load("sources/a-68.wav")

ipd.Audio(data=y[90*sr:110*sr], rate=sr)
ipd.Audio(data = y, rate=sr)

S_full, phase = librosa.magphase(librosa.stft(y))

idx = slice(*librosa.time_to_frames([90*110], sr=sr))
fig, ax = plt.pyplot.subplots()
img = display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max), y_axis='log', x_axis='time', sr=sr, ax=ax)
fig.colorbar(img, ax=ax)

S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric='cosine', width=int(librosa.time_to_frames(2,sr=sr)))
S_filter = np.minimum(S_full, S_filter)

margin_i, margin_v = 3, 11
power = 3

mask_i = librosa.util.softmask(S_filter, margin_i * (S_full - S_filter), power=power)
mask_v = librosa.util.softmask(S_full - S_filter, margin_v * S_filter, power=power)

S_foreground = mask_v *  S_full
S_background = mask_i * S_full

fig, ax = plt.pyplot.subplots(nrows=3, sharex=True, sharey = True)
img = display.specshow(librosa.amplitude_to_db(S_full[:,idx], ref=np.max), y_axis='log', x_axis='time', sr=sr, ax=ax[0])
ax[0].set(title='Full Spectrum')
ax[0].label_outer()

display.specshow(librosa.amplitude_to_db(S_background[:, idx], ref=np.max), y_axis='log', x_axis='time', sr=sr, ax=ax[1])
ax[1].set(title='Background Spectrum')
ax[1].label_outer()

display.specshow(librosa.amplitude_to_db(S_background[:, idx], ref=np.max), y_axis='log', x_axis='time', sr=sr, ax=ax[2])
ax[2].set(title='Foreground Spectrum')
ax[2].label_outer()

fig.colorbar(img, ax=ax)

y_foreground = librosa.istft(S_foreground * phase)
ipd.Audio(data=y_foreground[90*sr:110:sr], rate=sr)
"""