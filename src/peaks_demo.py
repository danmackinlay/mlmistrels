import numpy as np
import librosa as lr

from descriptors.util import compress
from descriptors.reconstruct import reconstruct

name = '303_bach'

audio,sr = lr.load('../samples/%s.wav'%name)

print(audio.shape)

n_fft = 2048
hop_length = n_fft/4

H_pitch, H_pitch_mag = lr.piptrack(audio, sr = sr, n_fft = n_fft, hop_length = hop_length)

features = compress(H_pitch, H_pitch_mag, n_peaks = 16)

print("features.shape=",features.shape)

recon = reconstruct(features, n_fft = n_fft, sr = sr, hop_length = hop_length)

recon = recon.astype(np.float32)

audio = lr.output.write_wav('%s_reconstructed.wav'%name, recon, sr = sr)