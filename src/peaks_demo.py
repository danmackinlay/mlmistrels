import librosa as lr
import descriptors.sfio as sfio
from descriptors.util import compress
from descriptors.reconstruct import reconstruct
from pathlib import Path

name = '303_bach'

here = Path(__file__).resolve().parent.parent

samples_dir = here.joinpath('samples')
output_dir = here.joinpath('output')

audio, sr = sfio.load(
    str(samples_dir.joinpath('%s.mp3' % name))
)

print(audio.shape)

n_fft = 2048
hop_length = n_fft/4

H_pitch, H_pitch_mag = lr.piptrack(
    audio,
    sr=sr,
    n_fft=n_fft,
    hop_length=hop_length)

features = compress(H_pitch, H_pitch_mag, n_peaks=16)

print("features.shape=", features.shape)

recon = reconstruct(
    features,
    n_fft=n_fft,
    sr=sr,
    hop_length=hop_length)

audio = sfio.save(
    str(output_dir.joinpath('%s_reconstructed.mp3' % name)),
    recon,
    sr=sr)
