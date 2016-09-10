#!/usr/local/bin/python3

from encoder import *
import librosa 

def plot_spec_approx(D, D_approx, comps, acts):

    plt.figure(figsize=(10,8))
    plt.subplot(3, 1, 1)
    librosa.display.specshow(librosa.logamplitude(D**2,
                                                  ref_power=np.max),
                             y_axis='log', x_axis='time')
    plt.title('Input spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.subplot(3, 2, 3)
    librosa.display.specshow(comps, y_axis='log')
    plt.title('Components')
    plt.subplot(3, 2, 4)
    librosa.display.specshow(acts, x_axis='time')
    plt.ylabel('Components')
    plt.title('Activations')
    plt.subplot(3, 1, 3)

    librosa.display.specshow(librosa.logamplitude(D_approx**2,
                                                  ref_power=np.max),
                             y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Reconstructed spectrogram')
    plt.show()

sound_file = '../samples/303_bach.wav'
diagnostic=False
win_len = 2048 

snd, sr = librosa.load(sound_file)

sparse_enc = SparseEncode(win_len)
bin_enc = BinEncode(win_len)

snd_sparse = sparse_enc.decode(sparse_enc.encode(snd))
snd_bin = bin_enc.decode(bin_enc.encode(snd))

# Normalise
snd_sparse = snd_sparse / snd_sparse.max()
snd_bin = snd_bin / snd_bin.max()
librosa.output.write_wav('../output/303_bach_sparse_encoded.wav', snd_sparse, sr)
librosa.output.write_wav('../output/303_bach_bin_encoded.wav', snd_bin, sr)


