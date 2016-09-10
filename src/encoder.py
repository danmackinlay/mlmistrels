#!/bin/local/bin/python3

# A utility, script, or potentially one day class to convert audio waveforms in to windowed, reduced descriptors, for some machine learning algorithm to go nuts on later

# Authors: James Nichols, Darwin Vickers

# Includes a test of converting then reversing the predictor to see how things sound. Uses Librosa extensively.

import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def randomise_phase(D):
    """ A function that takes reals of any and randomises all the phases,
        it does so by randomly modifying the angle of a complex number """

    # Create a univariate matrix, use the euler identity to make
    # uniformly distributed complex numbers of arg 1
    rands = np.exp(np.vectorize(complex)(0, 2.0 * np.pi * np.random.random(D.shape)))

    return D * rands

class Encode(object):

    def __init__(self, win_len = 2048):
        self.win_len = win_len

    def encode(self, sound):
        return sound
    def decode(self, A):
        return A

class SparseEncode(Encode):
    """ An encoder that uses sparse tensor representation of the spectrogram """

    def __init__(self, win_len = 2048, n_decomp = 4):
        import sklearn.decomposition

        self.win_len = win_len
        self.n_decomp = n_decomp
        self.T = sklearn.decomposition.MiniBatchDictionaryLearning(n_components=self.n_decomp)

    def encode(self, sound):
        D = librosa.stft(sound, n_fft=self.win_len)
        self.comps, A = librosa.decompose.decompose(np.abs(D), transformer=self.T, sort=True)
        return A

    def decode(self, A):
        return librosa.istft(randomise_phase(self.comps.dot(A)))

class BinEncode(Encode):

    def __init__(self, win_len = 2048, n_bins = 32):
        self.win_len = win_len
        self.n_bins = 32
        self.bin_size = self.win_len // (2 * self.n_bins)


    def encode(self, sound):
        D = librosa.stft(sound, n_fft=self.win_len)

        # Make the time series of predictors
        A = np.zeros([self.n_bins+1, D.shape[1]], dtype=np.complex)

        # Try bin-power
        for t in range(D.shape[1]):
            # Consider the 0 Hz component separately (Maybe get rid of this...?)
            A[0, t] = D[0, t]
            # Simple mean of the complex numbers n the bin...
            A[1:,t] = np.array([np.mean(D[b_start:b_start+self.bin_size,t])*self.bin_size for b_start in range(1, D.shape[0], self.bin_size)])

        return A

    def decode(self, A):

        D = np.zeros((self.win_len//2+1, A.shape[1]), dtype=np.complex)

        for t in range(A.shape[1]):

            # Simple covering of the bin with mean of the bin
            D[0, t] = A[0, t]
            D[1:, t] = np.repeat(A[1:, t], self.bin_size)

            # The center frequency is given the average
            #D_r[0, t] = TS[0, t]
            #D_r[1+bin_size//2:D_r.shape[0]:bin_size, t] = TS[1:, t]

            # Random frequency in bin is given the average
        return librosa.istft(randomise_phase(D))


class PeaksEncode(Encode):
    hop_length = 0
    def __init__(self, win_len = 2048):
        from descriptors import util
        self.win_len = win_len
        self.hop_length = win_len/4

        self.sr=22050

    def encode(self, sound):
        H_pitch, H_pitch_mag = lr.piptrack(audio, sr = self.sr, n_fft = self.win_len, hop_length = self.hop_length)

        features = compress(H_pitch, H_pitch_mag, n_peaks = 16)

        return features

    def decode(self, A):
        reconstruct(A, n_fft = self.win_length, sr = self.sr, hop_length = self.hop_length)
        return A
