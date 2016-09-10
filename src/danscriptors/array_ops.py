import numpy as np

def compress_peaks(pitch_f, pitch_amp, n_peaks=32):
    """
    see http://stackoverflow.com/a/11253931

    Returns 2d arrays of freq and mag/rms data;
    The top 16 amplitudes are sorted by frequency.
    """
    index = list(np.ix_(*[np.arange(i) for i in pitch_amp.shape]))
    index[0] = pitch_amp.argpartition(n_peaks, axis=0)[-n_peaks:, :]
    pitch_f = pitch_f[index]
    pitch_amp = pitch_amp[index]
    index[0] = pitch_f.argsort(axis=0)[::-1,:]
    return np.array(pitch_f[index]), np.array(pitch_amp[index])
