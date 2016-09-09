import numpy as np

#ewww
import sys, os
sys.path.insert(0, os.path.abspath('..'))

from danscriptors import analysis

# return an array of form [[freq, mag], [freq, mag], ...]
def compress_single_slice(freqs, mags, n_peaks = 16):
  # get n_peaks top peaks
  ind = np.argpartition(mags, -n_peaks)[-n_peaks:]
  return np.dstack((freqs[ind], mags[ind]))[0]

# turn output from danscriptors.analysis.harmonic_features into compressed peak data
# (see above for format)
def compress(H_pitch, H_pitch_mag, n_peaks = 16):
  # take the transpoe to make things easier
  H_pitch = H_pitch.T
  H_pitch_mag = H_pitch_mag.T
  
  features = np.zeros((H_pitch.shape[0],)+(n_peaks, 2))
  
  # i.e. H_pitch[t,f] is pitch, same for pitch_mag
  for t in range(H_pitch.shape[0]):
    features[t] = compress_single_slice(H_pitch[t], H_pitch_mag[t], n_peaks)
    
  return features
  
