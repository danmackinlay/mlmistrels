import numpy as np
import librosa as lr

from random import uniform
#from compress import compress

# still giving choppy output

# reconstruct single slice of audio
def reconstruct_slice(feature_slice, n_fft, sr):
  # form : [[freq, mag], [freq, mag], ...]
  # simply add sine waves at each (freq,mag)
  wave = np.zeros(n_fft, dtype=np.float32)
  
  time = np.linspace(0.0, n_fft/sr, n_fft)
  
  for (freq,mag) in feature_slice:
    # not sure about sqrt etc
    sin = np.sqrt(mag)*np.sin(2*np.pi*freq*time)
    if np.any(sin > 1.0):
      pass
      #print("clipping:",sin)
    wave += sin
  return wave
  

# turn a series of feature descriptors into a waveform
def reconstruct(features, n_fft = 2048, sr = 22050, hop_length = None):
  if (hop_length == None):
    hop_length = n_fft/4
    
  # will be a fraction longer than the original
  wave = np.zeros(lr.frames_to_samples(features.shape[0], hop_length, n_fft)[0], dtype=np.float32)
  
  for frame,feature_slice in enumerate(features):
    sample_start = lr.frames_to_samples(frame, hop_length, n_fft)[0]
    wave_slice = reconstruct_slice(feature_slice, n_fft, sr)
    sample_end = sample_start + len(wave_slice)
    
    # not too sure about this
    if len(wave[sample_start:sample_end]) < len(wave_slice):
      wave_slice = wave_slice[:len(wave[sample_start:sample_end])]
      
    wave[sample_start:sample_end] += wave_slice*np.hanning(len(wave_slice)) # do I need to scale?
    
  return wave