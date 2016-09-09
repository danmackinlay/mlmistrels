import numpy as np

#from compress import compress

# still giving choppy output

def reconstruct_slice(feature_slice, n_samples, sample_rate):
  # form : [[freq, mag], [freq, mag], ...]
  # simply add sine waves at each (freq,mag)
  wave = np.zeros(n_samples, dtype=np.float32)
  
  time = np.linspace(0.0, n_samples/sample_rate, n_samples)
  
  
  for (freq,mag) in feature_slice:
    # not sure about sqrt etc
    sin = np.sqrt(mag)*np.sin(2*np.pi*time*freq)
    if np.any(sin > 1.0):
      pass
     # print("clipping:",sin)
    wave += sin
  return wave
  

# turn a feature representation into a waveform
def reconstruct(features, n_samples, sample_rate):
  return np.concatenate(np.array([reconstruct_slice(feature_slice, n_samples, sample_rate) for feature_slice  in features]))