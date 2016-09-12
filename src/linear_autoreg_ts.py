#!/usr/local/bin/python3

from encoder import *
import librosa 
import scipy.optimize
import sfio
from pathlib import Path
import pdb

here = Path(__file__).resolve().parent.parent

sound_name = '303_bach'

samples_dir = here.joinpath('samples')
output_dir = here.joinpath('output')

snd, sr = sfio.load(str(samples_dir.joinpath('%s.mp3' % sound_name)))

diagnostic=False
win_len = 2048 

#sparse_enc = SparseEncode(win_len)
bin_enc = BinEncode(win_len)

d = bin_enc.encode(snd)

# Simplest possible ML - linear auto-regression, high dimensional though!

# Construct the lower triangular "auto-" design matrix (i.e. next sample dep. on last sample)
A = np.dstack([np.pad(d[:,i::-1], ((0,0),(0,d.shape[1]-i-2)), 'constant') for i in range(d.shape[1]-1)])

def auto_reg(x):
    
    # This gets us the right tensor multiplication
    y = np.tensordot(A, x, axes=([1],[0]))
    return np.abs((y - d[:,1:]).flatten())


x0 = np.random.random(A.shape[1])
x = scipy.optimize.leastsq(auto_reg, x0) 

print(x)
pdb.set_trace()
# Normalise
snd_bin = snd_bin / snd_bin.max()
librosa.output.write_wav('../output/303_bach_bin_encoded.wav', snd_bin, sr)



