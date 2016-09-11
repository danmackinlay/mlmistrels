import numpy as np
import librosa as lr

import encoder

from descriptors.util import compress
from descriptors.reconstruct import reconstruct

from keras_stuff.keras_rnn import train_rnn
from keras_stuff.keras_nn import train_nn

corpus = np.array([[0.0, 10.0], [0.0, 10.0], [1.0, 0.0]]*100)
scale_factor = np.max(np.abs(corpus))
corpus /= scale_factor
memory = 50

print (len(corpus), "samples")
model = train_nn(corpus, memory = memory, epochs = 200, batch_size = 1000, print_step = 30)

sentence = corpus[0: 0 + memory]
generated = np.array(sentence)
print('----- Generating with seed: "' + str(sentence) + '"')

p = len(corpus)//100
for i in range(len(corpus)):
    if (i%p == 0):
        print("i =",i)
        print(generated*scale_factor)
    x = np.array([sentence.flatten()])

    preds = model.predict(x, verbose=0)[0]
    next_feature = preds

    generated = np.concatenate((generated, [next_feature]))
    sentence = np.concatenate((sentence[1:], [next_feature]))

    #print(next_feature)
print("done!")
        
# name = '303_bach'
#
# audio,sr = lr.load('../samples/%s.wav'%name)
#
# print(audio.shape)
#
# n_fft = 2048
# hop_length = n_fft/4
#
# H_pitch, H_pitch_mag = lr.piptrack(audio, sr = sr, n_fft = n_fft, hop_length = hop_length)
#
# features = compress(H_pitch, H_pitch_mag, n_peaks = 16)
#
# print("features.shape=",features.shape)
#
# recon = reconstruct(features, n_fft = n_fft, sr = sr, hop_length = hop_length)
#
# recon = recon.astype(np.float32)
#
# audio = lr.output.write_wav('%s_reconstructed.wav'%name, recon, sr = sr)