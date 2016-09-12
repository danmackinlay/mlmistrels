import numpy as np
import librosa as lr

import encoder

from util import compress
from reconstruct import reconstruct

from easy_nn.easy_nn import train_nn

name = 'lamonte'
ext = 'mp3'
sr = 11025



audio, sr_ = lr.load('../samples/%s.%s'%(name,ext), sr = sr)



print (name, "loaded")

n_fft = 1024
hop_length = n_fft/4

H_pitch, H_pitch_mag = lr.piptrack(audio, sr = sr, n_fft = n_fft, hop_length = hop_length)

features = compress(H_pitch, H_pitch_mag, n_peaks = 16)

#corpus = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]*100)
corpus = features.reshape((len(features), -1))
scale_factor = np.max(np.abs(corpus))
corpus /= scale_factor

print (len(corpus), "samples")
memory = 50


def save_reconstruction(iteration, model, last = False):
    print ("callback", iteration)
    sentence = corpus[0: 0 + memory]
    generated = np.array(sentence)
    print('----- Generating with seed: "' + str(sentence) + '"')
    if last:
        n = len(corpus)*2
    else:
        n = len(corpus)//2
    p = n//10
    for i in range(n):
        if (i%p == 0):
            print(i,'/',n)
            #print(generated*scale_factor)
        x = np.array([sentence.flatten()])

        next_feature = model.predict(x, verbose=0)[0]

        generated = np.concatenate((generated, [next_feature]))
        sentence = np.concatenate((sentence[1:], [next_feature]))
    
    generated = generated.reshape((len(generated), -1, 2))*scale_factor
    print(generated)
    
    a = reconstruct(generated, n_fft = n_fft, sr = sr, hop_length = hop_length)
    if last:
        fn = '../output/%s_inspired_final.mp3'%(name,)
    else:
        fn = '../output/%s_inspired_%i.mp3'%(name, iteration)
    print("saving "+fn)
    lr.output.write_wav(fn, a, sr = sr)
    model.save('../models/%s.h5'%name)
    

model = train_nn(corpus, memory = memory, epochs = 2000, batch_size = 30, callback_step = 50, learn_rate = 0.001, callback = save_reconstruction)

print("done!")
