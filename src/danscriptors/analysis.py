import librosa
from pathlib import Path
import numpy as np
import json
from . import basicfilter
from . import sfio
# from functools import lru_cache
from scipy.ndimage import median_filter
import math


# @lru_cache(128, typed=False)
def harmonic_index(
        sourcefile,
        offset=0.0,
        duration=120.0,
        key=None,
        output_dir=None,
        n_fft=4096,
        hop_length=1024,
        pitch_median=20,  # how many frames for running media?
        high_pass_f=40.0,
        low_pass_f=4000.0,
        pitch_floor=-60,
        debug=False,
        cached=True,
        n_pitches=16,
        **kwargs):
    """
    Index spectral peaks
    """
    if debug:
        from librosa.display import specshow
        import matplotlib.pyplot as plt
    # args that will make a difference to content,
    # apart from the sourcefile itself
    argset = dict(
        analysis="harmonic_index",
        # sourcefile=sourcefile,
        offset=offset,
        duration=duration,
        n_fft=n_fft,
        hop_length=hop_length,
        high_pass_f=high_pass_f,
        low_pass_f=low_pass_f,
        pitch_median=pitch_median,
        pitch_floor=pitch_floor,
        n_pitches=n_pitches,
    )
    sourcefile = Path(sourcefile).resolve()
    if output_dir is None:
        output_dir = sourcefile.parent
    output_dir = Path(output_dir)

    if key is None:
        key = str(sourcefile.stem) + "___" + sfio.safeish_hash(argset)

    metadatafile = (output_dir/key).with_suffix(".json")
    if cached and metadatafile.exists():
        return json.load(metadatafile.open("r"))

    metadata = dict(
        key=key,
        args=argset,
        metadatafile=str(metadatafile),
    )
    y, sr = sfio.load(
        str(sourcefile), sr=None,
        mono=True,
        offset=offset, duration=duration
    )

    if high_pass_f is not None:
        y = basicfilter.high_passed(y, sr, high_pass_f)

    dur = librosa.get_duration(y=y, sr=sr)

    metadata["dur"] = dur
    metadata["sr"] = sr
    # convert to spectral frames
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    # Separate into harmonic and percussive. I think this preserves phase?
    H, P = librosa.decompose.hpss(D)
    # Resynthesize the harmonic component as waveforms
    y_harmonic = librosa.istft(H)
    harmonicfile = str(output_dir/key) + ".harmonic.wav"
    # sfio.save(
    #     harmonicfile,
    #     y_harmonic, sr=sr, norm=True)
    metadata["harmonicfile"] = harmonicfile

    # Now, power spectrogram
    H_mag, H_phase = librosa.magphase(H)
    y_harmonic_rms = librosa.feature.rmse(
        S=H_mag
    )

    H_pitch, H_pitch_mag = librosa.piptrack(
        S=H_mag, sr=sr, fmin=high_pass_f, fmax=low_pass_f,
        threshold=10**(pitch_floor/20.0))

    H_pitch_amp = np.real(H_pitch_mag**2)

    pitch_mag_floor = np.sort(H_pitch_amp, axis=0)[:, n_pitches:n_pitches+1]
    pitch_mag_mask = features['H_pitch_amp']>pitch_mag_floor
    
    if debug:
        plt.figure()
        specshow(
            librosa.logamplitude(np.abs(H_pitch_mag)**2, ref_power=np.max),
            y_axis='log',
            sr=sr)
        plt.title('Pitch Spect')

    pitch_mask = H_pitch > 0
    strong_pitch_mag = pitch_mask * H_pitch_mag
    # How much energy in pitches?
    y_pitch_mag_rms = librosa.feature.rmse(S=H_pitch_mag)

    json.dump(metadata, metadatafile.open("w"))

    peak_f = None
    peak_mag_sq = None

    return dict(
        metadata=metadata,
        peak_f=peak_f,
        peak_mag_sq=peak_mag_sq,
        H_pitch=H_pitch,
        H_pitch_amp=H_pitch_amp,
    )
