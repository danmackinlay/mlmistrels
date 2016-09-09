"""
Brute force file IO using sox and FFMPEG.

Slow because of invoking external util,
but not as slow as me adding 24 bit support to audioread.

For alternative approaches, see Tensorflow's FFMPEG loader,
and if you don't need 24 bit support, the built-in librosa loader is fine,
even for MP3.
"""
import tempfile
import subprocess
import os
import os.path
import librosa
import pathlib
import logging
import base64

_tmp_dir = None

# Sox will do these well
sox_file_extensions = {"aif", "aiff", "wav", "wave"}
# For the rest, we fall back to ffmpeg.


def load(filename, sr=44100, mono=True, offset=0.0, duration=None, **kwargs):
    """
    We never use librosa's importer with default settings
    because it will erroenously load 24 bit aiffs as 16 bit wavs and explode
    without raising an error
    """
    extn = pathlib.Path(filename).suffix
    if len(extn):
        extn = str.lower(extn[1:])
    if extn in sox_file_extensions:
        return _load_sox(
            filename,
            sr=sr,
            mono=mono,
            offset=offset,
            duration=duration,
            **kwargs)
    else:
        return _load_ffmpeg(
            filename,
            sr=sr,
            mono=mono,
            offset=offset,
            duration=duration,
            **kwargs)


def save(filename, y, sr=44100, norm=True, **kwargs):
    # librosa saves using scipy, which makes fat 32-bit float wavs
    # these are big and inconvenient.
    return _save_sox(
        filename,
        y,
        sr=sr,
        norm=norm,
        **kwargs)


def _load_sox(
        filename,
        sr=44100,
        mono=True,
        offset=0.0,
        duration=None,
        **kwargs):
    """
    If our file is not readable, we can try sox to convert it
    to a normalised 24 bit wav
    and load it using scipy.
    We handle stero or mono here.
    """
    newfilename = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    command = [
        "sox",
        filename,
        "-b 24",
        newfilename,
        "norm",
    ]
    if mono:
        command = command + ["channels", "1"]
    if offset or duration:
        command = command + ["trim", str(offset), str(duration)]
    r = subprocess.run(command)
    r.check_returncode()
    wav, sr = librosa.load(
        newfilename,
        sr=sr,
        mono=mono,
        offset=offset,
        duration=duration)
    os.unlink(newfilename)
    return wav, sr


def _save_sox(
        filename,
        y,
        sr=44100,
        norm=True,
        **kwargs):
    """
    sox this into a normal audio format
    """
    newfilename = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    logging.info("outfile", newfilename)
    librosa.output.write_wav(
        newfilename, y,
        sr=sr, norm=norm)
    command = [
        "sox",
        newfilename,
        "-b 24",
        filename,
    ]
    r = subprocess.run(command)
    r.check_returncode()
    os.unlink(newfilename)


def _load_ffmpeg(
        filename,
        sr=44100,
        mono=True,
        offset=0.0,
        duration=None,
        **kwargs):
    """
    If our file is not readable, we can try ffmpeg to convert it
    to a normalised 24 bit wav
    and load it using scipy.
    We handle stereo or mono here.
    If our input file is 5.1 something will probably go nasty.
    """
    newfilename = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    command = [
        "ffmpeg"
    ]
    if offset or duration:
        command = command + [
            "-ss", str(offset),
            "-t", str(duration)
        ]
    command = command + [
        "-i", filename,
        # after input specification, we are talking about output files
        # "-af", "aresample=resampler=soxr",
        "-acodec", "pcm_s24le",
    ]
    if sr:
        command = command + [
            "-ar", str(sr),
        ]

    if mono:
        command = command + ["-ac", "1"]
    command = command + [
        "-y",  # force overwrite
        newfilename
    ]
    # logging.info("ff " + repr(command))
    r = subprocess.run(command)
    r.check_returncode()
    wav, sr = librosa.load(
        newfilename, sr=sr, mono=mono, offset=offset, duration=duration)
    os.unlink(newfilename)
    return wav, sr


def get_tmp_dir():
    global _tmp_dir
    if _tmp_dir is None:
        _tmp_dir = tempfile.mkdtemp(prefix="easy_listener")
    return _tmp_dir


def safeish_hash(obj):
    """
    return a short path-safe string hash of an object based on its repr value
    """
    return base64.urlsafe_b64encode(
        (
            hash(repr(obj)) % (2**32)
        ).to_bytes(4, byteorder='big', signed=False)
    ).decode("ascii")[:6]
