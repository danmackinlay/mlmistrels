"""
Audition audio reconstruction from Supercollider, since I do not want to write
granular synthesis in python.
"""
from liblo import send, Address

ADDRESS = Address('localhost', 57120)

def file(filename):
    send(ADDRESS, '/file', filename)

def note(start_time, gain, rate, **kwargs):
    ns = list(start_time.ravel()) + list(gain.ravel()) + list(rate.ravel())
    floats = [
        float(f) for f in ns
    ]
    send(ADDRESS, '/note', *floats)
