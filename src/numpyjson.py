"""
Two handle jsonifying numpy arrays without tears
"""

import json
import numpy as np

def encode_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return json.JSONEncoder.default(None, obj)

def dumps(obj, *args, **kwargs):
    return json.dumps(obj, default=encode_numpy, *args, **kwargs)
