import librosa
from pathlib import Path
import json
from . import basicfilter
from .. import sfio
from ..metrics.harmonic import v_x_dissonance_sethares
import autograd.numpy as np
import autograd.scipy as sp
from autograd import grad
from autograd.util import flatten_func
from scipy.optimize import minimize
# from functools import lru_cache


# @lru_cache(128, typed=False)
def harmonic_synthesis(
        source_features,
        target_features,
        basis_size=8,
        gain_penalty=0.0,
        rate_penalty=0.0,
        rms_weight=100.0,
        dissonance_weight=1.0,
        debug=False,
        max_iters=100,
        **kwargs):
    """
    Reconstruct audio from descriptors based on approximate matching
    """
    if debug:
        from librosa.display import specshow
        import matplotlib.pyplot as plt

    gain = np.ones((1, basis_size))/basis_size
    rate = np.ones((1, basis_size))
    source_length = source_features['peak_f'].shape[1]

    target_peak_f = target_features['peak_f']
    target_peak_power = target_features['peak_power']

    start = np.random.randint(source_length, size=basis_size)

    source_peak_power = source_features['peak_power'][:, start]
    source_peak_f = source_features['peak_f'][:, start]

    source_power = source_features['rms'][:, start]
    target_power = target_features['rms']

    def reconstruct_peaks(gain, rate):
        reconstruction_peak_power = source_peak_power * gain * rate
        reconstruction_peak_f = source_peak_f * rate
        return reconstruction_peak_power, reconstruction_peak_f

    def reconstruct_power(gain, rate):
        return (gain * rate * source_power).sum()

    def dissonance_loss(gain, rate):
        reconstruct_peak_f, reconstruct_peak_power = reconstruct_peaks(
            gain, rate)
        return v_x_dissonance_sethares(
            reconstruct_peak_f, target_peak_f,
            reconstruct_peak_power, target_peak_power
        )

    def power_loss(gain, rate):
        return np.abs(
            reconstruct_power(gain, rate) -
            target_power
        ) ** 2

    def reconstruct_loss(gain, rate, rms_weight, dissonance_weight):
        print('diss loss', dissonance_loss(
            gain, rate
        ), 'power loss', power_loss(
            gain, rate
        ))
        return dissonance_weight * dissonance_loss(
            gain, rate
        ) + rms_weight * power_loss(
            gain, rate
        )

    def reconstruct_penalty(
            gain,
            rate,
            gain_penalty,
            rate_penalty):
        return gain_penalty * np.abs(
            gain
        ).sum() + rate_penalty * np.abs(
            np.log2(rate)
        ).sum()

    def objective(
            gain,
            rate,
            rms_weight,
            dissonance_weight,
            gain_penalty,
            rate_penalty
            ):
        return reconstruct_loss(
            gain,
            rate,
            rms_weight,
            dissonance_weight
        ) + reconstruct_penalty(
            gain,
            rate,
            gain_penalty,
            rate_penalty
        )

    def local_objective(params):
        gain, rate = params
        return objective(
            gain,
            rate,
            rms_weight,
            dissonance_weight,
            gain_penalty,
            rate_penalty
        )

    result = local_objective([gain, rate])
    fun, unflatten, flat_params = flatten_func(local_objective, [gain, rate])
    jac = grad(fun)

    result = minimize(
        fun,
        flat_params,
        method='BFGS',
        jac=grad(fun),
        # callback=callback_fun,
        options=dict(
            maxiter=max_iters,
            disp=True,
            gtol=1e-3
        )
    )
    return unflatten(result.x)
