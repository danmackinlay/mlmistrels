import autograd.numpy as np
from math import sqrt, exp, log


def x_dissonance_sethares(
        f1,
        f2,
        a1=1.0,
        a2=1.0,
        A=3.5,
        B=5.75,
        DSTAR=0.24,
        S1=0.21,
        S2=19):
    """
    Sethares' (1998) version of the Plomp and Levelt cross-dissonance.
    """
    df = abs(f2 - f1)
    s = DSTAR/(S1 * min(f1, f2) + S2)
    sdf = min(s * df, 20)
    return a1 * a2 * (exp(-A * sdf) - exp(-B * sdf))


def v_x_dissonance_sethares(
        f1,
        f2,
        a1=None,
        a2=None,
        A=3.5,
        B=5.75,
        DSTAR=0.24,
        S1=0.21,
        S2=19):
    """
    Sethares' (1998) version of the Plomp and Levelt cross-dissonance.
    Vector version.
    Note that cross-dissonance ignores self-dissonance;
    """
    if a1 is None:
        a1 = np.ones_like(f1)
    if a2 is None:
        a2 = np.ones_like(f2)
    f1 = f1.reshape(1, -1)
    a1 = a1.reshape(1, -1)
    f2 = f2.reshape(-1, 1)
    a2 = a2.reshape(-1, 1)
    df = abs(f2-f1)
    s = DSTAR/(S1 * np.minimum(f1, f2) + S2)
    sdf = np.minimum(s * df, 20.0)
    return (
        (a1 * a2) *
        (
            np.exp(-A * sdf) -
            np.exp(-B * sdf)
        )
    ).sum()


def v_dissonance_sethares(
        f,
        a=None,
        **kwargs
        ):
    """
    Full dissonance
    """
    if a is None:
        a = np.ones_like(f)
    return v_dissonance_sethares(f, f, a, a, **kwargs)


def max_dissonance_sethares(
        f,
        A=3.5,
        B=5.75,
        DSTAR=0.24,
        S1=0.21,
        S2=19
        ):
    """
    point of maximum dissonance
    """
    return f + (S1*f+S2)*(log(B)-log(A))/(DSTAR*(B-A))


def x_dissonance_parncutt(bk1, bk2, s1, s2):
    """
    this is another dissonance measure based on Parncutt and Barlow:
    D = sqrt(s1 * s2) * P(bk1 - bk2)
    where s1 & s2 are arrays in sones,
    bk1 & bk2 arrays in barks and
    P is the Parncutt dissonance measure
    dissmeasure2 {|bk1, bk2, s1, s2|
        var diss = 0, freqDiff, dnew;
        bk1.size.do{|i|
            bk2.size.do{|j|
                freqDiff = absdif(bk2[j], bk1[i]);
                dnew =  sqrt(s1[i] * s2[j]) *
                    (4 * freqDiff * (exp(1-(4 * freqDiff))));
                diss = diss + dnew;
            };
        };
        ^diss
    }
    """
    df = abs(bk1-bk2)
    return sqrt(
        s1 * s2
    ) * (
        4 * df * (exp(1.0 - (4.0 * df)))
    )

def v_x_dissonance_parncutt(bk1, bk2, s1, s2):
    raise NotImplementedError()
