# -*- coding: utf-8 -*-
"""
dsp_utils.py
------------
Pomocne DSP funkcije koje reprodukuju ponasanje MATLAB/Octave funkcija
koriscenih u projektnom zadatku:

    rcosdesign  -> rrcosdesign()      (RRC FIR filtar)
    upfirdn     -> upfirdn()          (upsample - FIR - downsample)
    awgn        -> awgn()             (dodavanje belog Gaussovog suma)
    fir1        -> fir1()             (FIR LP/BP filtar projektovan prozorom)

Telekomunikacije 2 - Oblast II, Tema 3 (TDMA + 16-QAM)
"""

import numpy as np
from scipy.signal import lfilter, firwin


# ---------------------------------------------------------------------------
# 1) RRC (square-root raised cosine) filtar  ==  MATLAB rcosdesign(...,'sqrt')
# ---------------------------------------------------------------------------
def rrcosdesign(rolloff, span, sps):
    """
    Generise koeficijente FIR filtra sa korenom iz kosinusoidalnog zaobljenja
    (Square-Root Raised Cosine). Ekvivalent MATLAB poziva:
        b = rcosdesign(rolloff, span, sps, 'sqrt')

    Parametri
    ---------
    rolloff : faktor zaobljenja (0..1)
    span    : duzina filtra u simbolima
    sps     : broj odbiraka po simbolu

    Vraca
    -----
    b : 1-D niz koeficijenata, duzine span*sps+1, normalizovan na jedinicnu
        energiju (sum(b^2) = 1), kao u MATLAB-u.
    """
    N = span * sps
    n = np.arange(-N / 2, N / 2 + 1)          # span*sps+1 odbiraka
    t = n / sps                               # vreme u jedinicama simbola
    b = np.zeros_like(t, dtype=float)
    beta = rolloff

    for i, ti in enumerate(t):
        if abs(ti) < 1e-10:
            # granicni slucaj t = 0
            b[i] = (1.0 - beta + 4.0 * beta / np.pi)
        elif beta > 0 and abs(abs(4.0 * beta * ti) - 1.0) < 1e-10:
            # granicni slucaj t = +-Ts/(4*beta)
            b[i] = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            num = (np.sin(np.pi * ti * (1.0 - beta))
                   + 4.0 * beta * ti * np.cos(np.pi * ti * (1.0 + beta)))
            den = (np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2))
            b[i] = num / den

    # normalizacija na jedinicnu energiju (isto kao MATLAB rcosdesign)
    b = b / np.sqrt(np.sum(b ** 2))
    return b


# ---------------------------------------------------------------------------
# 2) upfirdn  ==  MATLAB upfirdn(x, b, p, q)
# ---------------------------------------------------------------------------
def upfirdn(x, b, up=1, down=1):
    """
    Reprodukuje MATLAB funkciju upfirdn:
      - upsample signala x faktorom 'up' (umetanje up-1 nula),
      - FIR filtriranje koeficijentima b,
      - downsample faktorom 'down'.

    Koristi se:
      * na predaji za uoblicavanje impulsa:  upfirdn(x, b, sps)
      * na prijemu za uskladjeni filtar:     upfirdn(x, b, 1, sps)
    """
    x = np.asarray(x, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()

    # upsample
    if up > 1:
        xu = np.zeros(len(x) * up, dtype=float)
        xu[::up] = x
    else:
        xu = x

    # FIR filtriranje (puna konvolucija - isto kao MATLAB)
    y = np.convolve(xu, b)

    # downsample
    if down > 1:
        y = y[::down]

    return y


# ---------------------------------------------------------------------------
# 3) awgn  ==  MATLAB awgn(x, snr_db, 'measured')
# ---------------------------------------------------------------------------
def awgn(x, snr_db, rng=None):
    """
    Dodaje beli Gaussov sum signalu x tako da odnos signal/sum bude snr_db [dB].
    Ekvivalent MATLAB poziva awgn(x, snr_db, 'measured') - snaga signala se
    MERI iz samog signala.

    Radi i za realne i za kompleksne signale.
    """
    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(x)

    sig_power = np.mean(np.abs(x) ** 2)        # izmerena snaga signala
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / snr_lin

    if np.iscomplexobj(x):
        # kompleksan sum: snaga se deli na realni i imaginarni deo
        noise = (np.sqrt(noise_power / 2.0)
                 * (rng.standard_normal(x.shape)
                    + 1j * rng.standard_normal(x.shape)))
    else:
        noise = np.sqrt(noise_power) * rng.standard_normal(x.shape)

    return x + noise


# ---------------------------------------------------------------------------
# 4) fir1  ==  MATLAB fir1(N, Wn[, 'low'/'bandpass'])
# ---------------------------------------------------------------------------
def fir1(numtaps, Wn, pass_zero=True):
    """
    FIR filtar projektovan metodom prozora (Hamming), ekvivalent MATLAB fir1.

    numtaps   : red filtra N -> filtar ima N+1 koeficijenata (kao MATLAB)
    Wn        : normalizovana ucestanost(i) preseka (1.0 = Fs/2)
                - skalar  -> NF filtar
                - [w1,w2] -> PO (bandpass) filtar
    pass_zero : True  -> propusnik niskih ucestanosti
                False -> propusnik opsega
    """
    return firwin(numtaps + 1, Wn, pass_zero=pass_zero, window='hamming')


def apply_fir(b, x):
    """Filtriranje signala x FIR filtrom b (MATLAB filter(b,1,x))."""
    return lfilter(b, [1.0], x)
