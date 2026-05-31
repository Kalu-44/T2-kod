# -*- coding: utf-8 -*-
"""
qam_tdm.py
----------
Jezgro simulacije: 16-QAM mapiranje/demapiranje, TDM multipleksiranje,
kvadraturna modulacija i demodulacija.

Telekomunikacije 2 - Oblast II, Tema 3 (TDMA + 16-QAM)
"""

import numpy as np


# ---------------------------------------------------------------------------
# 16-QAM konstelacija sa Gray-evim kodiranjem
# ---------------------------------------------------------------------------
# Nivoi po osi: 2-bitni Gray kod -> amplituda
#   00 -> -3 ,  01 -> -1 ,  11 -> +1 ,  10 -> +3
# 4 bita po simbolu: [b3 b2 b1 b0] ; (b3,b2)->I grana , (b1,b0)->Q grana
GRAY2LEVEL = {(0, 0): -3, (0, 1): -1, (1, 1): +1, (1, 0): +3}
LEVEL2GRAY = {v: k for k, v in GRAY2LEVEL.items()}

# faktor normalizacije: srednja energija 16-QAM simbola sa nivoima {+-1,+-3}
# E[|s|^2] = 2 * E[I^2] = 2 * (1+9)/2 = 10  ->  norm = sqrt(10)
QAM16_NORM = np.sqrt(10.0)


def bits_to_16qam(bits):
    """
    Mapira niz bita u niz kompleksnih 16-QAM simbola (Gray kodiranje).
    bits : 1-D niz {0,1}, duzina mora biti deljiva sa 4.
    Vraca : kompleksni niz, normalizovan na jedinicnu srednju energiju.
    """
    bits = np.asarray(bits, dtype=int).ravel()
    assert bits.size % 4 == 0, "Broj bita mora biti deljiv sa 4 (16-QAM)."
    groups = bits.reshape(-1, 4)
    I = np.array([GRAY2LEVEL[(g[0], g[1])] for g in groups], dtype=float)
    Q = np.array([GRAY2LEVEL[(g[2], g[3])] for g in groups], dtype=float)
    return (I + 1j * Q) / QAM16_NORM


def _nearest_level(x):
    """Odlucivanje na nivou simbola: najblizi od nivoa {-3,-1,+1,+3}."""
    levels = np.array([-3.0, -1.0, 1.0, 3.0])
    x = np.atleast_1d(x)
    idx = np.argmin(np.abs(x[:, None] - levels[None, :]), axis=1)
    return levels[idx]


def qam16_to_bits(symbols):
    """
    Demapira kompleksne 16-QAM simbole nazad u bite (odlucivanje na nivou
    simbola - optimalni pragovi za jednako verovatne simbole su 0 i +-2).
    """
    s = np.asarray(symbols) * QAM16_NORM     # vracamo na nivoe {+-1,+-3}
    I = _nearest_level(np.real(s))
    Q = _nearest_level(np.imag(s))
    bits = np.zeros((len(s), 4), dtype=int)
    for k in range(len(s)):
        b3, b2 = LEVEL2GRAY[I[k]]
        b1, b0 = LEVEL2GRAY[Q[k]]
        bits[k] = [b3, b2, b1, b0]
    return bits.ravel()


# ---------------------------------------------------------------------------
# TDM multipleksiranje
# ---------------------------------------------------------------------------
def tdm_mux(channels, sym_per_slot):
    """
    Vremensko multipleksiranje N kanala. U svakom 'slotu' uzima se
    'sym_per_slot' uzastopnih elemenata (16-QAM simbola) iz svakog kanala.

    channels     : lista od N nizova jednake duzine
    sym_per_slot : broj elemenata po kanalu u jednom TDM ciklusu
    Vraca        : multipleksiran niz.
    """
    N = len(channels)
    L = len(channels[0])
    assert all(len(c) == L for c in channels), "Kanali moraju biti jednake duzine."
    assert L % sym_per_slot == 0, "Duzina kanala mora biti deljiva sa sym_per_slot."
    n_slots = L // sym_per_slot
    out = np.empty(N * L, dtype=channels[0].dtype)
    pos = 0
    for slot in range(n_slots):
        a = slot * sym_per_slot
        b = a + sym_per_slot
        for ch in range(N):
            out[pos:pos + sym_per_slot] = channels[ch][a:b]
            pos += sym_per_slot
    return out


def tdm_demux(stream, N, sym_per_slot):
    """Inverzna operacija za tdm_mux - razdvaja multipleks na N kanala."""
    L = len(stream) // N
    n_slots = L // sym_per_slot
    channels = [np.empty(L, dtype=stream.dtype) for _ in range(N)]
    pos = 0
    for slot in range(n_slots):
        a = slot * sym_per_slot
        b = a + sym_per_slot
        for ch in range(N):
            channels[ch][a:b] = stream[pos:pos + sym_per_slot]
            pos += sym_per_slot
    return channels


# ---------------------------------------------------------------------------
# Kvadraturna (koherentna) modulacija i demodulacija
# ---------------------------------------------------------------------------
def quad_modulate(baseband, fc, fs):
    """
    Kvadraturni modulator:
        s(t) = I(t)*cos(2*pi*fc*t) - Q(t)*sin(2*pi*fc*t)
    baseband : kompleksni signal u osnovnom opsegu (I + jQ)
    fc, fs   : ucestanost nosioca i ucestanost odabiranja
    Vraca    : realni RF signal.
    """
    n = np.arange(len(baseband))
    c = np.cos(2 * np.pi * fc * n / fs)
    s = np.sin(2 * np.pi * fc * n / fs)
    return np.real(baseband) * c - np.imag(baseband) * s


def quad_demodulate(rf, fc, fs, phase_err=0.0):
    """
    Kvadraturni demodulator sa mogucnoscu unosenja greske faze nosioca.

    rf        : realni primljeni RF signal
    fc, fs    : ucestanost nosioca i odabiranja
    phase_err : greska sinhronizacije faze lokalnog nosioca [rad]

    Napomena: greska sinhronizacije u vremenu se NE modeluje ovde, vec u
    glavnoj simulaciji kao pomeranje trenutka odabiranja simbola posle
    uskladjenog filtra.

    Vraca : kompleksni signal u osnovnom opsegu (I + jQ).
    """
    n = np.arange(len(rf))
    c = np.cos(2 * np.pi * fc * n / fs + phase_err)
    s = np.sin(2 * np.pi * fc * n / fs + phase_err)
    # mnozenje sa 2 kompenzuje faktor 1/2 iz proizvoda nosilaca, tako da
    # se na grani dobijaju simboli na ispravnim (jedinicno-normalizovanim)
    # nivoima konstelacije. Posledica po sum se uracunava u SNR formuli.
    I = 2.0 * rf * c
    Q = -2.0 * rf * s
    return I + 1j * Q
