# -*- coding: utf-8 -*-
# Telekomunikacije 2 - Oblast II, Tema 3
# Cetvorokanalni TDMA + 16-QAM sistem: generisanje signala, prenos kroz ABGS
# kanal sa kvadraturnom modulacijom, procena BER i SGSS, i analiza uticaja
# gresaka sinhronizacije faze nosioca i sinhronizacije u vremenu.

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import upfirdn, firwin, filtfilt, freqz
from scipy.special import erfc


broj_kanala     = 4
broj_bita       = 256000
amplituda       = 1.0
protok_simbola  = 512
sps_referentno  = 64
fs              = sps_referentno * protok_simbola
fc              = fs / 4
span            = 6
bita_po_simbolu = 4
blok_simbola    = 8
ebn0_db         = np.arange(0, 16, 3)
rolloff_lista   = [0.5, 1.0]
ebn0_const      = 15
ebn0_sync       = 25
greska_faze     = np.pi / 6
greska_vremena  = 4
nfft            = 4096
seme            = 12345

# Korekcija za prenos na realnoj nosecoj: srednja snaga pojasnog signala je
# upola manja od kompleksnog baznog opsega, pa se SNR koriguje za 10*log10(2).
passband_korekcija_db = 10 * np.log10(2)

# Dva nacina formiranja TDM signala (razlikuju se po broju odbiraka po simbolu
# pri fiksnoj ucestanosti odabiranja, tj. po sirini spektra).
tdm_slucajevi = {
    "A: TDM sa kompresijom (Rs=2048)": sps_referentno // broj_kanala,
    "B: TDM bez kompresije (Rs=512)":  sps_referentno,
}

generator_izvora = np.random.default_rng(seme)

gray_nivoi    = np.array([-3.0, -1.0, 3.0, 1.0])
nivoi_sortir  = np.array([-3, -1, 1, 3])
nivo_u_par    = np.array([0, 1, 3, 2])


def bits_to_16qam(bita):
    # Mapira matricu bita oblika (broj_simbola, 4) u kompleksne 16-QAM simbole (Gray).
    indeks_i = 2 * bita[:, 0] + bita[:, 1]
    indeks_q = 2 * bita[:, 2] + bita[:, 3]
    return gray_nivoi[indeks_i] + 1j * gray_nivoi[indeks_q]


def odluci_nivo(vrednost):
    # Odlucuje koji je nivo iz {-3,-1,1,3} najblizi datoj vrednosti po jednoj osi.
    return np.where(vrednost < -2, -3,
                    np.where(vrednost < 0, -1,
                             np.where(vrednost < 2, 1, 3)))


def qam16_to_bits(simboli):
    # Demapira primljene kompleksne 16-QAM simbole nazad u matricu bita (broj_simbola, 4).
    nivo_i = odluci_nivo(simboli.real).astype(int)
    nivo_q = odluci_nivo(simboli.imag).astype(int)
    par_i = nivo_u_par[(nivo_i + 3) // 2]
    par_q = nivo_u_par[(nivo_q + 3) // 2]
    bita = np.empty((len(simboli), 4), dtype=np.int8)
    bita[:, 0] = (par_i >> 1) & 1
    bita[:, 1] = par_i & 1
    bita[:, 2] = (par_q >> 1) & 1
    bita[:, 3] = par_q & 1
    return bita


def rcosdesign(rolloff, span, sps):
    # Generise koeficijente RRC filtra (koren iz kosinusnog zaobljenja) jedinicne energije.
    duzina = span * sps
    vreme = (np.arange(duzina + 1) - duzina / 2.0) / sps
    odziv = np.zeros_like(vreme)

    nula = np.isclose(vreme, 0.0)
    odziv[nula] = 1.0 - rolloff + 4.0 * rolloff / np.pi

    if rolloff > 0:
        singularno_vreme = 1.0 / (4.0 * rolloff)
        singularno = np.isclose(np.abs(vreme), singularno_vreme)
        odziv[singularno] = (rolloff / np.sqrt(2.0)) * (
            (1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff)) +
            (1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff))
        )
    else:
        singularno = np.zeros_like(vreme, dtype=bool)

    ostalo = ~(nula | singularno)
    vreme_ostalo = vreme[ostalo]
    brojilac = (np.sin(np.pi * vreme_ostalo * (1 - rolloff))
                + 4 * rolloff * vreme_ostalo * np.cos(np.pi * vreme_ostalo * (1 + rolloff)))
    imenilac = np.pi * vreme_ostalo * (1 - (4 * rolloff * vreme_ostalo) ** 2)
    odziv[ostalo] = brojilac / imenilac

    return odziv / np.sqrt(np.sum(odziv ** 2))


def awgn_measured(signal, snr_db, generator):
    # Dodaje realni beli Gausov sum sa izmerenom snagom signala za zadati odnos S/N.
    snaga_signala = np.mean(signal ** 2)
    snaga_suma = snaga_signala / (10 ** (snr_db / 10))
    return signal + np.sqrt(snaga_suma) * generator.standard_normal(signal.shape)


def make_bandpass(rolloff, sps, broj_koeficijenata=201):
    # Pravi ekvivalentni propusnik opsega cija je sirina dva puta veca od spektra signala.
    protok_linije = fs / sps
    poluopseg = protok_linije * (1 + rolloff)
    donja = max(fc - poluopseg, 1.0)
    gornja = min(fc + poluopseg, fs / 2 - 1.0)
    if broj_koeficijenata % 2 == 0:
        broj_koeficijenata += 1
    return firwin(broj_koeficijenata, [donja, gornja], pass_zero=False, fs=fs), (donja, gornja)


def run_link(tdm_simboli, sps, rolloff, ebn0_db, generator,
             greska_faze=0.0, greska_vremena=0, vrati_pojasni=False):
    # Prolaz kroz ceo lanac (predaja, ABGS, prijem) i vraca odabrane primljene simbole.
    rrc = rcosdesign(rolloff, span, sps)
    broj_simbola = len(tdm_simboli)

    bazni = upfirdn(rrc, tdm_simboli, sps, 1)
    indeks = np.arange(len(bazni))
    omega = 2 * np.pi * fc / fs
    pojasni = np.real(bazni * np.exp(1j * omega * indeks))

    snr_db = (ebn0_db + 10 * np.log10(bita_po_simbolu)
              - 10 * np.log10(sps) + passband_korekcija_db)
    sa_sumom = awgn_measured(pojasni, snr_db, generator)

    bpf, _ = make_bandpass(rolloff, sps)
    filtrirano = filtfilt(bpf, [1.0], sa_sumom)

    nosilac = 2.0 * np.exp(-1j * (omega * np.arange(len(filtrirano)) + greska_faze))
    demodulisano = filtrirano * nosilac

    usaglaseno = upfirdn(rrc, demodulisano, 1, 1)

    pocetak = span * sps + greska_vremena
    indeksi_odabiranja = pocetak + sps * np.arange(broj_simbola)
    primljeni = usaglaseno[indeksi_odabiranja]

    if vrati_pojasni:
        return primljeni, filtrirano
    return primljeni


def tdm_mux(simboli_po_kanalu):
    # Vremenski multipleksira kanale preplitanjem blokova od blok_simbola simbola.
    broj = len(simboli_po_kanalu)
    simbola_po_kanalu = len(simboli_po_kanalu[0])
    broj_blokova = simbola_po_kanalu // blok_simbola
    matrica = np.stack(simboli_po_kanalu, axis=0)
    matrica = matrica[:, :broj_blokova * blok_simbola]
    matrica = matrica.reshape(broj, broj_blokova, blok_simbola)
    matrica = np.transpose(matrica, (1, 0, 2))
    return matrica.reshape(-1), broj_blokova


def tdm_demux(tdm_simboli, broj_blokova):
    # Razdvaja multipleksirani niz nazad na simbole po kanalima.
    matrica = tdm_simboli.reshape(broj_blokova, broj_kanala, blok_simbola)
    matrica = np.transpose(matrica, (1, 0, 2))
    return matrica.reshape(broj_kanala, -1)


def theory_ber_16qam(ebn0_db):
    # Teorijska verovatnoca greske po bitu za Gray-kodiran 16-QAM u ABGS kanalu.
    ebn0 = 10 ** (ebn0_db / 10)
    return (3.0 / 8.0) * erfc(np.sqrt(0.4 * ebn0))


def psd_bartlett(signal, nfft=nfft, fs=fs):
    # Procenjuje SGSS metodom usrednjenog periodograma (Bartlett) nad nepreklapajucim segmentima.
    broj_segmenata = len(signal) // nfft
    if broj_segmenata == 0:
        raise ValueError("Signal je kraci od NFFT.")
    segmenti = signal[:broj_segmenata * nfft].reshape(broj_segmenata, nfft)
    periodogrami = np.abs(np.fft.fft(segmenti, nfft, axis=1)) ** 2 / (fs * nfft)
    sgss = np.fft.fftshift(np.mean(periodogrami, axis=0))
    frekvencije = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))
    return frekvencije, sgss


def main(broj_bita=broj_bita, izlazni_dir="."):
    # Pokrece celu simulaciju za oba TDM slucaja i snima sve trazene grafike.
    print(f"Fs = {fs} Hz, fc = {fc} Hz, bita/kanal = {broj_bita}")
    simbola_po_kanalu = broj_bita // bita_po_simbolu
    simbola_po_kanalu -= simbola_po_kanalu % blok_simbola
    print(f"16-QAM simbola po kanalu = {simbola_po_kanalu}")

    bita_po_kanalu = []
    simboli_po_kanalu = []
    for _ in range(broj_kanala):
        bita = (generator_izvora.random(simbola_po_kanalu * bita_po_simbolu) < 0.5).astype(np.int8)
        bita_po_kanalu.append(bita)
        simboli_po_kanalu.append(bits_to_16qam(bita.reshape(-1, 4)))
    tdm_simboli, broj_blokova = tdm_mux(simboli_po_kanalu)

    ber_rezultati = {}
    sgss_rezultati = {}
    konstelacija = {}
    sync_rezultati = {}
    teorija = theory_ber_16qam(ebn0_db)

    for naziv_slucaja, sps in tdm_slucajevi.items():
        ber_rezultati[naziv_slucaja] = {}
        sgss_rezultati[naziv_slucaja] = {}
        sync_rezultati[naziv_slucaja] = {}
        for rolloff in rolloff_lista:
            print(f"\n=== {naziv_slucaja} | rolloff={rolloff} | sps={sps} ===")
            ber_ukupno = []
            for ebn0 in ebn0_db:
                cuvaj_pojasni = (ebn0 == ebn0_const)
                generator = np.random.default_rng(seme + int(ebn0))
                izlaz = run_link(tdm_simboli, sps, rolloff, ebn0, generator,
                                 vrati_pojasni=cuvaj_pojasni)
                if cuvaj_pojasni:
                    primljeni, filtrirano = izlaz
                    sgss_rezultati[naziv_slucaja][rolloff] = psd_bartlett(filtrirano)
                else:
                    primljeni = izlaz

                primljeni_po_kanalu = tdm_demux(primljeni, broj_blokova)
                greske_kanala = []
                for kanal in range(broj_kanala):
                    primljeni_bita = qam16_to_bits(primljeni_po_kanalu[kanal]).reshape(-1)
                    greske_kanala.append(np.mean(primljeni_bita != bita_po_kanalu[kanal]))
                ber_ukupno.append(np.mean(greske_kanala))

                if cuvaj_pojasni and sps == sps_referentno:
                    konstelacija[rolloff] = primljeni_po_kanalu.copy()

                print(f"  Eb/N0={ebn0:2d} dB  BER={np.mean(greske_kanala):.3e}  "
                      f"(teorija {theory_ber_16qam(ebn0):.3e})")

            ber_rezultati[naziv_slucaja][rolloff] = np.array(ber_ukupno)

            poslati_bita = qam16_to_bits(tdm_simboli).reshape(-1)
            generator = np.random.default_rng(seme + 999)
            primljeni_faza = run_link(tdm_simboli, sps, rolloff, ebn0_sync, generator,
                                      greska_faze=greska_faze)
            generator = np.random.default_rng(seme + 998)
            primljeni_vreme = run_link(tdm_simboli, sps, rolloff, ebn0_sync, generator,
                                       greska_vremena=greska_vremena)
            ber_faza = np.mean(qam16_to_bits(primljeni_faza).reshape(-1) != poslati_bita)
            ber_vreme = np.mean(qam16_to_bits(primljeni_vreme).reshape(-1) != poslati_bita)
            sync_rezultati[naziv_slucaja][rolloff] = dict(faza=ber_faza, vreme=ber_vreme)
            print(f"  [25 dB] greska faze pi/6: BER={ber_faza:.3e} | "
                  f"greska vremena {greska_vremena} odb.: BER={ber_vreme:.3e}")

    nacrtaj_ber(ber_rezultati, sync_rezultati, izlazni_dir)
    nacrtaj_konstelaciju(konstelacija, izlazni_dir)
    nacrtaj_sgss(sgss_rezultati, izlazni_dir)
    nacrtaj_sync_konstelaciju(tdm_simboli, izlazni_dir)
    nacrtaj_filtre(izlazni_dir)

    print("\nGotovo. Figure su snimljene u:", izlazni_dir)
    return ber_rezultati, sync_rezultati


def nacrtaj_ber(ber_rezultati, sync_rezultati, izlazni_dir):
    # Crta procenjeni BER u funkciji Eb/N0 za oba slucaja, sa teorijom i tackama gresaka sinhr.
    figura, ose = plt.subplots(1, len(rolloff_lista), figsize=(13, 5.5), sharey=True)
    ebn0_gusto = np.linspace(0, ebn0_sync, 200)
    markeri = {"A: TDM sa kompresijom (Rs=2048)": 'o',
               "B: TDM bez kompresije (Rs=512)": 's'}
    for osa, rolloff in zip(ose, rolloff_lista):
        osa.semilogy(ebn0_gusto, theory_ber_16qam(ebn0_gusto), 'k--', label='Teorija 16-QAM')
        for naziv_slucaja in ber_rezultati:
            ber = np.asarray(ber_rezultati[naziv_slucaja][rolloff], dtype=float)
            ber = np.where(ber > 0, ber, np.nan)
            osa.semilogy(ebn0_db, ber, marker=markeri.get(naziv_slucaja, 'x'),
                         label=naziv_slucaja.split(':')[0] + " (idealno)")
            sync = sync_rezultati[naziv_slucaja][rolloff]
            osa.semilogy(ebn0_sync, sync['faza'], marker='^', linestyle='none',
                         markersize=10, label=f"{naziv_slucaja.split(':')[0]} faza pi/6")
            osa.semilogy(ebn0_sync, sync['vreme'], marker='v', linestyle='none',
                         markersize=10, label=f"{naziv_slucaja.split(':')[0]} vreme {greska_vremena} odb.")
        osa.set_title(f"BER, rolloff = {rolloff}")
        osa.set_xlabel("Eb/N0 [dB]")
        osa.grid(True, which='both', alpha=0.3)
        osa.set_ylim(1e-6, 1)
        osa.legend(fontsize=7, loc='lower left')
    ose[0].set_ylabel("Pe,b (verovatnoca greske po bitu)")
    figura.suptitle("Verovatnoca greske po bitu (TDMA + 16-QAM)")
    figura.tight_layout()
    figura.savefig(f"{izlazni_dir}/01_BER.png", dpi=130)
    plt.close(figura)


def nacrtaj_konstelaciju(konstelacija, izlazni_dir):
    # Crta konstelacioni dijagram na prijemu za sva 4 kanala uz idealne pozicije simbola.
    if not konstelacija:
        return
    idealne = np.array([osa_i + 1j * osa_q for osa_i in nivoi_sortir for osa_q in nivoi_sortir])
    for rolloff, primljeni in konstelacija.items():
        figura, ose = plt.subplots(2, 2, figsize=(9, 9))
        for kanal, osa in enumerate(ose.ravel()):
            uzorak = primljeni[kanal][:4000]
            osa.scatter(uzorak.real, uzorak.imag, s=3, alpha=0.3, color='C0')
            osa.scatter(idealne.real, idealne.imag, s=80, marker='x',
                        color='red', label='idealne pozicije')
            osa.set_title(f"Kanal {kanal + 1}")
            osa.set_xlabel("I"); osa.set_ylabel("Q")
            osa.grid(True, alpha=0.3)
            osa.axhline(0, color='k', lw=0.5); osa.axvline(0, color='k', lw=0.5)
            osa.set_aspect('equal'); osa.set_xlim(-5, 5); osa.set_ylim(-5, 5)
            if kanal == 0:
                osa.legend(fontsize=8)
        figura.suptitle(f"Konstelacioni dijagram na prijemu, Eb/N0={ebn0_const} dB "
                        f"(slucaj B, rolloff={rolloff})")
        figura.tight_layout()
        figura.savefig(f"{izlazni_dir}/02_konstelacija_rolloff{rolloff}.png", dpi=130)
        plt.close(figura)


def nacrtaj_sgss(sgss_rezultati, izlazni_dir):
    # Crta procenjenu SGSS modulisanog signala na prijemu za oba TDM slucaja.
    figura, ose = plt.subplots(1, len(rolloff_lista), figsize=(13, 5))
    for osa, rolloff in zip(ose, rolloff_lista):
        for naziv_slucaja in sgss_rezultati:
            if rolloff in sgss_rezultati[naziv_slucaja]:
                frekvencije, sgss = sgss_rezultati[naziv_slucaja][rolloff]
                osa.plot(frekvencije, 10 * np.log10(sgss + 1e-20),
                         label=naziv_slucaja.split(':')[0])
        osa.set_title(f"SGSS na prijemu, rolloff={rolloff}, Eb/N0={ebn0_const} dB")
        osa.set_xlabel("f [Hz]"); osa.set_ylabel("SGSS [dB/Hz]")
        osa.grid(True, alpha=0.3); osa.legend(fontsize=8)
        osa.set_xlim(0, fs / 2)
    figura.tight_layout()
    figura.savefig(f"{izlazni_dir}/03_SGSS.png", dpi=130)
    plt.close(figura)


def nacrtaj_sync_konstelaciju(tdm_simboli, izlazni_dir):
    # Crta konstelaciju pri idealnoj sinhronizaciji i pri greskama faze i vremena na 25 dB.
    sps = sps_referentno
    rolloff = 0.5
    generator = np.random.default_rng(seme + 1)
    primljeni_idealno = run_link(tdm_simboli, sps, rolloff, ebn0_sync, generator)
    generator = np.random.default_rng(seme + 2)
    primljeni_faza = run_link(tdm_simboli, sps, rolloff, ebn0_sync, generator, greska_faze=greska_faze)
    generator = np.random.default_rng(seme + 3)
    primljeni_vreme = run_link(tdm_simboli, sps, rolloff, ebn0_sync, generator, greska_vremena=greska_vremena)

    idealne = np.array([osa_i + 1j * osa_q for osa_i in nivoi_sortir for osa_q in nivoi_sortir])
    naslovi = ["Idealna sinhronizacija", "Greska faze = pi/6",
               f"Greska vremena = {greska_vremena} odbirka"]
    podaci = [primljeni_idealno, primljeni_faza, primljeni_vreme]
    figura, ose = plt.subplots(1, 3, figsize=(15, 5))
    for osa, primljeni, naslov in zip(ose, podaci, naslovi):
        uzorak = primljeni[:4000]
        osa.scatter(uzorak.real, uzorak.imag, s=3, alpha=0.3)
        osa.scatter(idealne.real, idealne.imag, s=80, marker='x', color='red')
        osa.set_title(naslov); osa.set_aspect('equal'); osa.grid(True, alpha=0.3)
        osa.set_xlim(-5, 5); osa.set_ylim(-5, 5)
        osa.set_xlabel("I"); osa.set_ylabel("Q")
    figura.suptitle(f"Uticaj gresaka sinhronizacije, Eb/N0={ebn0_sync} dB (slucaj B)")
    figura.tight_layout()
    figura.savefig(f"{izlazni_dir}/04_sync_konstelacija.png", dpi=130)
    plt.close(figura)


def nacrtaj_filtre(izlazni_dir):
    # Crta impulsni odziv RRC filtra i frekvencijski odziv ekvivalentnog propusnika opsega.
    figura, ose = plt.subplots(1, 2, figsize=(13, 5))
    for rolloff in rolloff_lista:
        for oznaka, sps in [("sps=64", sps_referentno), ("sps=16", sps_referentno // broj_kanala)]:
            ose[0].plot(rcosdesign(rolloff, span, sps), label=f"rolloff={rolloff}, {oznaka}")
    ose[0].set_title("RRC impulsni odziv (rcosdesign 'sqrt')")
    ose[0].set_xlabel("odbirak"); ose[0].grid(True, alpha=0.3); ose[0].legend(fontsize=7)
    for naziv_slucaja, sps in tdm_slucajevi.items():
        for rolloff in rolloff_lista:
            bpf, _ = make_bandpass(rolloff, sps)
            frekvencije, odziv = freqz(bpf, worN=4096, fs=fs)
            ose[1].plot(frekvencije, 20 * np.log10(np.abs(odziv) + 1e-9),
                        label=f"{naziv_slucaja.split(':')[0]}, r={rolloff}")
    ose[1].set_title("Ekvivalentni BPF (frekvencijski odziv)")
    ose[1].set_xlabel("f [Hz]"); ose[1].set_ylabel("|H| [dB]")
    ose[1].set_xlim(0, fs / 2); ose[1].set_ylim(-80, 5)
    ose[1].grid(True, alpha=0.3); ose[1].legend(fontsize=7)
    figura.tight_layout()
    figura.savefig(f"{izlazni_dir}/05_filtri.png", dpi=130)
    plt.close(figura)


if __name__ == "__main__":
    main()
