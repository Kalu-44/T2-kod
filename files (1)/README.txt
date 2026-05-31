==============================================================
 PROJEKAT: Cetvorokanalni TDMA + 16-QAM sistem
 Telekomunikacije 2 (13E033T2) - Oblast II, Tema 3
==============================================================

SADRZAJ PAKETA
--------------
  dsp_utils.py          - DSP pomocne funkcije (ekvivalenti MATLAB
                          funkcija: rcosdesign, upfirdn, awgn, fir1)
  qam_tdm.py            - 16-QAM mapiranje, TDM multipleks, kvadraturna
                          modulacija i demodulacija
  main_simulacija.py    - glavna simulaciona skripta (pokrece se ova)
  Izvestaj_TDMA_16QAM.docx - pisani izvestaj sa teorijom i rezultatima
  grafici/              - svi izlazni grafici (PNG) i rezultati.txt


POTREBNI PAKETI
---------------
  Python 3.9 ili noviji, sa bibliotekama:
    numpy, scipy, matplotlib

  Instalacija (ako nisu vec instalirane):
    pip install numpy scipy matplotlib


KAKO POKRENUTI
--------------
  Sve tri .py datoteke moraju biti u ISTOM direktorijumu.
  Iz tog direktorijuma pokrenuti:

    python main_simulacija.py

  Skripta ce:
    - generisati izvor za 4 kanala (256000 bita po kanalu),
    - sprovesti ceo lanac prenosa za Eb/N0 = 0..15 dB,
    - proceniti BER i uporediti sa teorijom,
    - nacrtati konstelacione dijagrame i SGSS,
    - analizirati greske sinhronizacije faze i vremena,
    - sve grafike sacuvati u poddirektorijum  figs/  ,
    - numericke rezultate upisati u  figs/rezultati.txt .

  Trajanje: oko 2-4 minuta (zavisno od racunara).


IZLAZNI GRAFICI
---------------
  01_ber_idealna_sinhr.png        - BER vs Eb/N0 (sim. vs teorija)
  02_konstelacija_15dB.png        - konstelacije 4 kanala @15 dB
  03_sgss_15dB.png                - SGSS modulisanog signala
  04_sinhr_greske_konstelacija.png- uticaj gresaka sinhronizacije
  05_ber_sa_greskama.png          - objedinjeni BER grafik
  06_tdm_poredjenje_spektar.png   - poredjenje dva nacina TDM-a


NAPOMENA O REPRODUKOVANJU REZULTATA
-----------------------------------
  Skripta koristi fiksiran seed slucajnog generatora (RNG, seed=2025),
  pa svako pokretanje daje iste rezultate. Za drugaciju realizaciju
  promeniti vrednost seed-a u main_simulacija.py (promenljiva RNG).

  Parametri se menjaju u sekciji "GLOBALNI PARAMETRI" na pocetku
  datoteke main_simulacija.py (npr. NSIM za broj simbola, EBN0_DB za
  opseg odnosa Eb/N0, ROLLOFFS za faktore zaobljenja, itd).
==============================================================
