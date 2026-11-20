"""
szukaj_cpu — wykrywanie osób (i innych obiektów) na żywo z kamer RTSP.

Pod: Debian 13, Intel i5-7500 (iGPU HD 630), bez dedykowanego GPU.

  kamera RTSP ──> ffmpeg (1 połączenie na kamerę)
                   ├─ kopia strumienia -> bufor segmentów w RAM (do wycinania klipów)
                   └─ dekodowanie całego strumienia na iGPU (VAAPI), wybór 1 klatki co N s
                      (albo tylko klatki kluczowe), pobranie do RAM i skalowanie na CPU
                        -> YOLO na CPU (OpenVINO lub PyTorch)
                           -> zdarzenia -> miniatury, klipy, wyniki.csv, index.html
                           -> (opcjonalnie) powiadomienie ntfy
                           -> (opcjonalnie) zdarzenie na osi czasu Shinobi (API /motion)

Instalacja: /opt/szukaj_cpu (pakiet .deb), usługa systemd szukaj_cpu@<użytkownik>.service.
Konfiguracja: ~/.szukaj_cpu/szukaj_cpu.toml (modele w ~/.szukaj_cpu/models).

  szukaj-cpu --init            # utwórz ~/.szukaj_cpu/szukaj_cpu.toml z szablonu
  szukaj-cpu --test            # test kamer, modelu i API Shinobi, potem koniec
  szukaj-cpu                   # praca ciągła (zwykle uruchamiana przez systemd)

Moduły:
  common       stałe, logger, flaga STOP, drobne narzędzia
  config       odczyt i kontrola szukaj_cpu.toml
  camera       kamery: ffmpeg, odbiór klatek, bufor segmentów, watchdog
  detector     model YOLO (OpenVINO FP32/INT8 lub PyTorch), strefa, przycięcie, rysowanie
  events       zdarzenia: łączenie wykryć, zapis, pomijanie stojących pojazdów
  gallery      galeria HTML/CSV, sprzątanie
  clips        klipy i pełne klatki ze strumienia głównego
  notify       ntfy i Shinobi
  web/         serwer WWW, panel obszaru detekcji i konfiguratora (HTML w osobnych plikach)
  main         argumenty, pętla główna, tryb testowy
"""

import os as _os

# PyTorch/OpenMP: bez aktywnego czekania wątków po każdej operacji – inaczej wątki OpenMP
# zajmują rdzenie potrzebne OpenVINO (w usłudze analiza była przez to 2–3x wolniejsza).
# Musi być ustawione przed pierwszym importem torch.
_os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
_os.environ.setdefault("KMP_BLOCKTIME", "0")

__version__ = "1.13.0"
