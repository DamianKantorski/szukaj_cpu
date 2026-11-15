#!/usr/bin/env python3
"""Eksport modelu YOLO do OpenVINO dla szukaj_cpu: wersja standardowa (FP32) i INT8.

Wymaga Ultralytics + PyTorch (CPU) + OpenVINO + NNCF – na komputerze budującym albo przez
`sudo szukaj-cpu-export-tools install` na serwerze. Do pracy usługi nie są potrzebne.

Nazwy katalogów (szukaj_cpu wybiera po nich model):
    yolo11s_960x544_openvino_model        FP32 – wejście 960×544 (16:9)
    yolo11s_960x544_int8_openvino_model   INT8 – kwantyzacja NNCF

INT8 jest kalibrowany na zbiorze obrazów. Domyślnie coco128 (pobierany przez Ultralytics).
Lepszy wynik dają klatki z własnych kamer (dzień i noc, ~100–300 plików .jpg), np. katalog
`pelne/` albo `miniatury/` z galerii:  --calib /ścieżka/do/klatek

Przykłady:
    python3 export_models.py --sizes 960x544 --out ./models
    python3 export_models.py --sizes 960x544 640x384 --int8-only --calib ~/klatki --out ./models
"""
import argparse
import os
import re
import sys
import zipfile
from pathlib import Path

_here = Path(__file__).resolve().parent
for _p in (_here.parent / "src", _here.parent):          # repozytorium albo /opt/szukaj_cpu
    if (_p / "szukaj_cpu" / "export.py").is_file():
        sys.path.insert(0, str(_p))
        break
from szukaj_cpu.export import export_one  # noqa: E402


def make_zip(model_dir):
    """Plik .zip z katalogiem modelu – do wgrania w panelu (Konfiguracja → Model)."""
    z = model_dir.with_name(model_dir.name + ".zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
        for p in sorted(model_dir.rglob("*")):
            if p.is_file():
                f.write(p, p.relative_to(model_dir.parent))
    print(f"   paczka do panelu: {z}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo11s.pt", help="wagi YOLO (.pt); brakujące pobiera Ultralytics")
    ap.add_argument("--sizes", nargs="+", default=["960x544"], help="wejścia modelu SZERxWYS, np. 960x544 640x384")
    ap.add_argument("--out", default="models", help="katalog wynikowy")
    ap.add_argument("--calib", help="katalog z klatkami do kalibracji INT8 (domyślnie coco128)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--int8-only", action="store_true", help="tylko INT8")
    g.add_argument("--fp32-only", action="store_true", help="tylko FP32")
    ap.add_argument("--force", action="store_true", help="eksportuj ponownie istniejące")
    ap.add_argument("--zip", action="store_true", help="dodatkowo .zip każdego modelu do wgrania w panelu WWW")
    a = ap.parse_args()

    out = Path(a.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    pt = Path(a.model).expanduser()
    if not pt.is_file():                       # nazwa z repozytorium Ultralytics -> pobierz do katalogu wyników
        os.chdir(out)
        from ultralytics import YOLO
        YOLO(pt.name)
        pt = out / pt.name
    pt = pt.resolve()
    for s in a.sizes:
        m = re.fullmatch(r"(\d+)x(\d+)", s)
        if not m or int(m.group(1)) % 32 or int(m.group(2)) % 32:
            sys.exit(f"Zły rozmiar {s!r}: SZERxWYS, obie wartości podzielne przez 32 (np. 960x544)")
        w, h = int(m.group(1)), int(m.group(2))
        done = []
        if not a.int8_only:
            done.append(export_one(pt, w, h, False, out, force=a.force))
        if not a.fp32_only:
            done.append(export_one(pt, w, h, True, out, calib=a.calib, force=a.force))
        if a.zip:
            for d in done:
                make_zip(d)


if __name__ == "__main__":
    main()
