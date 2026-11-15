#!/usr/bin/env python3
"""Porównanie dwóch modeli OpenVINO (zwykle FP32 i INT8) na tych samych obrazach:
czas na klatkę i zgodność wykryć (INT8 względem FP32).

    python3 compare_models.py MODEL_FP32 MODEL_INT8 KATALOG_OBRAZÓW [--classes person car] [--conf 0.5]

Najlepiej na klatkach z własnych kamer (dzień i noc), np. katalog pelne/ albo miniatury/
z galerii. Na serwerze:  /opt/szukaj_cpu/venv/bin/python /opt/szukaj_cpu/tools/compare_models.py …
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

_here = Path(__file__).resolve().parent
for _p in (_here.parent / "src", _here.parent):
    if (_p / "szukaj_cpu" / "detector.py").is_file():
        sys.path.insert(0, str(_p))
        break

import cv2  # noqa: E402

from szukaj_cpu.common import COCO80  # noqa: E402
from szukaj_cpu.detector import OpenVinoYolo  # noqa: E402


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def run(model, frames, cls, conf):
    times, dets = [], []
    model(frames[0], cls, conf)                       # rozgrzewka
    for f in frames:
        t0 = time.perf_counter()
        dets.append(model(f, cls, conf))
        times.append((time.perf_counter() - t0) * 1000)
    return times, dets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", help="model odniesienia (FP32), katalog *_openvino_model")
    ap.add_argument("test", help="model porównywany (INT8)")
    ap.add_argument("images", help="katalog z obrazami .jpg/.png")
    ap.add_argument("--classes", nargs="+", default=["person", "car", "truck", "bicycle", "motorcycle", "dog"])
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--native", action="store_true",
                    help="bez wymuszania obliczeń FP32 (nowsze CPU liczą model FP32 w bf16 – szybciej niż serwer)")
    a = ap.parse_args()

    files = sorted(p for p in Path(a.images).expanduser().rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    frames = [f for f in (cv2.imread(str(p)) for p in files[: a.limit]) if f is not None]
    if not frames:
        sys.exit(f"Brak obrazów w {a.images}")
    cls = [COCO80.index(c) for c in a.classes if c in COCO80]
    hint = None if a.native else "f32"
    ref = OpenVinoYolo(Path(a.ref), a.threads, precision_hint=hint)
    tst = OpenVinoYolo(Path(a.test), a.threads, precision_hint=hint)
    t_ref, d_ref = run(ref, frames, cls, a.conf)
    t_tst, d_tst = run(tst, frames, cls, a.conf)

    matched = missed = extra = 0
    dconf = []
    for r, t in zip(d_ref, d_tst):
        used = set()
        for b in r:
            best, bi = 0.0, None
            for i, c in enumerate(t):
                if i not in used and c[5] == b[5]:
                    v = iou(b, c)
                    if v > best:
                        best, bi = v, i
            if bi is not None and best >= 0.5:
                used.add(bi)
                matched += 1
                dconf.append(t[bi][4] - b[4])
            else:
                missed += 1
        extra += len(t) - len(used)
    n_ref = matched + missed
    m_ref, m_tst = statistics.median(t_ref), statistics.median(t_tst)
    print(f"Obrazy: {len(frames)} z {a.images}, klasy: {', '.join(a.classes)}, próg {a.conf}, wątki {a.threads}"
          f"{'' if a.native else ', obliczenia FP32 wymuszone (jak na CPU bez bf16, np. i5-7500)'}")
    print(f"Czas na klatkę (mediana): {Path(a.ref).name}: {m_ref:.0f} ms   {Path(a.test).name}: {m_tst:.0f} ms"
          f"   → INT8 {m_ref / m_tst:.2f}× szybciej (na tym komputerze; na serwerze uruchom to narzędzie ponownie)")
    print(f"Wykrycia odniesienia: {n_ref}; zgodne w modelu porównywanym: {matched}"
          f" ({100 * matched / n_ref if n_ref else 100:.1f}%), pominięte: {missed}, dodatkowe: {extra}")
    if dconf:
        print(f"Średnia zmiana pewności zgodnych wykryć: {statistics.mean(dconf):+.3f}")


if __name__ == "__main__":
    main()
