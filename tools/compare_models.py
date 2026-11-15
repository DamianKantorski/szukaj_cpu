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


def load_labels(img_path, shape, names):
    """Etykiety w formacie YOLO (…/labels/…/plik.txt obok …/images/…) -> ramki w pikselach."""
    parts = list(img_path.parts)
    if "images" not in parts:
        return None
    parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
    lab = Path(*parts).with_suffix(".txt")
    if not lab.is_file():
        return None
    h, w = shape[:2]
    out = []
    for line in lab.read_text().splitlines():
        v = line.split()
        if len(v) >= 5 and int(v[0]) < len(COCO80) and COCO80[int(v[0])] in names:
            cx, cy, bw, bh = (float(x) for x in v[1:5])
            out.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h, 1.0,
                        COCO80[int(v[0])]))
    return out


def match(ref, test):
    """Pary ramek tej samej klasy z IoU ≥ 0,5 (zachłannie, od najpewniejszych)."""
    used, pairs = set(), []
    for b in sorted(ref, key=lambda d: -d[4]):
        best, bi = 0.0, None
        for i, c in enumerate(test):
            if i not in used and c[5] == b[5]:
                v = iou(b, c)
                if v > best:
                    best, bi = v, i
        if bi is not None and best >= 0.5:
            used.add(bi)
            pairs.append((b, test[bi]))
    return pairs


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
    loaded = [(p, cv2.imread(str(p))) for p in files[: a.limit]]
    loaded = [(p, f) for p, f in loaded if f is not None]
    frames = [f for _, f in loaded]
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
        pairs = match(r, t)
        matched += len(pairs)
        missed += len(r) - len(pairs)
        extra += len(t) - len(pairs)
        dconf += [b[4] - a_[4] for a_, b in pairs]
    gts = [load_labels(p, f.shape, set(a.classes)) for p, f in loaded]
    n_ref = matched + missed
    m_ref, m_tst = statistics.median(t_ref), statistics.median(t_tst)
    print(f"Obrazy: {len(frames)} z {a.images}, klasy: {', '.join(a.classes)}, próg {a.conf}, wątki {a.threads}"
          f"{'' if a.native else ', obliczenia FP32 wymuszone (jak na CPU bez bf16, np. i5-7500)'}")
    print(f"Czas na klatkę (mediana): {Path(a.ref).name}: {m_ref:.0f} ms   {Path(a.test).name}: {m_tst:.0f} ms"
          f"   → porównywany {m_ref / m_tst:.2f}× szybszy (na tym komputerze; na serwerze uruchom to narzędzie ponownie)")
    print(f"Wykrycia odniesienia: {n_ref}; zgodne w modelu porównywanym: {matched}"
          f" ({100 * matched / n_ref if n_ref else 100:.1f}%), pominięte: {missed}, dodatkowe: {extra}")
    if dconf:
        print(f"Średnia zmiana pewności zgodnych wykryć: {statistics.mean(dconf):+.3f}")
    if any(g is not None for g in gts):                 # obrazy z etykietami (np. coco128): prawdziwa trafność
        n_gt = sum(len(g) for g in gts if g is not None)
        for label, dets in ((Path(a.ref).name, d_ref), (Path(a.test).name, d_tst)):
            tp = nd = 0
            for g, d in zip(gts, dets):
                if g is not None:
                    tp += len(match(g, d))
                    nd += len(d)
            print(f"Względem etykiet ({n_gt} obiektów): {label}: wykryte {100 * tp / n_gt if n_gt else 0:.1f}%"
                  f" (recall), trafne {100 * tp / nd if nd else 0:.1f}% wykryć (precision)")


if __name__ == "__main__":
    main()
