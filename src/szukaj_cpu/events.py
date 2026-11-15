"""Zdarzenia: łączenie wykryć, zapis do galerii, pomijanie stojących pojazdów."""

import json
import time
from datetime import datetime
from pathlib import Path

import cv2

from .common import FILE_TS, TS_FMT, log
from .detector import detect, draw
from .notify import notify, shinobi_event


def on_detections(cfg, cam, t, frame, dets):
    best = max(d[4] for d in dets)
    labels = {d[5] for d in dets}
    ev = cam.event
    if ev is not None and t - ev["end"] <= cfg.gap:
        ev["end"] = t
        ev["hits"] += 1
        ev["max_n"] = max(ev["max_n"], len(dets))
        ev["labels"] |= labels
        if best > ev["best"]:
            ev.update(best=best, frame=frame, dets=dets, best_t=t)
    else:
        ev = cam.event = {"start": t, "end": t, "hits": 1, "max_n": len(dets), "best": best,
                          "frame": frame, "dets": dets, "labels": set(labels), "announced": False, "best_t": t,
                          "sh_sent": 0.0}
    if not ev["announced"] and ev["hits"] >= cfg.min_hits:
        ev["announced"] = True
        log.info("[%s] wykryto: %s", cam.name, ", ".join(sorted(ev["labels"])))
        notify(cfg, cam, frame, dets, ev["labels"])
    # oś czasu Shinobi: przy starcie zdarzenia i co "ponawiaj" s, dopóki obiekt jest widoczny
    if ev["announced"] and cfg.sh_repeat >= 0 and (
            ev["sh_sent"] == 0 or (cfg.sh_repeat > 0 and t - ev["sh_sent"] >= cfg.sh_repeat)):
        ev["sh_sent"] = t
        shinobi_event(cfg, cam, frame, dets)


def close_event(cfg, cam, gallery, clipper):
    ev, cam.event = cam.event, None
    if ev is None or ev["hits"] < cfg.min_hits:
        return
    base = f"{cam.name}_{datetime.fromtimestamp(ev['start']).strftime(FILE_TS)}"
    thumb = f"miniatury/{base}.jpg"
    cv2.imwrite(str(gallery.dir / thumb), draw(ev["frame"], ev["dets"], cam.zone, cam.crop),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    full = ""
    if cfg.clips and cfg.full:   # pełna klatka ze strumienia głównego (z bufora segmentów)
        full = f"pelne/{base}.{cfg.full_fmt}"
        clipper.add_full(cam, ev["best_t"], gallery.dir / full, gallery.dir / thumb)
    clip = ""
    if cfg.clips:
        start = ev["start"] - cfg.pre
        end = min(ev["end"] + cfg.post, start + cfg.max_len)
        clip = f"klipy/{base}.mp4"
        clipper.add(cam, start, end, gallery.dir / clip)
    gallery.add({
        "kamera": cam.name,
        "poczatek": datetime.fromtimestamp(ev["start"]).strftime(TS_FMT),
        "koniec": datetime.fromtimestamp(ev["end"]).strftime(TS_FMT),
        "obiekty": ", ".join(sorted(ev["labels"])),
        "trafien": ev["hits"],
        "max_obiektow": ev["max_n"],
        "max_pewnosc": round(ev["best"], 3),
        "miniatura": thumb,
        "klip": clip,
        "pelna": full,
    })
    log.info("[%s] zdarzenie %s–%s zapisane (%d klatek, max %d, pewność %.2f)", cam.name,
             datetime.fromtimestamp(ev["start"]).strftime("%H:%M:%S"),
             datetime.fromtimestamp(ev["end"]).strftime("%H:%M:%S"), ev["hits"], ev["max_n"], ev["best"])


# ── obiekty nieruchome ──
# Auto zaparkowane na podjeździe jest wykrywane w każdej klatce. W nocy pewność skacze wokół
# progu, więc znika na dłużej niż event_gap i każde ponowne wykrycie dawało nowe zdarzenie.
# Ramka, która stoi w tym samym miejscu dłużej niż static_after, jest zapamiętywana jako
# nieruchoma i dalsze wykrycia w tym miejscu są pomijane. Przyjazd, odjazd i ruch (inna ramka)
# nadal wywołują zdarzenie. Dotyczy tylko klas ze static_classes (domyślnie bez "person").
STATIC_IOU = 0.6

STATIC_MAX = 64


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def static_filter(cfg, cam, t, frame, dets):
    if not cfg.static_cls or cfg.static_after <= 0:
        return dets
    h, w = frame.shape[:2]
    keep = []
    for d in dets:
        if d[5] not in cfg.static_cls:
            keep.append(d)
            continue
        box = (d[0] / w, d[1] / h, d[2] / w, d[3] / h)
        best, score = None, STATIC_IOU
        for tr in cam.tracks:
            if tr["label"] == d[5]:
                s = box_iou(box, tr["box"])
                if s >= score:
                    best, score = tr, s
        if best is None:
            # nowe miejsce (przyjazd albo ruch) – osobny ślad, dotychczasowe zostają
            cam.tracks.append({"label": d[5], "box": box, "since": t, "seen": t, "static": False})
            keep.append(d)
            continue
        best["seen"] = t
        if not best["static"] and t - best["since"] >= cfg.static_after:
            best["static"] = True
            log.info("[%s] %s stoi w miejscu od %.0f s – kolejne wykrycia w tym miejscu pomijam",
                     cam.name, d[5], t - best["since"])
        if not best["static"]:
            keep.append(d)
    # zapominanie: nieruchome po static_forget bez wykrycia, pozostałe szybko
    cam.tracks = [tr for tr in cam.tracks
                  if t - tr["seen"] <= (cfg.static_forget if tr["static"] else max(cfg.static_after, cfg.gap))]
    del cam.tracks[:-STATIC_MAX]
    return keep


def static_state_path(cfg):
    return Path(cfg.path).parent / "nieruchome.json"


def save_static(cfg, cams):
    data = {c.name: [tr for tr in c.tracks if tr["static"]] for c in cams}
    try:
        tmp = static_state_path(cfg).with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(static_state_path(cfg))
    except OSError as e:
        log.warning("Nie zapisano stanu obiektów nieruchomych: %s", e)


def load_static(cfg, cams):
    try:
        data = json.loads(static_state_path(cfg).read_text())
    except (OSError, ValueError):
        return
    now = time.time()
    for c in cams:
        c.tracks = [dict(tr, box=tuple(tr["box"])) for tr in data.get(c.name, [])
                    if tr.get("label") in cfg.static_cls and now - tr.get("seen", 0) <= cfg.static_forget]
        if c.tracks:
            log.info("[%s] obiekty nieruchome z poprzedniego uruchomienia: %s", c.name,
                     ", ".join(tr["label"] for tr in c.tracks))


def process(cfg, det, cam, t, frame):
    dets = static_filter(cfg, cam, t, frame, detect(det, cam, frame))
    if dets:
        on_detections(cfg, cam, t, frame, dets)
