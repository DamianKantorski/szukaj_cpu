"""Powiadomienia ntfy i zdarzenia na osi czasu Shinobi."""

import json
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime

import cv2

from .common import log
from .detector import draw


def notify(cfg, cam, frame, dets, labels):
    if not cfg.ntfy:
        return
    now = time.time()
    if now - cam.last_notify < cfg.ntfy_gap:
        return
    cam.last_notify = now
    ok, jpg = cv2.imencode(".jpg", draw(frame, dets, cam.zone, cam.crop), [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return
    title = f"{cam.name}: {', '.join(sorted(labels))} ({datetime.now():%H:%M:%S})"
    sep = "&" if "?" in cfg.ntfy else "?"
    url = f"{cfg.ntfy}{sep}{urllib.parse.urlencode({'title': title, 'filename': f'{cam.name}.jpg'})}"

    headers = {"Tags": "rotating_light"}
    if cfg.ntfy_prio:
        headers["Priority"] = cfg.ntfy_prio
    if cfg.ntfy_click:
        headers["Click"] = cfg.ntfy_click
    if cfg.ntfy_token:
        headers["Authorization"] = f"Bearer {cfg.ntfy_token}"

    def post():
        try:
            req = urllib.request.Request(url, data=jpg.tobytes(), method="PUT", headers=headers)
            urllib.request.urlopen(req, timeout=15)
        except Exception as e:
            log.warning("Powiadomienie nieudane: %s", e)
    threading.Thread(target=post, daemon=True).start()


_sh_errors = {"n": 0}


def shinobi_event(cfg, cam, frame, dets, reason="object", wait=False):
    """Zdarzenie na osi czasu Shinobi: GET /[API]/motion/[GRUPA]/[MONITOR]?data={...}"""
    if not (cfg.shinobi and cam.monitor):
        return None
    h, w = frame.shape[:2]
    data = {
        "plug": "szukaj_cpu",
        "name": cam.name,
        "reason": reason,
        "confidence": round(max((d[4] for d in dets), default=1.0) * 100, 1),
        "imgWidth": w,
        "imgHeight": h,
        "matrices": [{"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1, "tag": label,
                      "confidence": round(c * 100, 1), "id": i}
                     for i, (x1, y1, x2, y2, c, label) in enumerate(dets)],
    }
    url = (f"{cfg.sh_url}/{cfg.sh_api}/motion/{cfg.sh_group}/{urllib.parse.quote(cam.monitor)}?"
           + urllib.parse.urlencode({"data": json.dumps(data, separators=(",", ":"))}))

    def send():
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                body = r.read(500).decode(errors="replace")
            if '"ok":false' in body.replace(" ", ""):
                raise RuntimeError(body)
            _sh_errors["n"] = 0
            return body
        except Exception as e:
            _sh_errors["n"] += 1
            if _sh_errors["n"] <= 3 or _sh_errors["n"] % 50 == 0:
                log.warning("[%s] Shinobi API: zdarzenie nieudane (%s)", cam.name, e)
            return None
    if wait:
        return send()
    threading.Thread(target=send, daemon=True).start()
    return None
