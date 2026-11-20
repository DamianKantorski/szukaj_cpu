"""Przeglądarka nagrań: lista plików i zdarzeń na dany dzień, wysyłanie plików z obsługą Range."""

import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path

from ..common import NAME_DATE

FILE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.mp4$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MAX_SEG = 3 * 3600          # nagranie dłuższe niż 3 h – coś nie tak z czasem modyfikacji, przytnij


def _start(name):
    m = NAME_DATE.match(name)
    if not m:
        return None
    try:
        return datetime.strptime(f"{m.group(1)}T{m.group(2)}-{m.group(3)}-{m.group(4)}", "%Y-%m-%dT%H-%M-%S")
    except ValueError:
        return None


def rec_cams(cams):
    return [c for c in cams if c.record_dir is not None]


def _files(d, day):
    """Pliki z danego dnia i z poprzedniego (nagranie sprzed północy może sięgać w ten dzień)."""
    out = []
    for dd in (day - timedelta(days=1), day):
        try:
            out += [p for p in d.glob(f"{dd.isoformat()}T*.mp4") if FILE_RE.match(p.name)]
        except OSError:
            pass
    return out


def day_view(cfg, cams, gallery, day_s):
    if not DATE_RE.match(day_s or ""):
        raise ValueError("nieprawidłowa data")
    day = date.fromisoformat(day_s)
    d0 = datetime.combine(day, datetime.min.time())
    d1 = d0 + timedelta(days=1)
    now = datetime.now()
    seg_len = cfg.rec_min * 60
    out = []
    for c in rec_cams(cams):
        items = []
        for p, live in [(p, False) for p in _files(c.record_dir, day)] + \
                       [(p, True) for p in (_files(c.staging, day) if c.staging and c.staging.is_dir() else [])]:
            t = _start(p.name)
            try:
                st = p.stat()
            except OSError:
                continue
            if t is None or st.st_size < 1024:
                continue
            end = datetime.fromtimestamp(st.st_mtime)
            if live:
                end = max(end, now) if c.rec_proc is not None and c.rec_proc.poll() is None else end
            elif end - t > timedelta(seconds=seg_len + 120):     # przepisany później (np. remux po restarcie)
                end = t + timedelta(seconds=seg_len)
            end = min(end, t + timedelta(seconds=MAX_SEG))
            if end <= d0 or t >= d1 or end <= t:
                continue
            items.append({"file": ("~" if live else "") + p.name, "start": (t - d0).total_seconds(),
                          "end": (end - d0).total_seconds(), "size": st.st_size, "live": live})
        items.sort(key=lambda x: x["start"])
        for a, b in zip(items, items[1:]):          # nakładanie (np. mtime po remuxie) – utnij do następnego
            if a["end"] > b["start"]:
                a["end"] = max(a["start"] + 1, b["start"])
        out.append({"name": c.name, "recording": bool(c.record), "segments": items})
    events = []
    if gallery is not None:
        with gallery.lock:
            rows = list(gallery.rows)
        names = {c.name for c in cams}
        for r in rows:
            try:
                t = datetime.strptime(r["poczatek"], "%Y-%m-%d %H:%M:%S")
                e = datetime.strptime(r.get("koniec") or r["poczatek"], "%Y-%m-%d %H:%M:%S")
            except (KeyError, ValueError):
                continue
            if e < d0 or t >= d1:
                continue
            clip, full = r.get("klip") or "", r.get("pelna") or ""
            events.append({"cam": r.get("kamera", ""), "known": r.get("kamera", "") in names,
                           "start": (t - d0).total_seconds(), "end": (e - d0).total_seconds(),
                           "thumb": r.get("miniatura", ""), "clip": clip if clip and (gallery.dir / clip).exists() else "",
                           "full": full if full and (gallery.dir / full).exists() else "",
                           "objects": r.get("obiekty", ""), "conf": r.get("max_pewnosc", ""),
                           "count": r.get("max_obiektow", "")})
        events.sort(key=lambda x: x["start"])
    return {"date": day.isoformat(), "now": (now - d0).total_seconds() if d0 <= now < d1 else None,
            "cams": out, "events": events}


def days_view(cams, gallery, limit=60):
    """Dni z nagraniami (do kalendarza): {dzień: liczba plików}, zdarzenia na dzień."""
    days = {}
    for c in rec_cams(cams):
        for d in (c.record_dir, c.staging):
            if d is None or not d.is_dir():
                continue
            try:
                for e in os.scandir(d):
                    if FILE_RE.match(e.name):
                        days[e.name[:10]] = days.get(e.name[:10], 0) + 1
            except OSError:
                pass
    ev = {}
    if gallery is not None:
        with gallery.lock:
            for r in gallery.rows:
                k = (r.get("poczatek") or "")[:10]
                if k:
                    ev[k] = ev.get(k, 0) + 1
    keys = sorted(set(days) | set(ev), reverse=True)[:limit]
    return {"days": [{"date": k, "files": days.get(k, 0), "events": ev.get(k, 0)} for k in keys]}


def resolve_file(cams, cam_name, fname):
    """Bezpieczne odwzorowanie /rec/<kamera>/<plik> na ścieżkę (tylko pliki nagrań tej kamery)."""
    cam = next((c for c in rec_cams(cams) if c.name == cam_name), None)
    if cam is None:
        return None
    live = fname.startswith("~")
    name = fname[1:] if live else fname
    if not FILE_RE.match(name):
        return None
    p = (cam.staging if live else cam.record_dir) / name
    return p if p.is_file() else None


RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


def send_file(handler, p: Path, download=False):
    """Plik MP4 z obsługą Range (przewijanie w odtwarzaczu). Plik w trakcie zapisu: rozmiar z chwili zapytania."""
    try:
        f = open(p, "rb")
    except OSError:
        return handler._send(404, {"error": "brak pliku"})
    with f:
        size = os.fstat(f.fileno()).st_size
        start, end, code = 0, size - 1, 200
        m = RANGE_RE.match(handler.headers.get("Range", "").strip())
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
            else:
                start = max(0, size - int(m.group(2)))
            if start > end or start >= size:
                handler.send_response(416)
                handler.send_header("Content-Range", f"bytes */{size}")
                handler.send_header("Content-Length", "0")
                handler.end_headers()
                return
            code = 206
        length = end - start + 1
        handler.send_response(code)
        handler.send_header("Content-Type", "video/mp4")
        handler.send_header("Accept-Ranges", "bytes")
        handler.send_header("Content-Length", str(length))
        handler.send_header("Cache-Control", "no-store")
        if code == 206:
            handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            handler.send_header("Content-Disposition", f'attachment; filename="{p.parent.name}_{p.name}"')
        handler.end_headers()
        if handler.command == "HEAD":
            return
        f.seek(start)
        left = length
        try:
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                handler.wfile.write(chunk)
                left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass
