"""Odczyt i kontrola pliku konfiguracji szukaj_cpu.toml."""

import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .common import CONF_DIR, APP_DIR, expand


def check_zone(pts):
    """Wielokąt strefy [[x, y], ...] w ułamkach kadru; [] = bez strefy. ValueError przy błędzie."""
    if not pts:
        return None
    try:
        arr = np.array([(float(x), float(y)) for x, y in pts], dtype=np.float32)
    except (TypeError, ValueError):
        raise ValueError("zone must be a list of [x, y] pairs, e.g. [[0, 0.35], [1, 0.35], [1, 1], [0, 1]]")
    if len(arr) < 3:
        raise ValueError("zone needs at least 3 points")
    if (arr < 0).any() or (arr > 1).any():
        raise ValueError("zone coordinates must be between 0 and 1")
    return arr


def parse_zone(pts, where):
    try:
        return check_zone(pts)
    except ValueError as e:
        sys.exit(f"{where}: {e}")


def find_tool(value, name, cfg_path):
    """"auto"/"" -> statyczny ffmpeg z ~/.szukaj_cpu/, potem z katalogu programu (/opt/szukaj_cpu), na końcu z PATH.
    Ścieżka względna liczona od katalogu skryptu."""
    v = (value or "").strip()
    if v.lower() in ("", "auto"):
        for local in (CONF_DIR / name, APP_DIR / name):
            if local.is_file() and os.access(local, os.X_OK):
                return str(local)
        found = shutil.which(name)
        if not found:
            sys.exit(f"{cfg_path}: nie znaleziono {name} (ani w {CONF_DIR}, ani w {APP_DIR}, ani w PATH)")
        return found
    p = Path(os.path.expanduser(v))
    if not p.is_absolute() and ("/" in v):
        p = APP_DIR / p
    if p.is_absolute() or "/" in v:
        if not (p.is_file() and os.access(p, os.X_OK)):
            sys.exit(f"{cfg_path}: {name} = \"{v}\" – brak pliku wykonywalnego {p}")
        return str(p)
    return v                                   # sama nazwa – szukana w PATH przez system


def tool_version(binary):
    try:
        out = subprocess.run([binary, "-hide_banner", "-version"], capture_output=True, text=True, timeout=10).stdout
        return out.splitlines()[0].replace("ffmpeg version ", "").replace("ffprobe version ", "").split(" Copyright")[0]
    except Exception:
        return "?"


def check_crop(box):
    """Prostokąt analizy [x1, y1, x2, y2] (0–1); [] = cały kadr. ValueError przy błędzie."""
    if not box:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        raise ValueError("crop must be [x1, y1, x2, y2], e.g. [0.2, 0.3, 1.0, 1.0]")
    x1, y1, x2, y2 = (min(1.0, max(0.0, v)) for v in (x1, y1, x2, y2))
    if x2 - x1 < 0.05 or y2 - y1 < 0.05:
        raise ValueError("crop is too small or inverted (x2 > x1, y2 > y1, min. 5% of the frame)")
    if (x1, y1, x2, y2) == (0.0, 0.0, 1.0, 1.0):
        return None
    return (x1, y1, x2, y2)


def parse_crop(box, where):
    try:
        return check_crop(box)
    except ValueError as e:
        sys.exit(f"{where}: {e}")


# klucze dozwolone w każdej sekcji (literówki są zgłaszane w logu)
SCHEMA = {
    "paths": {"recordings", "gallery", "buffer", "model", "ffmpeg", "ffprobe", "vaapi_device", "log_file"},
    "detection": {"classes", "confidence", "imgsz", "aspect_ratio", "precision", "engine", "threads", "min_hits", "event_gap",
                  "static_classes", "static_after", "static_forget", "max_height", "watchdog", "frame_selection", "analysis_interval"},
    "clips": {"enabled", "pre_seconds", "post_seconds", "max_length", "segment_length"},
    "gallery": {"retention_days", "refresh_seconds", "http_port", "http_user", "http_password",
                "full_frame", "full_format",
                "full_quality", "thumb_width"},
    "notifications": {"ntfy_url", "min_interval", "ntfy_token", "priority", "click_url"},
    "shinobi": {"url", "api_key", "group_key", "repeat_seconds"},
    "camera": {"name", "url", "analysis_url", "enabled", "zone", "crop", "shinobi_monitor"},
}


def load_config(path):
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        sys.exit(f"Config file not found: {path}")
    except tomllib.TOMLDecodeError as e:
        sys.exit(f"Syntax error in {path}: {e}")

    for sec, val in data.items():
        if sec not in SCHEMA:
            print(f"! {path}: unknown section [{sec}] – ignored", file=sys.stderr)
            continue
        for item in (val if isinstance(val, list) else [val]):
            for key in set(item) - SCHEMA[sec]:
                print(f"! {path}: unknown key '{key}' in [{sec}] – ignored", file=sys.stderr)

    def sec(name):
        v = data.get(name, {})
        return v if isinstance(v, dict) else {}

    def get(section, key, default, typ):
        v = sec(section).get(key, default)
        if typ is float and isinstance(v, int) and not isinstance(v, bool):
            v = float(v)
        if not isinstance(v, typ) or (typ is not bool and isinstance(v, bool)):
            sys.exit(f"{path}: [{section}] {key} must be {typ.__name__}, got {v!r}")
        return v

    c = SimpleNamespace()
    c.path = Path(path)
    c.nagrania = expand(get("paths", "recordings", "", str))
    c.galeria = expand(get("paths", "gallery", "", str))
    if c.galeria is None:
        if c.nagrania is None:
            sys.exit(f"{path}: set [paths] gallery or recordings")
        c.galeria = c.nagrania.parent / "wyniki_osoby"
    c.bufor = expand(get("paths", "buffer", "/dev/shm/szukaj_cpu", str)) or Path("/dev/shm/szukaj_cpu")
    c.model = expand(get("paths", "model", "~/.szukaj_cpu/models/yolo11s.pt", str))
    c.ffmpeg = find_tool(get("paths", "ffmpeg", "auto", str), "ffmpeg", path)
    c.ffprobe = find_tool(get("paths", "ffprobe", "auto", str), "ffprobe", path)
    c.vaapi = get("paths", "vaapi_device", "/dev/dri/renderD128", str).strip() or None
    c.logfile = expand(get("paths", "log_file", "", str))

    c.klasy = [str(x).strip() for x in get("detection", "classes", ["person"], list) if str(x).strip()]
    c.conf = get("detection", "confidence", 0.5, float)
    c.imgsz = get("detection", "imgsz", 640, int)
    ar = get("detection", "aspect_ratio", "16:9", str).strip().lower()
    m = re.fullmatch(r"(\d+)\s*:\s*(\d+)", ar)
    if ar in ("", "square", "1:1"):
        c.input_hw = (c.imgsz, c.imgsz)
    elif m and int(m.group(1)) > 0 and int(m.group(2)) > 0:
        aw, ah = int(m.group(1)), int(m.group(2))
        up32 = lambda v: -(-round(v) // 32) * 32       # w górę do wielokrotności 32
        long_side = up32(c.imgsz)                      # imgsz = dłuższy bok kadru
        if aw >= ah:
            c.input_hw = (up32(long_side * ah / aw), long_side)
        else:
            c.input_hw = (long_side, up32(long_side * aw / ah))
    else:
        sys.exit(f'{path}: [detection] aspect_ratio must look like "16:9" or "square"')
    c.precision = get("detection", "precision", "fp32", str).strip().lower()
    if c.precision not in ("fp32", "int8"):
        sys.exit(f'{path}: [detection] precision must be "fp32" or "int8"')
    c.silnik = get("detection", "engine", "openvino", str).strip().lower()
    if c.silnik not in ("openvino", "pytorch"):
        sys.exit(f"{path}: [detection] engine must be \"openvino\" or \"pytorch\"")
    c.watki = get("detection", "threads", 2, int)
    c.min_hits = get("detection", "min_hits", 1, int)
    c.gap = get("detection", "event_gap", 10.0, float)
    c.static_cls = {str(x).strip() for x in get("detection", "static_classes",
                    ["car", "truck", "bus", "motorcycle", "bicycle"], list) if str(x).strip()}
    c.static_after = get("detection", "static_after", 60.0, float)
    c.static_forget = get("detection", "static_forget", 6 * 3600.0, float)
    c.max_h = get("detection", "max_height", 1080, int)
    c.watchdog = get("detection", "watchdog", 30.0, float)
    c.selection = get("detection", "frame_selection", "interval", str).strip().lower()
    if c.selection not in ("interval", "keyframes"):
        sys.exit(f'{path}: [detection] frame_selection must be "interval" or "keyframes"')
    c.interval = get("detection", "analysis_interval", 1.0, float)
    if c.interval <= 0:
        sys.exit(f"{path}: [detection] analysis_interval must be > 0")

    c.clips = get("clips", "enabled", True, bool)
    c.pre = get("clips", "pre_seconds", 3.0, float)
    c.post = get("clips", "post_seconds", 5.0, float)
    c.max_len = get("clips", "max_length", 300.0, float)
    c.seg = max(2, get("clips", "segment_length", 4, int))

    c.retencja = get("gallery", "retention_days", 5.0, float)
    c.odswiezanie = get("gallery", "refresh_seconds", 60, int)
    c.port = get("gallery", "http_port", 0, int)
    c.http_user = get("gallery", "http_user", "", str).strip()
    c.http_password = get("gallery", "http_password", "", str)
    c.full = get("gallery", "full_frame", True, bool)
    c.full_fmt = get("gallery", "full_format", "jxl", str).strip().lower().lstrip(".")
    if c.full_fmt == "jpeg":
        c.full_fmt = "jpg"
    if c.full_fmt not in ("jxl", "jpg", "webp"):
        sys.exit(f'{path}: [gallery] full_format must be "jxl", "jpg" or "webp"')
    c.full_q = get("gallery", "full_quality", 90, int)
    c.thumb_w = get("gallery", "thumb_width", 1280, int)

    c.ntfy = get("notifications", "ntfy_url", "", str).strip()
    c.ntfy_gap = get("notifications", "min_interval", 120.0, float)
    c.ntfy_token = get("notifications", "ntfy_token", "", str).strip()
    c.ntfy_prio = get("notifications", "priority", "high", str).strip()
    c.ntfy_click = get("notifications", "click_url", "", str).strip()

    c.sh_url = get("shinobi", "url", "", str).strip().rstrip("/")
    c.sh_api = get("shinobi", "api_key", "", str).strip()
    c.sh_group = get("shinobi", "group_key", "", str).strip()
    c.sh_repeat = get("shinobi", "repeat_seconds", 30.0, float)
    c.shinobi = bool(c.sh_url and c.sh_api and c.sh_group)

    c.cams = []
    cams = data.get("camera", [])
    if not isinstance(cams, list):
        sys.exit(f"{path}: cameras must be defined as [[camera]] blocks")
    for i, cam in enumerate(cams, 1):
        where = f"{path}: [[camera]] #{i}"
        name = re.sub(r"[^\w-]", "_", str(cam.get("name", f"camera{i}")).strip()) or f"camera{i}"
        if cam.get("enabled", True) is not True:
            continue
        url = str(cam.get("url", "")).strip()
        if not url or "FILL_IN" in url.upper():
            print(f"! {where} ({name}): no url – camera skipped", file=sys.stderr)
            continue
        if any(x.name == name for x in c.cams):
            sys.exit(f"{where}: duplicate camera name '{name}'")
        c.cams.append(SimpleNamespace(name=name, url=url, zone=parse_zone(cam.get("zone", []), where),
                                      monitor=str(cam.get("shinobi_monitor", "")).strip(),
                                      analysis_url=str(cam.get("analysis_url", "")).strip(),
                                      crop=parse_crop(cam.get("crop", []), where)))
    if not c.cams:
        sys.exit(f"{path}: no cameras configured – fill in url in the [[camera]] blocks")
    return c
