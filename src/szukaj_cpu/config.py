"""Odczyt i kontrola pliku konfiguracji szukaj_cpu.toml."""

import ipaddress
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .common import COCO80, CONF_DIR, APP_DIR, expand


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


def default_record_dir(root, group, monitor, name):
    """Katalog nagrań kamery jak w Shinobi: <katalog wideo>/<klucz grupy>/<ID monitora>.
    [paths] recordings może wskazywać katalog wideo Shinobi albo już katalog grupy –
    wtedy klucz grupy nie jest dodawany drugi raz."""
    if root is None:
        return None
    root = Path(root)
    if group and monitor:
        if root.name == group or (not (root / group).is_dir() and (root / monitor).is_dir()):
            return root / monitor
        return root / group / monitor
    return root / (monitor or name)


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
    "gallery": {"retention_days", "refresh_seconds", "http_port", "http_listen", "http_user", "http_password",
                "login_max_failures", "login_lockout_minutes",
                "full_frame", "full_format",
                "full_quality", "thumb_width"},
    "notifications": {"ntfy_url", "min_interval", "ntfy_token", "priority", "click_url"},
    "shinobi": {"url", "api_key", "group_key", "repeat_seconds"},
    "recording": {"enabled", "segment_minutes", "retention_days", "min_free_gb", "audio"},
    "go2rtc": {"enabled", "binary", "api_port", "rtsp_port", "rtsp_listen", "webrtc_port"},
    "homekit": {"enabled", "port", "address", "motion_seconds"},
    "camera": {"name", "url", "analysis_url", "enabled", "zone", "crop", "shinobi_monitor", "record", "record_dir",
               "homekit", "homekit_motion"},
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
    listen = get("gallery", "http_listen", "0.0.0.0", str)
    c.http_listen = [a.strip().strip("[]") for a in listen.split(",") if a.strip()] or ["0.0.0.0"]
    for a in c.http_listen:
        try:
            ipaddress.ip_address(a)
        except ValueError:
            sys.exit(f"{path}: [gallery] http_listen: \"{a}\" nie jest adresem IP (np. 0.0.0.0, 127.0.0.1, 100.x.y.z)")
    c.login_max_fail = max(0, get("gallery", "login_max_failures", 5, int))
    c.login_lockout = max(1.0, get("gallery", "login_lockout_minutes", 15, float))
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

    # zapis ciągły na dysk (zamiast Shinobi)
    c.rec = get("recording", "enabled", False, bool)
    c.rec_min = get("recording", "segment_minutes", 15, int)
    if not 1 <= c.rec_min <= 60:
        sys.exit(f"{path}: [recording] segment_minutes must be 1–60")
    c.rec_days = get("recording", "retention_days", 14.0, float)
    c.rec_free = get("recording", "min_free_gb", 50.0, float)
    c.rec_audio = get("recording", "audio", False, bool)
    if c.rec and c.nagrania is None:
        sys.exit(f"{path}: [recording] enabled needs [paths] recordings")

    # go2rtc: jedno połączenie z kamerą na strumień, dalej lokalny restream
    c.g2r = get("go2rtc", "enabled", True, bool)
    c.g2r_bin = get("go2rtc", "binary", "auto", str).strip() or "auto"
    c.g2r_api = get("go2rtc", "api_port", 1984, int)
    c.g2r_rtsp = get("go2rtc", "rtsp_port", 8554, int)
    c.g2r_listen = get("go2rtc", "rtsp_listen", "127.0.0.1", str).strip() or "127.0.0.1"
    c.g2r_webrtc = get("go2rtc", "webrtc_port", 8555, int)

    c.hk = get("homekit", "enabled", False, bool)
    c.hk_port = get("homekit", "port", 51826, int)
    c.hk_address = get("homekit", "address", "auto", str).strip() or "auto"
    if c.hk_address != "auto":
        try:
            ipaddress.ip_address(c.hk_address)
        except ValueError:
            sys.exit(f"{path}: [homekit] address: \"{c.hk_address}\" nie jest adresem IP (albo \"auto\")")
    c.hk_motion_s = max(2.0, get("homekit", "motion_seconds", 10.0, float))

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
        monitor = str(cam.get("shinobi_monitor", "")).strip()
        record = cam.get("record", True)
        if not isinstance(record, bool):
            sys.exit(f"{where}: record must be true or false")
        hk = cam.get("homekit", False)
        if not isinstance(hk, bool):
            sys.exit(f"{where}: homekit must be true or false")
        hk_motion = cam.get("homekit_motion", [])
        if not isinstance(hk_motion, list) or not all(isinstance(x, str) for x in hk_motion):
            sys.exit(f"{where}: homekit_motion must be a list of class names, e.g. [\"person\", \"car\"]")
        hk_motion = list(dict.fromkeys(x.strip() for x in hk_motion if x.strip()))
        bad = [x for x in hk_motion if x not in COCO80]
        if bad:
            sys.exit(f"{where}: homekit_motion: unknown class {', '.join(bad)} (COCO names, e.g. person, car)")
        rdir = expand(str(cam.get("record_dir", "")).strip())
        if rdir is None:
            rdir = default_record_dir(c.nagrania, c.sh_group, monitor, name)
        c.cams.append(SimpleNamespace(name=name, url=url, zone=parse_zone(cam.get("zone", []), where),
                                      monitor=monitor,
                                      analysis_url=str(cam.get("analysis_url", "")).strip(),
                                      crop=parse_crop(cam.get("crop", []), where),
                                      record=record and c.rec, record_dir=rdir,
                                      homekit=hk and c.hk, homekit_motion=hk_motion if hk and c.hk else []))
    rdirs = [str(x.record_dir) for x in c.cams if x.record and x.record_dir]
    dup = {d for d in rdirs if rdirs.count(d) > 1}
    if dup:
        sys.exit(f"{path}: two cameras record to the same directory {', '.join(sorted(dup))} – set record_dir "
                 f"or a different shinobi_monitor")
    if not c.cams:
        sys.exit(f"{path}: no cameras configured – fill in url in the [[camera]] blocks")
    return c
