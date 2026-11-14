#!/usr/bin/env python3
"""
szukaj_cpu.py — wykrywanie osób (i innych obiektów) na żywo z kamer RTSP.

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
"""

import os as _os
# PyTorch/OpenMP: bez aktywnego czekania wątków po każdej operacji – inaczej wątki OpenMP
# zajmują rdzenie potrzebne OpenVINO (w usłudze analiza była przez to 2–3x wolniejsza).
# Musi być ustawione przed pierwszym importem torch.
_os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
_os.environ.setdefault("KMP_BLOCKTIME", "0")

import argparse
import base64
import hmac
import csv
import functools
import html
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import tomllib
import urllib.parse
import urllib.request
from collections import deque
from datetime import datetime, timedelta
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

log = logging.getLogger("szukaj_cpu")
STOP = threading.Event()

CSV_FIELDS = ["kamera", "poczatek", "koniec", "obiekty", "trafien", "max_obiektow", "pelna",
              "max_pewnosc", "miniatura", "klip"]
NAME_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})")   # jak w Shinobi
SEG_RE = re.compile(r"^(\d{8}-\d{6})\.ts$")
TS_FMT = "%Y-%m-%d %H:%M:%S"
FILE_TS = "%Y-%m-%dT%H-%M-%S"
__version__ = "1.0.5"
CONF_DIR = Path(os.path.expanduser("~/.szukaj_cpu"))
DEFAULT_CONF = "~/.szukaj_cpu/szukaj_cpu.toml"


# ═════════════════════════════ konfiguracja ═════════════════════════════

def expand(v):
    v = (v or "").strip()
    return Path(os.path.expanduser(v)) if v else None


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


SCRIPT_DIR = Path(__file__).resolve().parent


def find_tool(value, name, cfg_path):
    """"auto"/"" -> statyczny ffmpeg z ~/.szukaj_cpu/, potem obok szukaj_cpu.py, na końcu z PATH.
    Ścieżka względna liczona od katalogu skryptu."""
    v = (value or "").strip()
    if v.lower() in ("", "auto"):
        for local in (CONF_DIR / name, SCRIPT_DIR / name):
            if local.is_file() and os.access(local, os.X_OK):
                return str(local)
        found = shutil.which(name)
        if not found:
            sys.exit(f"{cfg_path}: nie znaleziono {name} (ani w {CONF_DIR}, ani obok szukaj_cpu.py, ani w PATH)")
        return found
    p = Path(os.path.expanduser(v))
    if not p.is_absolute() and ("/" in v):
        p = SCRIPT_DIR / p
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


def mask_url(url):
    return re.sub(r"//[^@/]*@", "//***@", url)


# klucze dozwolone w każdej sekcji (literówki są zgłaszane w logu)
SCHEMA = {
    "paths": {"recordings", "gallery", "buffer", "model", "ffmpeg", "ffprobe", "vaapi_device", "log_file"},
    "detection": {"classes", "confidence", "imgsz", "aspect_ratio", "engine", "threads", "min_hits", "event_gap",
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


# ═════════════════════════════ ffmpeg / kamery ═════════════════════════════

def unpin():
    """OpenVINO przypina główny wątek do jednego rdzenia; wątki i procesy ffmpeg uruchomione
    później dziedziczyłyby to przypięcie (wszystkie na rdzeniu 0). Przywróć wszystkie rdzenie."""
    try:
        os.sched_setaffinity(0, range(os.cpu_count() or 1))
    except (AttributeError, OSError):
        pass


def is_rtsp(url):
    return url.lower().startswith(("rtsp://", "rtsps://"))


def probe(cfg, url):
    cmd = [cfg.ffprobe, "-v", "error"]
    if is_rtsp(url):
        cmd += ["-rtsp_transport", "tcp", "-timeout", "10000000"]
    cmd += ["-select_streams", "v:0", "-show_entries", "stream=width,height,codec_name",
            "-of", "json", url]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        st = json.loads(r.stdout)["streams"][0]
        return int(st["width"]), int(st["height"]), st.get("codec_name", "?")
    except Exception:
        return None


def out_size(w, h, max_h):
    if max_h and h > max_h:
        oh = max_h - max_h % 2
        ow = int(round(w * oh / h / 2)) * 2
    else:
        ow, oh = w - w % 2, h - h % 2
    return ow, oh


def src_opts(url):
    if is_rtsp(url):
        return ["-rtsp_transport", "tcp", "-timeout", "10000000"]
    return ["-re"]  # plik zamiast kamery (testy): czytaj w tempie rzeczywistym


def build_cmd(cfg, cam, mode, ow, oh, hw_select=True):
    """Jeden ffmpeg na kamerę. Wejście nagrania (klipy, kopia) i wejście analizy mogą być
    tym samym strumieniem albo dwoma różnymi (np. główny + substream)."""
    c = [cfg.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
    # Osobny strumień analizy -> ten proces robi tylko analizę; bufor klipów ze strumienia
    # głównego nagrywa osobny proces (build_rec_cmd). W jednym procesie ffmpeg 7+ wyrównuje
    # wejścia względem siebie, co dodawało kilka sekund opóźnienia analizy.
    separate = cam.separate
    an_url = cam.analysis_url if separate else cam.url

    every = "interval" if cfg.selection == "interval" else "keyframes"
    scale = f"scale={ow}:{oh}:flags=bilinear"
    # wybór klatek: pierwsza klatka po upływie analysis_interval od poprzednio wybranej
    # (bez duplikowania klatek przy przerwach w strumieniu, w przeciwieństwie do filtra fps)
    select = f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{cfg.interval:g})'"
    c += src_opts(an_url)
    if mode == "vaapi":
        # dekodowanie na iGPU (scale_vaapi wymaga VideoProc, którego nie ma wolny sterownik iHD –
        # skalowanie na CPU)
        c += ["-hwaccel", "vaapi", "-hwaccel_device", cfg.vaapi]
        if every == "interval" and hw_select:
            # klatki zostają na GPU, do RAM trafia tylko wybrana; -reinit_filter 0: nie
            # przebudowuj filtrów przy zmianie parametrów strumienia
            c += ["-hwaccel_output_format", "vaapi", "-reinit_filter", "0"]
            vf = f"{select},hwdownload,format=nv12|p010le,{scale}"
        elif every == "interval":
            # tryb zapasowy: iGPU dekoduje, ffmpeg sam pobiera każdą klatkę do RAM, wybór na CPU
            vf = f"{select},{scale}"
        else:
            vf = scale
    else:
        vf = f"{select},{scale}" if every == "interval" else scale
    if every == "keyframes":
        c += ["-skip_frame", "nokey"]
    c += ["-i", an_url]
    if cfg.clips and not separate:
        # kopia strumienia (bez dekodowania) do bufora segmentów – do wycinania klipów
        c += segment_output(cfg, cam)
    c += ["-map", "0:v:0", "-vf", vf, "-fps_mode", "passthrough",
          "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    return c


def segment_output(cfg, cam):
    return ["-map", "0:v:0", "-c", "copy", "-f", "segment", "-segment_time", str(cfg.seg),
            "-reset_timestamps", "1", "-strftime", "1", str(cam.buf / "%Y%m%d-%H%M%S.ts")]


def build_rec_cmd(cfg, cam):
    """Osobny proces: kopia strumienia głównego do bufora segmentów (klipy, pełne klatki)."""
    return ([cfg.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
            + src_opts(cam.url) + ["-i", cam.url] + segment_output(cfg, cam))


class Camera:
    def __init__(self, spec, cfg):
        self.name, self.url, self.zone, self.cfg = spec.name, spec.url, spec.zone, cfg
        self.monitor = spec.monitor
        self.analysis_url = spec.analysis_url
        self.crop = spec.crop
        self.snap = None            # ostatnia klatka analizy (podgląd w panelu WWW)
        self.separate = bool(spec.analysis_url) and spec.analysis_url != spec.url
        self.rec_proc = None
        self.rec_err = deque(maxlen=10)
        self.buf = cfg.bufor / self.name
        self.lock = threading.Lock()
        self.latest = None          # (czas, klatka) – zawsze tylko najnowsza
        self.proc = None
        self.last_frame = 0.0
        self.mode, self.codec, self.size, self.src_size = "-", "?", (0, 0), (0, 0)
        self.vaapi_fail = 0
        self.hw_select = True       # wybór klatek na GPU (False = tryb zapasowy)
        self.filter_fail = 0
        self.frames = 0             # licznik do statystyk
        self.dropped = 0
        self.event = None
        self.tracks = []            # obiekty stojące w miejscu (np. zaparkowane auto), patrz static_filter
        self.last_notify = 0.0
        self.err = deque(maxlen=15)

    def take(self):
        with self.lock:
            item, self.latest = self.latest, None
        return item

    def loop(self):
        unpin()
        cfg, backoff = self.cfg, 5
        while not STOP.is_set():
            an_url = self.analysis_url or self.url
            info = probe(cfg, an_url)
            main = info if an_url == self.url else probe(cfg, self.url)
            if not info or not main:
                bad = an_url if not info else self.url
                log.warning("[%s] brak połączenia z %s – ponowna próba za %d s",
                            self.name, mask_url(bad), backoff)
                STOP.wait(backoff)
                backoff = min(backoff * 2, 120)
                continue
            w, h, an_codec = info
            self.codec = main[2]            # kodek strumienia nagrania (dla klipów)
            self.src_size = (w, h)
            fixed = ""
            if an_url != self.url and main[0] and main[1] and abs(w / h - main[0] / main[1]) > 0.02:
                # substream o innych proporcjach (np. 704x576 z kadru 16:9 – piksele niekwadratowe):
                # rozciągnij do proporcji strumienia głównego, żeby sylwetki nie były zniekształcone
                tw = int(round(h * main[0] / main[1]))
                fixed = f" (proporcje strumienia głównego {main[0]}x{main[1]})"
            else:
                tw = w
            ow, oh = out_size(tw, h, cfg.max_h)
            self.size = (ow, oh)
            self.main_size = (main[0], main[1]) if main[0] and main[1] else (w, h)
            mode = "vaapi" if cfg.vaapi and self.vaapi_fail < 3 else "cpu"
            if cfg.clips:
                self.buf.mkdir(parents=True, exist_ok=True)
            label = mode if (mode != "vaapi" or self.hw_select or cfg.selection != "interval") else "vaapi (wybór na CPU)"
            src = "substream" if an_url != self.url else "strumień główny"
            log.info("[%s] połączono: analiza z %s %dx%d %s -> %dx%d%s, dekodowanie %s",
                     self.name, src, w, h, an_codec, ow, oh, fixed, label)
            started = time.time()
            got = self._run(build_cmd(cfg, self, mode, ow, oh, self.hw_select), ow, oh, label)
            if STOP.is_set():
                break
            if mode == "vaapi":
                if got == 0:
                    self.vaapi_fail += 1
                    log.warning("[%s] VAAPI: brak klatek, komunikaty ffmpeg:\n    %s", self.name,
                                "\n    ".join(self.err) or "(brak)")
                    if self.vaapi_fail >= 2:
                        log.warning("[%s] VAAPI nie działa – przechodzę na dekodowanie CPU", self.name)
                    STOP.wait(2)      # błąd dekodera, nie sieci – ponów od razu, bez wydłużania przerwy
                    continue
                self.vaapi_fail = 0
            ran = time.time() - started
            errs = "\n    ".join(list(self.err)[-6:]) or "(brak komunikatu)"
            filter_err = any(("vf#" in e or "hwdownload" in e or "filter" in e.lower()) for e in self.err)
            if got > 0 and filter_err:
                # błąd przetwarzania klatek, nie sieci – łącz od razu
                if mode == "vaapi" and self.hw_select and cfg.selection == "interval":
                    self.filter_fail += 1
                    # uszkodzona klatka na GPU (np. Smart Codec / H.264+) – ten tryb będzie się
                    # wysypywał regularnie, więc od razu przejdź na tryb zapasowy
                    gpu_frame_err = any(("Failed to sync surface" in e or "Failed to download frame" in e)
                                        for e in self.err)
                    if self.filter_fail >= 3 or gpu_frame_err:
                        self.hw_select = False
                        log.warning("[%s] powtarzające się błędy wyboru klatek na GPU – przechodzę na "
                                    "tryb zapasowy (dekodowanie iGPU, wybór klatek na CPU)", self.name)
                log.warning("[%s] strumień przerwany po %.0f s (błąd filtra) – ponowne połączenie za 2 s:\n    %s",
                            self.name, ran, errs)
                backoff = 5
                STOP.wait(2)
                continue
            if got > 0:
                backoff = 5
            log.warning("[%s] strumień przerwany po %.0f s – ponowne połączenie za %d s:\n    %s",
                        self.name, ran, backoff, errs)
            STOP.wait(backoff)
            backoff = min(backoff * 2, 120)

    def _run(self, cmd, ow, oh, mode):
        size = ow * oh * 3
        self.mode = mode
        self.last_frame = time.time()
        self.err.clear()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.proc = proc

        def drain():
            for line in proc.stderr:
                self.err.append(line.decode(errors="replace").strip())
        threading.Thread(target=drain, daemon=True).start()

        got = 0
        try:
            while not STOP.is_set():
                buf = proc.stdout.read(size)
                if len(buf) < size:
                    break
                t = time.time()
                frame = np.frombuffer(buf, np.uint8).reshape(oh, ow, 3)
                with self.lock:
                    if self.latest is not None:
                        self.dropped += 1
                    self.latest = (t, frame)
                self.snap = frame
                self.last_frame = t
                self.frames += 1
                got += 1
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            self.proc = None
        return got

    def rec_loop(self):
        """Bufor klipów ze strumienia głównego – osobny proces ffmpeg (tylko kopia, bez dekodowania)."""
        unpin()
        backoff = 5
        while not STOP.is_set():
            self.buf.mkdir(parents=True, exist_ok=True)
            started = time.time()
            self.rec_err.clear()
            proc = subprocess.Popen(build_rec_cmd(self.cfg, self), stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE)
            self.rec_proc = proc
            for line in proc.stderr:
                self.rec_err.append(line.decode(errors="replace").strip())
            proc.wait()
            self.rec_proc = None
            if STOP.is_set():
                break
            if time.time() - started > 60:
                backoff = 5
            log.warning("[%s] bufor klipów (strumień główny) przerwany po %.0f s – ponownie za %d s: %s",
                        self.name, time.time() - started, backoff, self.rec_err[-1] if self.rec_err else "")
            STOP.wait(backoff)
            backoff = min(backoff * 2, 60)

    def watchdog(self, now):
        p = self.proc
        if p is not None and p.poll() is None and now - self.last_frame > self.cfg.watchdog:
            log.warning("[%s] brak klatek od %.0f s – restart połączenia", self.name, now - self.last_frame)
            p.kill()
        r = self.rec_proc
        if r is not None and r.poll() is None:
            segs = list_segments(self.buf)
            try:
                newest = max((p.stat().st_mtime for _, p in segs), default=0)
            except OSError:
                newest = now
            if newest and now - newest > max(self.cfg.watchdog, 4 * self.cfg.seg):
                log.warning("[%s] bufor klipów nie rośnie od %.0f s – restart", self.name, now - newest)
                r.kill()

    def stop(self):
        for p in (self.proc, self.rec_proc):
            if p is not None and p.poll() is None:
                p.kill()


# ═════════════════════════════ model ═════════════════════════════

COCO80 = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
          "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
          "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
          "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
          "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
          "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
          "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
          "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
          "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
          "teddy bear", "hair drier", "toothbrush"]


def read_names(model_dir):
    """Nazwy klas z metadata.yaml zapisanego przez eksport Ultralytics (bez zależności od PyYAML)."""
    names, inside = {}, False
    try:
        for line in (model_dir / "metadata.yaml").read_text(encoding="utf-8").splitlines():
            if line.startswith("names:"):
                inside = True
                continue
            if inside:
                m = re.match(r"\s+(\d+):\s*(.+)$", line)
                if m:
                    names[int(m.group(1))] = m.group(2).strip().strip("'\"")
                elif line.strip() and not line.startswith(" "):
                    break
    except OSError:
        pass
    return names or dict(enumerate(COCO80))


class OpenVinoYolo:
    """YOLO (eksport Ultralytics) uruchamiany bezpośrednio w OpenVINO – bez PyTorch i Ultralytics.
    Przygotowanie obrazu (letterbox) i NMS przez OpenCV."""

    def __init__(self, model_dir, threads, iou=0.7, max_det=100):
        import openvino as ov
        xml = next(model_dir.glob("*.xml"))
        core = ov.Core()
        props = {"PERFORMANCE_HINT": "LATENCY"}
        if threads > 0:
            props["INFERENCE_NUM_THREADS"] = int(threads)
        self.compiled = core.compile_model(core.read_model(xml), "CPU", props)
        self.req = self.compiled.create_infer_request()
        self.out = self.compiled.output(0)
        _, _, self.in_h, self.in_w = [int(d) for d in self.compiled.input(0).shape]
        self.names = read_names(model_dir)
        self.iou, self.max_det = iou, max_det

    def __call__(self, frame, classes, conf):
        h, w = frame.shape[:2]
        r = min(self.in_w / w, self.in_h / h)
        nw, nh = int(round(w * r)), int(round(h * r))
        img = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else frame
        left = (self.in_w - nw) // 2
        top = (self.in_h - nh) // 2
        img = cv2.copyMakeBorder(img, top, self.in_h - nh - top, left, self.in_w - nw - left,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
        blob = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))[None].astype(np.float32) * (1 / 255.0)

        pred = self.req.infer({0: blob})[self.out][0].T          # (N, 4 + liczba klas)
        scores = pred[:, 4:][:, classes]
        k = scores.argmax(1)
        best = scores[np.arange(len(scores)), k]
        keep = best >= conf
        if not keep.any():
            return []
        boxes, best, k = pred[keep, :4], best[keep], k[keep]
        # cx,cy,w,h (wejście modelu) -> x,y,w,h (klatka)
        bw, bh = boxes[:, 2] / r, boxes[:, 3] / r
        bx = (boxes[:, 0] - left) / r - bw / 2
        by = (boxes[:, 1] - top) / r - bh / 2
        rects = np.stack([bx, by, bw, bh], 1)
        cls_ids = np.asarray(classes)[k]
        idx = cv2.dnn.NMSBoxesBatched(rects.tolist(), best.tolist(), cls_ids.tolist(), conf, self.iou)
        res = []
        for i in np.array(idx).reshape(-1)[: self.max_det]:
            x, y, ww, hh = rects[i]
            res.append((int(max(0, x)), int(max(0, y)), int(min(w - 1, x + ww)), int(min(h - 1, y + hh)),
                        float(best[i]), self.names.get(int(cls_ids[i]), str(cls_ids[i]))))
        return res


class Detector:
    def __init__(self, cfg):
        self.cfg = cfg
        pt = cfg.model
        pt.parent.mkdir(parents=True, exist_ok=True)
        os.chdir(pt.parent)   # tu Ultralytics pobierze model, jeśli go brak

        if cfg.silnik == "openvino":
            h, w = cfg.input_hw
            ov = pt.parent / (f"{pt.stem}_{w}_openvino_model" if h == w else f"{pt.stem}_{w}x{h}_openvino_model")
            if not ov.exists():
                # Ultralytics/PyTorch potrzebne tylko do jednorazowego eksportu
                log.info("Eksport %s do OpenVINO (wejście %dx%d) – jednorazowo, potrwa chwilę…", pt.name, w, h)
                try:
                    from ultralytics import YOLO
                except ImportError:
                    sys.exit(f"Brak modelu {ov} i brak narzędzi do eksportu (Ultralytics/PyTorch).\n"
                             f"Zainstaluj je jednorazowo:  sudo szukaj-cpu-export-tools install\n"
                             f"a po eksporcie można je usunąć:  sudo szukaj-cpu-export-tools remove")
                out = YOLO(pt.name).export(format="openvino", imgsz=[h, w])
                Path(out).rename(ov)
            self.ov = OpenVinoYolo(ov, cfg.watki)
            self.names = self.ov.names
            src = f"{ov.name}, wątki: {cfg.watki or 'auto'}"
            wejscie = f"{self.ov.in_w}x{self.ov.in_h}"
        else:
            import torch
            torch.set_num_threads(max(1, cfg.watki))
            from ultralytics import YOLO
            self.ov = None
            self.model = YOLO(pt.name)
            self.imgsz = cfg.imgsz
            dummy = np.zeros((720, 1280, 3), np.uint8)
            self.model.predict(dummy, imgsz=self.imgsz, device="cpu", verbose=False)
            self.names = {int(k): v for k, v in self.model.names.items()}
            src, wejscie = pt.name, str(cfg.imgsz)

        wanted = {n.lower() for n in cfg.klasy}
        self.cls = [i for i, n in self.names.items() if n.lower() in wanted]
        unknown = wanted - {n.lower() for n in self.names.values()}
        if unknown:
            log.warning("Nieznane klasy (pomijam): %s", ", ".join(sorted(unknown)))
        if not self.cls:
            sys.exit("Żadna z klas z konfiguracji nie występuje w modelu.")

        dummy = np.zeros((576, 1024, 3), np.uint8)
        self._run(dummy)                                     # rozgrzewka
        t0 = time.perf_counter()
        for _ in range(5):
            self._run(dummy)
        ms = (time.perf_counter() - t0) / 5 * 1000
        log.info("Model %s (%s), klasy: %s, wejście %s – ok. %.0f ms na klatkę",
                 src, cfg.silnik, ", ".join(self.names[i] for i in self.cls), wejscie, ms)
        self.ms_total, self.n = 0.0, 0
        self.lock = threading.Lock()

    def _run(self, frame):
        lock = getattr(self, "lock", None)
        if lock is not None:
            with lock:
                return self._run_unlocked(frame)
        return self._run_unlocked(frame)

    def _run_unlocked(self, frame):
        if self.ov is not None:
            return self.ov(frame, self.cls, self.cfg.conf)
        r = self.model.predict(frame, classes=self.cls, conf=self.cfg.conf, imgsz=self.imgsz,
                               device="cpu", verbose=False)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return []
        xyxy = b.xyxy.cpu().numpy()
        conf = b.conf.cpu().numpy()
        cls = b.cls.cpu().numpy().astype(int)
        return [(int(x1), int(y1), int(x2), int(y2), float(c), self.names[int(k)])
                for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls)]

    def predict(self, frame):
        t0 = time.perf_counter()
        res = self._run(frame)
        self.ms_total += (time.perf_counter() - t0) * 1000
        self.n += 1
        return res


def in_zone(det, zone, w, h):
    if zone is None:
        return True
    x1, y1, x2, y2 = det[:4]
    pt = ((x1 + x2) / 2 / w, y2 / h)     # środek dolnej krawędzi = "stopy"
    return cv2.pointPolygonTest(zone, pt, False) >= 0


def detect(det, cam, frame):
    """Analiza z przycięciem obrazu (crop) i filtrem strefy (zone); ramki we współrzędnych klatki."""
    h, w = frame.shape[:2]
    crop = cam.crop
    if crop is not None:
        x1, y1 = int(crop[0] * w), int(crop[1] * h)
        x2, y2 = max(x1 + 2, int(round(crop[2] * w))), max(y1 + 2, int(round(crop[3] * h)))
        part = np.ascontiguousarray(frame[y1:y2, x1:x2])
        dets = [(a + x1, b + y1, c + x1, d + y1, p, lbl) for (a, b, c, d, p, lbl) in det.predict(part)]
    else:
        dets = det.predict(frame)
    return [d for d in dets if in_zone(d, cam.zone, w, h)]


def draw(frame, dets, zone=None, crop=None):
    img = frame.copy()
    h, w = img.shape[:2]
    if crop is not None:                       # obszar analizy – przerywana ramka
        x1, y1, x2, y2 = int(crop[0] * w), int(crop[1] * h), int(crop[2] * w) - 1, int(crop[3] * h) - 1
        for a, b in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
            n = max(1, int(np.hypot(b[0] - a[0], b[1] - a[1]) // 12))
            for i in range(0, n, 2):
                p = (int(a[0] + (b[0] - a[0]) * i / n), int(a[1] + (b[1] - a[1]) * i / n))
                q = (int(a[0] + (b[0] - a[0]) * (i + 1) / n), int(a[1] + (b[1] - a[1]) * (i + 1) / n))
                cv2.line(img, p, q, (60, 140, 255), 2)
    if zone is not None:
        pts = (zone * np.array([w, h], np.float32)).astype(np.int32)
        cv2.polylines(img, [pts], True, (255, 180, 0), 2)
    for (x1, y1, x2, y2, c, label) in dets:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(img, f"{label} {c:.2f}", (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
    return img


# ═════════════════════════════ galeria ═════════════════════════════

class Gallery:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dir = cfg.galeria
        self.lock = threading.Lock()
        try:
            for sub in ("miniatury", "klipy", "pelne"):
                (self.dir / sub).mkdir(parents=True, exist_ok=True)
            test = self.dir / ".zapis_test"
            test.write_text("ok")
            test.unlink()
        except OSError as e:
            sys.exit(f"Brak zapisu do katalogu galerii {self.dir}: {e}\n"
                     f"Utwórz go i nadaj uprawnienia: sudo mkdir -p {self.dir} && "
                     f"sudo chown {os.getenv('USER', 'UŻYTKOWNIK')}: {self.dir}")
        self.csv = self.dir / "wyniki.csv"
        self.rows = []
        if self.csv.exists():
            with open(self.csv, encoding="utf-8") as f:
                self.rows = [r for r in csv.DictReader(f) if r.get("poczatek")]

    def add(self, row):
        with self.lock:
            self.rows.append(row)
        self.save()

    def save(self):
        with self.lock:
            rows = list(self.rows)
        tmp = self.csv.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(rows)
        tmp.replace(self.csv)
        self.render(rows)

    def render(self, rows=None):
        if rows is None:
            with self.lock:
                rows = list(self.rows)
        rows = sorted(rows, key=lambda r: r["poczatek"], reverse=True)
        parts, day = [], None
        for r in rows:
            d = r["poczatek"][:10]
            if d != day:
                if day is not None:
                    parts.append("</div>")
                day = d
                cnt = sum(1 for x in rows if x["poczatek"][:10] == d)
                parts.append(f'<h2>{d} <small>({cnt})</small></h2><div class="g">')
            clip = r.get("klip") or ""
            link = clip if clip and (self.dir / clip).exists() else r["miniatura"]
            vid = " ▶" if link == clip and clip else ""
            parts.append(
                f'<figure><a href="{html.escape(link)}"><img loading="lazy" src="{html.escape(r["miniatura"])}"></a>'
                f'<figcaption><b>{r["poczatek"][11:]}–{r["koniec"][11:]}</b>{vid} · {html.escape(r["kamera"])}<br>'
                f'{html.escape(r.get("obiekty", ""))} · max: {r.get("max_obiektow", "")} · '
                f'pewność: {r.get("max_pewnosc", "")} · klatek: {r.get("trafien", "")}'
                + (f' · <a href="{html.escape(full)}">pełna klatka</a>'
                   if (full := (r.get("pelna") or "")) and (self.dir / full).exists() else "")
                + '</figcaption></figure>')
        if day is not None:
            parts.append("</div>")
        refresh = f'<meta http-equiv="refresh" content="{self.cfg.odswiezanie}">' if self.cfg.odswiezanie else ""
        now = datetime.now().strftime(TS_FMT)
        page = f"""<!doctype html><html lang="pl"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">{refresh}
<title>Wykrycia na kamerach</title>
<style>
body{{font-family:system-ui,sans-serif;margin:16px;background:#111;color:#ddd}}
h2{{margin:28px 0 10px;font-size:18px}} h2 small{{color:#888;font-weight:normal}}
.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px}}
figure{{margin:0;background:#1c1c1c;border-radius:8px;overflow:hidden}}
img{{width:100%;display:block}} figcaption{{padding:8px;font-size:14px;line-height:1.4}}
small{{color:#888}} a{{color:inherit}}
</style>
<h1>Wykrycia na kamerach ({len(rows)})</h1>
{'<p><a href="/admin/">⚙ Obszar detekcji kamer</a></p>' if self.cfg.port and self.cfg.http_user and self.cfg.http_password else ''}
<p><small>Aktualizacja: {now}. Kliknij miniaturę, aby otworzyć klip (▶). Przechowywane {self.cfg.retencja:g} dni.</small></p>
{''.join(parts)}</html>"""
        tmp = self.dir / ".index.tmp"
        tmp.write_text(page, encoding="utf-8")
        tmp.replace(self.dir / "index.html")

    def cleanup(self):
        cutoff = datetime.now() - timedelta(days=self.cfg.retencja)
        with self.lock:
            before = len(self.rows)
            kept = []
            for r in self.rows:
                try:
                    if datetime.strptime(r["poczatek"], TS_FMT) >= cutoff:
                        kept.append(r)
                except ValueError:
                    pass
            self.rows = kept
            removed_rows = before - len(kept)
        removed_files = 0
        for sub in ("miniatury", "klipy", "pelne"):
            for f in (self.dir / sub).iterdir():
                if not f.is_file():
                    continue
                m = NAME_DATE.search(f.name)
                try:
                    ts = (datetime.strptime(f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}", TS_FMT)
                          if m else datetime.fromtimestamp(f.stat().st_mtime))
                except (ValueError, OSError):
                    continue
                if ts < cutoff:
                    f.unlink(missing_ok=True)
                    removed_files += 1
        if removed_rows or removed_files:
            log.info("Retencja %g dni: usunięto %d wpisów i %d plików", self.cfg.retencja,
                     removed_rows, removed_files)
        self.save()


# ═════════════════════════════ klipy z bufora ═════════════════════════════

def save_image(cfg, img, path, quality):
    """Zapis JPG / WebP (OpenCV) albo JXL (ffmpeg + libjxl)."""
    ext = path.suffix.lower()
    if ext == ".jxl":
        h, w = img.shape[:2]
        # jakość 0–100 -> odległość Butteraugli libjxl (90 -> 1.0, wizualnie bezstratnie)
        dist = 0.1 + (100 - quality) * 0.09 if quality >= 30 else 6.4 + (30 - quality) ** 2 / 2250 * 6.4
        cmd = [cfg.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-i", "pipe:0",
               "-frames:v", "1", "-c:v", "libjxl", "-distance", f"{dist:.2f}", "-effort", "5",
               "-f", "image2", "-update", "1", str(path)]
        r = subprocess.run(cmd, input=np.ascontiguousarray(img).tobytes(), capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.decode(errors="replace").strip()[-300:])
    elif ext == ".webp":
        if not cv2.imwrite(str(path), img, [cv2.IMWRITE_WEBP_QUALITY, int(quality)]):
            raise RuntimeError("zapis WebP nieudany")
    else:
        if not cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)]):
            raise RuntimeError("zapis JPG nieudany")


def list_segments(d):
    segs = []
    try:
        for f in d.iterdir():
            m = SEG_RE.match(f.name)
            if m:
                try:
                    segs.append((datetime.strptime(m.group(1), "%Y%m%d-%H%M%S").timestamp(), f))
                except ValueError:
                    pass
    except FileNotFoundError:
        pass
    return sorted(segs)


class Clipper:
    def __init__(self, cfg, cams, gallery, det=None):
        self.cfg, self.cams, self.gallery, self.det = cfg, {c.name: c for c in cams}, gallery, det
        self.jobs = []        # słowniki: kind, cam, keep, due, ...
        self.lock = threading.Lock()

    def add(self, cam, start, end, out):
        with self.lock:
            self.jobs.append({"kind": "clip", "cam": cam.name, "keep": start, "due": end + self.cfg.seg + 2,
                              "start": start, "end": end, "out": out})

    def add_full(self, cam, t, out_full, out_thumb):
        with self.lock:
            self.jobs.append({"kind": "full", "cam": cam.name, "keep": t - 2, "due": t + self.cfg.seg + 3,
                              "t": t, "out": out_full, "thumb": out_thumb})

    def make(self, cam, start, end, out):
        segs = list_segments(cam.buf)
        chosen = [p for (t0, p), (t1, _) in zip(segs, segs[1:]) if t1 > start and t0 < end]
        if not chosen:
            log.warning("[%s] brak segmentów w buforze dla klipu %s", cam.name, out.name)
            return
        lst = cam.buf / f".lista_{out.stem}.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in chosen))
        cmd = [self.cfg.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-f", "concat", "-safe", "0", "-i", str(lst), "-map", "0:v", "-c", "copy"]
        if cam.codec in ("hevc", "h265"):
            cmd += ["-tag:v", "hvc1"]
        cmd += ["-movflags", "+faststart", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        lst.unlink(missing_ok=True)
        if r.returncode != 0:
            log.warning("[%s] klip nieudany %s: %s", cam.name, out.name, r.stderr.strip()[-300:])
            out.unlink(missing_ok=True)

    def _segments_precise(self, cam):
        """Segmenty z dokładnym czasem początku: koniec poprzedniego segmentu (mtime) zamiast
        nazwy pliku z dokładnością do sekundy."""
        segs = list_segments(cam.buf)
        out = []
        for i, (t0, p) in enumerate(segs):
            if i > 0:
                try:
                    prev_end = segs[i - 1][1].stat().st_mtime
                    if abs(prev_end - t0) < 2.5:
                        t0 = prev_end
                except OSError:
                    pass
            out.append((t0, p))
        return out

    def _grab_window(self, seg, a, b, step):
        """Klatki z segmentu w oknie [a, b) sekund od jego początku, co `step` s (jeden przebieg)."""
        info = probe(self.cfg, str(seg))
        if not info:
            return []
        w, h, _ = info
        # -ss po -i: dekodowanie od początku segmentu (zaczyna się klatką kluczową);
        # przewijanie przed -i w krótkich segmentach MPEG-TS nie zwraca klatek
        cmd = [self.cfg.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-threads", "2",
               "-i", str(seg), "-ss", f"{max(0.0, a):.2f}", "-t", f"{max(step, b - max(0.0, a)):.2f}",
               "-map", "0:v:0", "-vf", f"fps={1 / step:g}", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
        r = subprocess.run(cmd, capture_output=True)
        size = w * h * 3
        n = len(r.stdout) // size
        return [(max(0.0, a) + i * step, np.frombuffer(r.stdout[i * size:(i + 1) * size], np.uint8).reshape(h, w, 3))
                for i in range(n)]

    def make_full(self, cam, t, out_full, out_thumb, before=1.5, after=0.5, step=0.5):
        """Klatka w pełnej rozdzielczości ze strumienia głównego wokół momentu najlepszego wykrycia.
        Strumień główny i substream nie są zsynchronizowane co do klatki, więc analizowane jest
        kilka klatek z okna [t-before, t+after] i wybierana ta z najpewniejszym wykryciem."""
        segs = self._segments_precise(cam)
        a, b = t - before, t + after
        cands = []
        for (t0, p), (t1, _) in zip(segs, segs[1:]):          # bez ostatniego (otwartego) segmentu
            if t1 > a and t0 < b:
                cands += [(t0 + off, fr) for off, fr in self._grab_window(p, a - t0, min(b, t1) - t0, step)]
        if not cands:
            log.warning("[%s] brak klatek w buforze dla pełnej klatki %s", cam.name, out_full.name)
            return
        best = None                     # (pewność, czas, klatka, ramki)
        for ts, fr in cands:
            h, w = fr.shape[:2]
            dets = detect(self.det, cam, fr) if self.det else []
            score = max((d[4] for d in dets), default=0.0)
            key = (score, -abs(ts - t))
            if best is None or key > (best[0], -abs(best[1] - t)):
                best = (score, ts, fr, dets)
        _, ts, frame, dets = best
        h, w = frame.shape[:2]
        img = draw(frame, dets, cam.zone, cam.crop)
        try:
            save_image(self.cfg, img, out_full, self.cfg.full_q)
        except Exception as e:
            log.warning("[%s] zapis pełnej klatki nieudany (%s): %s", cam.name, out_full.name, e)
            return
        if dets:   # ostrzejsza miniatura z pełnej klatki (bez wykrycia zostaje miniatura z analizy)
            tw = min(self.cfg.thumb_w, w) if self.cfg.thumb_w > 0 else w
            th = int(round(h * tw / w))
            small = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA) if tw != w else img
            cv2.imwrite(str(out_thumb), small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        log.info("[%s] pełna klatka %dx%d (%+.1f s, z %d kandydatów) zapisana: %s – wykryto: %s", cam.name,
                 w, h, ts - t, len(cands), out_full.name,
                 ", ".join(f"{d[5]} {d[4]:.2f}" for d in dets) or "nic (miniatura z analizy)")

    def run_due(self, force=False):
        now = time.time()
        with self.lock:
            due = [j for j in self.jobs if force or now >= j["due"]]
            self.jobs = [j for j in self.jobs if j not in due]
        for j in sorted(due, key=lambda j: j["kind"] != "full"):
            cam = self.cams[j["cam"]]
            try:
                if j["kind"] == "full":
                    self.make_full(cam, j["t"], j["out"], j["thumb"])
                else:
                    self.make(cam, j["start"], j["end"], j["out"])
            except Exception:
                log.exception("[%s] błąd zadania %s", cam.name, j["kind"])
        if due:
            self.gallery.render()

    def prune_buffers(self):
        now = time.time()
        with self.lock:
            pending = list(self.jobs)
        for cam in self.cams.values():
            keep = now - (self.cfg.pre + 3 * self.cfg.seg + 30)
            ev = cam.event
            if ev is not None:
                keep = min(keep, ev["start"] - self.cfg.pre - self.cfg.seg)
            for j in pending:
                if j["cam"] == cam.name:
                    keep = min(keep, j["keep"] - self.cfg.seg)
            keep = max(keep, now - self.cfg.max_len - 600)   # górny limit bufora w RAM
            segs = list_segments(cam.buf)
            for (t0, p), (t1, _) in zip(segs, segs[1:]):
                if t1 < keep:
                    p.unlink(missing_ok=True)

    def loop(self):
        unpin()
        while not STOP.wait(2):
            try:
                self.run_due()
                self.prune_buffers()
            except Exception:
                log.exception("Błąd w obsłudze klipów")


# ═════════════════════════════ powiadomienia / WWW ═════════════════════════════

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


# ── edycja pliku konfiguracji (z zachowaniem komentarzy i układu) ──

CONFIG_LOCK = threading.Lock()
START_TIME = time.time()


def toml_value(v):
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    return f"{round(float(v), 4):g}"


def _bracket_depth(line):
    code = re.sub(r'"(?:\\.|[^"\\])*"|\'[^\']*\'', '""', line).split("#", 1)[0]
    return code.count("[") - code.count("]")


def toml_set_camera(text, cam_name, values):
    """Ustaw klucze (już sformatowane wartości) w bloku [[camera]] o danej nazwie."""
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    i = 0
    while i < len(lines):
        if lines[i].strip() != "[[camera]]":
            i += 1
            continue
        # granice bloku: do następnego nagłówka [...] poza wartością wieloliniową
        end, depth = len(lines), 0
        for j in range(i + 1, len(lines)):
            st = lines[j].strip()
            if depth == 0 and st.startswith("["):
                end = j
                break
            depth += _bracket_depth(lines[j])
        block = lines[i + 1:end]
        name = None
        for ln in block:
            m = re.match(r'^\s*name\s*=\s*"([^"]*)"', ln)
            if m:
                name = re.sub(r"[^\w-]", "_", m.group(1).strip())
        if name != cam_name:
            i = end
            continue
        for key, val in values.items():
            k, found = 0, False
            while k < len(block):
                if re.match(rf"^\s*{re.escape(key)}\s*=", block[k]):
                    d, e = _bracket_depth(block[k]), k + 1
                    while d > 0 and e < len(block):          # wartość na kilku liniach
                        d += _bracket_depth(block[e])
                        e += 1
                    block[k:e] = [f"{key} = {val}\n"]
                    found = True
                    break
                k += 1
            if not found:
                last = max((n for n, ln in enumerate(block) if ln.strip() and not ln.strip().startswith("#")),
                           default=-1)
                block.insert(last + 1, f"{key} = {val}\n")
        return "".join(lines[:i + 1] + block + lines[end:])
    raise KeyError(f"brak bloku [[camera]] o nazwie '{cam_name}'")


def save_camera_area(cfg, cam, zone, crop):
    """Zapisz zone/crop kamery w pliku konfiguracji (kopia .bak, zapis atomowy) i zastosuj od razu."""
    z = check_zone(zone)
    c = check_crop(crop)
    zone_s = toml_value([[round(float(x), 4), round(float(y), 4)] for x, y in z]) if z is not None else "[]"
    crop_s = toml_value([round(v, 4) for v in c]) if c is not None else "[]"
    with CONFIG_LOCK:
        path = cfg.path
        old = path.read_text(encoding="utf-8")
        new = toml_set_camera(old, cam.name, {"zone": zone_s, "crop": crop_s})
        data = tomllib.loads(new)                       # musi się parsować
        blk = next(b for b in data.get("camera", [])
                   if re.sub(r"[^\w-]", "_", str(b.get("name", "")).strip()) == cam.name)
        check_zone(blk.get("zone", []))
        check_crop(blk.get("crop", []))
        mode = path.stat().st_mode & 0o777
        bak = path.with_name(path.name + ".bak")
        bak.write_text(old, encoding="utf-8")
        os.chmod(bak, mode)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(new, encoding="utf-8")
        os.chmod(tmp, mode)
        tmp.replace(path)
    cam.zone, cam.crop = z, c                           # od razu, bez restartu
    log.info("[%s] zapisano obszar detekcji: zone = %s, crop = %s", cam.name, zone_s, crop_s)
    return f"zone = {zone_s}\ncrop = {crop_s}"


# ── panel WWW ──

ADMIN_HTML = r"""<!doctype html><html lang="pl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Obszar detekcji – szukaj_cpu</title>
<style>
:root{--bg:#111;--card:#1b1d22;--fg:#e6e6e6;--mut:#9aa0a6;--line:#2c2f36;--zone:#4ea1ff;--crop:#ff9f2e;--ok:#4cc38a;--err:#ff6b6b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:0 0 4px}.sub{color:var(--mut);margin:0 0 14px;font-size:13.5px}
a{color:var(--zone)}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0}
select,button{font:inherit;color:var(--fg);background:var(--card);border:1px solid var(--line);border-radius:8px;padding:7px 12px;cursor:pointer}
button:hover{border-color:#444}button.on{border-color:var(--zone);box-shadow:inset 0 0 0 1px var(--zone)}
button.crop.on{border-color:var(--crop);box-shadow:inset 0 0 0 1px var(--crop)}
button.pri{background:#24456e;border-color:#2f5a8f}button.warn{border-color:#6e3a24}
.stage{position:relative;background:#000;border-radius:10px;overflow:hidden;touch-action:none;user-select:none}
.stage img{display:block;width:100%;height:auto}.stage canvas{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair}
.help{color:var(--mut);font-size:13px;margin:8px 0}
pre{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;overflow-x:auto;font-size:13px}
#msg{min-height:1.4em;font-size:14px}.ok{color:var(--ok)}.err{color:var(--err)}
.leg{display:inline-flex;align-items:center;gap:6px;color:var(--mut);font-size:13px}
.leg i{display:inline-block;width:14px;height:3px;border-radius:2px}
details.intro{background:var(--card);border:1px solid var(--line);border-radius:10px;margin:12px 0;padding:0 14px}
details.intro summary{cursor:pointer;padding:11px 0;font-weight:600}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding-bottom:12px}
@media(max-width:720px){.cards{grid-template-columns:1fr}}
.card{border:1px solid var(--line);border-radius:8px;padding:10px 12px;background:#15171b}
.card h3{margin:0 0 6px;font-size:15px}
.card h3 i{width:14px;height:3px;border-radius:2px;display:inline-block;vertical-align:middle;margin-right:8px}
.card p,.card li{font-size:13.5px;color:#c9ccd1;margin:4px 0}.card ul{margin:4px 0;padding-left:18px}
.card svg{display:block;width:100%;max-width:260px;height:auto;margin:6px 0;border-radius:6px}
.tip{font-size:13.5px;color:#c9ccd1;padding-bottom:12px;margin:0}
#info{font-size:13.5px;margin:6px 0;padding:8px 12px;border-left:3px solid var(--line);background:var(--card);border-radius:0 8px 8px 0}
#info b{color:var(--fg)}#info .w{color:#ffc46b}
</style></head><body><main>
<h1>Obszar detekcji kamer</h1>
<p class="sub"><a href="/">← galeria</a> · zmiany działają od razu po zapisie i trafiają do pliku konfiguracji</p>
<div class="bar">
  <select id="cam"></select>
  <button id="mZone" class="on">Strefa (wielokąt)</button>
  <button id="mCrop" class="crop">Przycięcie obrazu (prostokąt)</button>
  <button id="refresh">↻ Odśwież obraz</button>
  <span class="leg"><i style="background:var(--zone)"></i>strefa</span>
  <span class="leg"><i style="background:var(--crop)"></i>przycięcie</span>
</div>
<details class="intro" id="intro" open>
<summary>Co to jest strefa i przycięcie? (kliknij, aby zwinąć)</summary>
<div class="cards">
 <div class="card">
  <h3><i style="background:var(--zone)"></i>Strefa – <em>gdzie</em> osoba ma się liczyć</h3>
  <svg viewBox="0 0 260 120" aria-hidden="true"><rect width="260" height="120" fill="#000"/>
   <polygon points="40,50 220,40 250,118 10,118" fill="rgba(78,161,255,.18)" stroke="#4ea1ff" stroke-width="2"/>
   <g fill="#e6e6e6"><circle cx="70" cy="22" r="5"/><rect x="65" y="28" width="10" height="20" rx="3"/></g>
   <g fill="#4cc38a"><circle cx="150" cy="62" r="6"/><rect x="144" y="69" width="12" height="26" rx="3"/></g>
   <text x="88" y="38" fill="#9aa0a6" font-size="11">poza strefą – pomijana</text>
   <text x="104" y="110" fill="#4cc38a" font-size="11">w strefie – alarm</text></svg>
  <ul><li>Model nadal analizuje <b>cały kadr</b>; strefa tylko filtruje wyniki.</li>
   <li>Liczy się miejsce <b>stóp</b> osoby (dół ramki), więc obrysuj podłoże: podjazd, ścieżkę, furtkę.</li>
   <li>Do czego: pomijanie przechodniów na chodniku i ulicy, sąsiedniej posesji.</li>
   <li>Brak strefy = liczy się cały kadr.</li></ul>
 </div>
 <div class="card">
  <h3><i style="background:var(--crop)"></i>Przycięcie – <em>co</em> model w ogóle ogląda</h3>
  <svg viewBox="0 0 260 120" aria-hidden="true"><rect width="260" height="120" fill="#000"/>
   <rect x="0" y="0" width="260" height="120" fill="rgba(0,0,0,.55)"/>
   <rect x="120" y="10" width="130" height="80" fill="#0d1a26" stroke="#ff9f2e" stroke-width="2" stroke-dasharray="7 5"/>
   <g fill="#e6e6e6"><circle cx="185" cy="38" r="4"/><rect x="181" y="43" width="8" height="15" rx="2"/></g>
   <g fill="#555"><circle cx="50" cy="60" r="6"/><rect x="44" y="67" width="12" height="26" rx="3"/></g>
   <text x="8" y="112" fill="#9aa0a6" font-size="11">poza prostokątem – niewidoczne</text>
   <text x="126" y="84" fill="#ffb35c" font-size="11">powiększony fragment</text></svg>
  <ul><li>Do modelu trafia <b>tylko fragment</b> w prostokącie, poza nim nikt nie zostanie wykryty.</li>
   <li>Mniejszy fragment jest powiększany, więc <b>dalekie, małe postacie</b> wykrywa się lepiej.</li>
   <li>Powiększenie działa, gdy prostokąt jest wyraźnie <b>węższy</b> od kadru; pas na całą szerokość nic nie powiększa.</li>
   <li>Brak przycięcia = model ogląda cały kadr (zalecane na start).</li></ul>
 </div>
</div>
<p class="tip"><b>Którą wybrać?</b> Chcesz ignorować ulicę, chodnik lub sąsiada? Użyj <b>strefy</b>. Interesuje Cię odległy fragment (np. furtka w głębi kadru) i osoby są tam za małe? Użyj <b>przycięcia</b>. Można łączyć oba: strefa działa wtedy w obrębie przyciętego fragmentu. Po zapisie sprawdź w galerii, czy zdarzenia się pojawiają.</p>
</details>
<div class="stage" id="stage"><img id="img" alt=""><canvas id="cv"></canvas></div>
<p class="help" id="help"></p>
<div id="info"></div>
<div class="bar">
  <button id="undo">Cofnij punkt</button>
  <button id="clearZone">Usuń strefę</button>
  <button id="clearCrop">Cały kadr (bez przycięcia)</button>
  <span style="flex:1"></span>
  <button id="save" class="pri">Zapisz i zastosuj</button>
  <button id="restart" class="warn">Restart usługi</button>
</div>
<div id="msg"></div>
<pre id="toml"></pre>
</main><script>
const $ = id => document.getElementById(id);
const HELP = {
  zone: "Kliknij, aby dodać narożnik strefy. Przeciągnij punkt, aby go przesunąć; prawy przycisk na punkcie usuwa go. Liczą się osoby, których stopy (dół ramki) są w strefie.",
  crop: "Przeciągnij, aby zaznaczyć prostokąt – model analizuje tylko ten fragment kadru (większe postacie = lepsza wykrywalność w oddali). Przeciągnij narożnik, aby zmienić rozmiar."
};
let model = null;
let cams = [], cam = null, mode = "zone", zone = [], crop = null, drag = null, restartOk = false;
const cv = $("cv"), ctx = cv.getContext("2d"), img = $("img");
const r4 = v => Math.round(Math.min(1, Math.max(0, v)) * 10000) / 10000;
function msg(t, cls) { $("msg").textContent = t; $("msg").className = cls || ""; }
async function api(path, body) {
  const o = body ? {method: "POST", headers: {"Content-Type": "application/json", "X-Szukaj": "1"}, body: JSON.stringify(body)} : {};
  const r = await fetch(path, o);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || r.status + " " + r.statusText);
  return j;
}
function tomlText() {
  const z = zone.length >= 3 ? "[" + zone.map(p => `[${r4(p[0])}, ${r4(p[1])}]`).join(", ") + "]" : "[]";
  const c = crop ? `[${crop.map(r4).join(", ")}]` : "[]";
  return `# [[camera]] name = "${cam ? cam.name : ""}"
zone = ${z}
crop = ${c}`;
}
function size() { const b = img.getBoundingClientRect(); cv.width = b.width * devicePixelRatio; cv.height = b.height * devicePixelRatio; }
function draw() {
  size();
  const W = cv.width, H = cv.height, d = devicePixelRatio;
  ctx.clearRect(0, 0, W, H);
  if (crop) {
    const [x1, y1, x2, y2] = [crop[0] * W, crop[1] * H, crop[2] * W, crop[3] * H];
    ctx.fillStyle = "rgba(0,0,0,.55)";
    ctx.fillRect(0, 0, W, y1); ctx.fillRect(0, y2, W, H - y2); ctx.fillRect(0, y1, x1, y2 - y1); ctx.fillRect(x2, y1, W - x2, y2 - y1);
    ctx.setLineDash([8 * d, 6 * d]); ctx.strokeStyle = "#ff9f2e"; ctx.lineWidth = 2 * d; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1); ctx.setLineDash([]);
    if (mode === "crop") for (const [x, y] of [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]) { ctx.fillStyle = "#ff9f2e"; ctx.fillRect(x - 5 * d, y - 5 * d, 10 * d, 10 * d); }
  }
  if (zone.length) {
    ctx.beginPath(); zone.forEach(([x, y], i) => i ? ctx.lineTo(x * W, y * H) : ctx.moveTo(x * W, y * H));
    if (zone.length >= 3) { ctx.closePath(); ctx.fillStyle = "rgba(78,161,255,.15)"; ctx.fill(); }
    ctx.strokeStyle = "#4ea1ff"; ctx.lineWidth = 2 * d; ctx.stroke();
    zone.forEach(([x, y]) => { ctx.beginPath(); ctx.arc(x * W, y * H, (mode === "zone" ? 6 : 3) * d, 0, 7); ctx.fillStyle = "#4ea1ff"; ctx.fill(); });
  }
  $("toml").textContent = tomlText();
  info();
}
function pct(v) { return Math.round(v * 100) + "%"; }
function info() {
  const out = [];
  if (zone.length >= 3) out.push(`<b>Strefa:</b> alarm tylko dla osób stojących w niebieskim obszarze.`);
  else if (zone.length) out.push(`<b>Strefa:</b> <span class="w">dodaj jeszcze ${3 - zone.length} pkt (min. 3), inaczej nie zostanie zapisana.</span>`);
  else out.push(`<b>Strefa:</b> brak – liczy się cały ${crop ? "przycięty fragment" : "kadr"}.`);
  if (crop) {
    const cw = crop[2] - crop[0], ch = crop[3] - crop[1];
    let s = `<b>Przycięcie:</b> model widzi ${pct(cw)} szerokości × ${pct(ch)} wysokości kadru; <span class="w">osoby poza pomarańczowym prostokątem nie będą wykrywane.</span>`;
    const W = img.naturalWidth, H = img.naturalHeight;
    if (model && W && H) {
      const g = Math.min(model[0] / (cw * W), model[1] / (ch * H)) / Math.min(model[0] / W, model[1] / H);
      s += g < 1.15
        ? ` Powiększenie ×${g.toFixed(1)}: <span class="w">postacie będą praktycznie tej samej wielkości co bez przycięcia. Zwęź prostokąt w poziomie, aby zyskać na wykrywalności, albo do samego ograniczania obszaru użyj strefy.</span>`
        : ` Powiększenie ×${g.toFixed(1)}: postacie w tym fragmencie są dla modelu ${g.toFixed(1)}× większe niż przy całym kadrze.`;
    }
    out.push(s);
  } else out.push(`<b>Przycięcie:</b> brak – model ogląda cały kadr.`);
  $("info").innerHTML = out.join("<br>");
}
function pos(e) { const b = cv.getBoundingClientRect(); return [(e.clientX - b.left) / b.width, (e.clientY - b.top) / b.height]; }
function near(p, q, px) { const b = cv.getBoundingClientRect(); return Math.hypot((p[0] - q[0]) * b.width, (p[1] - q[1]) * b.height) < px; }
cv.addEventListener("contextmenu", e => {
  e.preventDefault();
  if (mode !== "zone") return;
  const p = pos(e), i = zone.findIndex(q => near(p, q, 12));
  if (i >= 0) { zone.splice(i, 1); draw(); }
});
cv.addEventListener("pointerdown", e => {
  if (e.button === 2) return;
  cv.setPointerCapture(e.pointerId);
  const p = pos(e);
  if (mode === "zone") {
    const i = zone.findIndex(q => near(p, q, 14));
    if (i >= 0) drag = {t: "pt", i}; else { zone.push(p); drag = {t: "pt", i: zone.length - 1}; }
  } else {
    if (crop) {
      const cs = [[crop[0], crop[1]], [crop[2], crop[1]], [crop[2], crop[3]], [crop[0], crop[3]]];
      const k = cs.findIndex(q => near(p, q, 16));
      if (k >= 0) { drag = {t: "rect", ax: cs[(k + 2) % 4][0], ay: cs[(k + 2) % 4][1]}; draw(); return; }
    }
    drag = {t: "rect", ax: p[0], ay: p[1]}; crop = [p[0], p[1], p[0], p[1]];
  }
  draw();
});
cv.addEventListener("pointermove", e => {
  if (!drag) return;
  const [x, y] = pos(e).map(v => Math.min(1, Math.max(0, v)));
  if (drag.t === "pt") zone[drag.i] = [x, y];
  else crop = [Math.min(drag.ax, x), Math.min(drag.ay, y), Math.max(drag.ax, x), Math.max(drag.ay, y)];
  draw();
});
cv.addEventListener("pointerup", () => {
  if (drag && drag.t === "rect" && crop && (crop[2] - crop[0] < 0.05 || crop[3] - crop[1] < 0.05)) { crop = null; msg("Za mały prostokąt – min. 5% kadru.", "err"); }
  drag = null; draw();
});
function setMode(m) {
  mode = m; $("mZone").classList.toggle("on", m === "zone"); $("mCrop").classList.toggle("on", m === "crop");
  $("help").textContent = HELP[m]; $("undo").disabled = m !== "zone"; draw();
}
function loadCam() {
  cam = cams.find(c => c.name === $("cam").value);
  zone = (cam.zone || []).map(p => [...p]); crop = cam.crop && cam.crop.length ? [...cam.crop] : null;
  loadImg(); msg("");
}
function loadImg() {
  img.onerror = () => msg("Brak podglądu – kamera jeszcze nie dostarczyła klatki. Spróbuj za chwilę.", "err");
  img.onload = draw;
  img.src = "/api/snapshot?cam=" + encodeURIComponent(cam.name) + "&t=" + Date.now();
}
$("mZone").onclick = () => setMode("zone");
$("mCrop").onclick = () => setMode("crop");
$("refresh").onclick = loadImg;
$("undo").onclick = () => { zone.pop(); draw(); };
$("clearZone").onclick = () => { zone = []; draw(); };
$("clearCrop").onclick = () => { crop = null; draw(); };
$("cam").onchange = loadCam;
$("save").onclick = async () => {
  if (zone.length && zone.length < 3) return msg("Strefa musi mieć co najmniej 3 punkty (albo usuń ją).", "err");
  try {
    const j = await api("/api/area", {camera: cam.name, zone: zone.length >= 3 ? zone.map(p => p.map(r4)) : [], crop: crop ? crop.map(r4) : []});
    cam.zone = zone.map(p => [...p]); cam.crop = crop ? [...crop] : [];
    msg("Zapisano i zastosowano. Plik konfiguracji zaktualizowany (kopia: .bak).", "ok");
  } catch (e) { msg("Błąd zapisu: " + e.message, "err"); }
};
$("restart").onclick = async () => {
  if (!restartOk) return msg("Restart z poziomu WWW działa tylko, gdy szukaj_cpu jest uruchomione jako usługa systemd.", "err");
  if (!confirm("Zrestartować usługę szukaj_cpu? Analiza zatrzyma się na ok. 20 s.")) return;
  try { await api("/api/restart", {}); } catch (e) { return msg("Błąd: " + e.message, "err"); }
  msg("Restart… (ok. 20 s)");
  const t0 = Date.now();
  await new Promise(r => setTimeout(r, 4000));
  const poll = async () => {
    try { const s = await api("/api/status"); if (s.started * 1000 > t0 - 1000) return msg("Usługa uruchomiona ponownie.", "ok"); } catch (e) {}
    if (Date.now() - t0 > 120000) return msg("Usługa nie wróciła w 2 min – sprawdź: journalctl -u szukaj_cpu@…", "err");
    setTimeout(poll, 2000);
  };
  poll();
};
window.addEventListener("resize", draw);
(async () => {
  try {
    const j = await api("/api/cameras");
    cams = j.cameras; restartOk = j.restart; model = j.model || null;
    try { if (localStorage.getItem("szukaj_intro") === "0") $("intro").open = false; } catch (e) {}
    $("intro").addEventListener("toggle", () => { try { localStorage.setItem("szukaj_intro", $("intro").open ? "1" : "0"); } catch (e) {} });
    $("cam").innerHTML = cams.map(c => `<option>${c.name}</option>`).join("");
    if (!restartOk) $("restart").title = "Dostępne tylko przy uruchomieniu przez systemd";
    setMode("zone"); loadCam();
  } catch (e) { msg("Nie udało się wczytać kamer: " + e.message, "err"); }
})();
</script></body></html>"""


def start_http(cfg, cams=()):
    if not cfg.port:
        return
    cams_by_name = {c.name: c for c in cams}
    admin_on = bool(cfg.http_user and cfg.http_password)
    under_systemd = bool(os.environ.get("INVOCATION_ID"))

    class Handler(SimpleHTTPRequestHandler):
        server_version = "szukaj_cpu"
        sys_version = ""

        def log_message(self, *a):
            pass

        def _authorized(self):
            if not admin_on:
                return True
            h = self.headers.get("Authorization", "")
            if not h.startswith("Basic "):
                return False
            try:
                user, _, pw = base64.b64decode(h[6:]).decode("utf-8").partition(":")
            except Exception:
                return False
            return (hmac.compare_digest(user.encode(), cfg.http_user.encode())
                    & hmac.compare_digest(pw.encode(), cfg.http_password.encode()))

        def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
            data = body if isinstance(body, bytes) else (
                json.dumps(body, ensure_ascii=False) if not isinstance(body, str) else body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

        def _gate(self):
            """True = obsłużone (odmowa)."""
            if not self._authorized():
                self._send(401, "Wymagane logowanie", "text/plain; charset=utf-8",
                           {"WWW-Authenticate": 'Basic realm="szukaj_cpu", charset="UTF-8"'})
                return True
            return False

        def do_HEAD(self):
            if not self._gate():
                super().do_HEAD()

        def do_GET(self):
            if self._gate():
                return
            u = urllib.parse.urlparse(self.path)
            if not (u.path.startswith("/admin") or u.path.startswith("/api/")):
                return super().do_GET()
            if not admin_on:
                return self._send(403, "<!doctype html><meta charset=utf-8><p>Panel ustawień jest wyłączony. "
                                       "Ustaw <code>http_user</code> i <code>http_password</code> w sekcji "
                                       "<code>[gallery]</code> pliku konfiguracji i zrestartuj usługę.</p>",
                                  "text/html; charset=utf-8")
            if u.path == "/admin":
                return self._send(301, "", "text/plain", {"Location": "/admin/"})
            if u.path == "/admin/":
                return self._send(200, ADMIN_HTML, "text/html; charset=utf-8")
            if u.path == "/api/status":
                return self._send(200, {"ok": True, "started": START_TIME})
            if u.path == "/api/cameras":
                out = []
                for c in cams:
                    out.append({"name": c.name,
                                "zone": c.zone.round(4).tolist() if c.zone is not None else [],
                                "crop": list(c.crop) if c.crop is not None else []})
                return self._send(200, {"cameras": out, "restart": under_systemd,
                                        "model": [cfg.input_hw[1], cfg.input_hw[0]]})
            if u.path == "/api/snapshot":
                c = cams_by_name.get(urllib.parse.parse_qs(u.query).get("cam", [""])[0])
                snap = c.snap if c else None
                if snap is None:
                    return self._send(404, {"error": "brak klatki"})
                ok, jpg = cv2.imencode(".jpg", snap, [cv2.IMWRITE_JPEG_QUALITY, 85])
                return self._send(200, jpg.tobytes(), "image/jpeg")
            return self._send(404, {"error": "nie ma takiej strony"})

        def do_POST(self):
            if self._gate():
                return
            u = urllib.parse.urlparse(self.path)
            if not admin_on:
                return self._send(403, {"error": "panel wyłączony (brak http_user/http_password)"})
            # ochrona przed wysłaniem formularza z obcej strony (CSRF): wymagany nagłówek i JSON
            if self.headers.get("X-Szukaj") != "1" or not self.headers.get("Content-Type", "").startswith("application/json"):
                return self._send(400, {"error": "nieprawidłowe zapytanie"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(min(n, 65536)) or b"{}")
            except (ValueError, json.JSONDecodeError):
                return self._send(400, {"error": "nieprawidłowy JSON"})
            if u.path == "/api/area":
                c = cams_by_name.get(str(body.get("camera", "")))
                if c is None:
                    return self._send(404, {"error": "nie ma takiej kamery"})
                try:
                    snippet = save_camera_area(cfg, c, body.get("zone") or [], body.get("crop") or [])
                except (ValueError, KeyError, tomllib.TOMLDecodeError, StopIteration) as e:
                    return self._send(400, {"error": str(e)})
                except OSError as e:
                    return self._send(500, {"error": f"zapis pliku nieudany: {e}"})
                return self._send(200, {"ok": True, "toml": snippet})
            if u.path == "/api/restart":
                if not under_systemd:
                    return self._send(409, {"error": "restart działa tylko przy uruchomieniu przez systemd"})
                log.info("Restart usługi zlecony z panelu WWW (%s)", self.client_address[0])
                self._send(202, {"ok": True})
                # wyjście procesu – systemd (Restart=always) uruchomi usługę ponownie
                threading.Timer(0.5, STOP.set).start()
                return
            return self._send(404, {"error": "nie ma takiej operacji"})

    srv = ThreadingHTTPServer(("0.0.0.0", cfg.port),
                              functools.partial(Handler, directory=str(cfg.galeria)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log.info("Galeria WWW: http://<adres-serwera>:%d/%s", cfg.port,
             "  (panel obszaru detekcji: /admin/, logowanie)" if admin_on
             else "  (panel /admin/ wyłączony – ustaw http_user i http_password)")


# ═════════════════════════════ zdarzenia ═════════════════════════════

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


# ═════════════════════════════ tryb testowy ═════════════════════════════

def run_test(cfg, det, cams, gallery):
    ok = True
    for cam in cams:
        deadline = time.time() + 120
        item = None
        while time.time() < deadline and item is None:
            item = cam.take()
            if item is None:
                time.sleep(0.2)
        if item is None:
            log.error("[%s] TEST: brak klatki w 120 s (%s)", cam.name, cam.err[-1] if cam.err else "brak błędu")
            ok = False
            continue
        t, frame = item
        t0 = time.perf_counter()
        dets = detect(det, cam, frame)
        ms = (time.perf_counter() - t0) * 1000
        out = gallery.dir / f"test_{cam.name}.jpg"
        cv2.imwrite(str(out), draw(frame, dets, cam.zone, cam.crop))
        log.info("[%s] TEST OK: %dx%d %s -> %dx%d, dekodowanie %s, inferencja %.0f ms, wykryto: %s -> %s",
                 cam.name, *cam.src_size, cam.codec, *cam.size, cam.mode, ms,
                 ", ".join(f"{d[5]} {d[4]:.2f}" for d in dets) or "nic", out)
        if cfg.shinobi and cam.monitor:
            body = shinobi_event(cfg, cam, frame, dets, reason="test", wait=True)
            if body is not None:
                log.info("[%s] TEST Shinobi: zdarzenie wysłane do monitora %s, odpowiedź: %s",
                         cam.name, cam.monitor, body.strip()[:200])
            else:
                ok = False
        elif cfg.shinobi:
            log.info("[%s] TEST Shinobi: brak shinobi_monitor w bloku [[camera]] – pomijam", cam.name)
    return ok


# ═════════════════════════════ main ═════════════════════════════

def find_template():
    for p in (SCRIPT_DIR / "szukaj_cpu.toml.example",              # instalacja /opt/szukaj_cpu
              SCRIPT_DIR.parent / "config" / "szukaj_cpu.toml"):     # repozytorium
        if p.is_file():
            return p
    return None


def init_config(path):
    """Utwórz katalog konfiguracji i plik z szablonu (bez nadpisywania istniejącego)."""
    path = Path(path)
    if path.exists():
        print(f"{path} już istnieje – nie nadpisuję.")
        return 0
    tpl = find_template()
    if tpl is None:
        print("Nie znaleziono szablonu konfiguracji (szukaj_cpu.toml.example).", file=sys.stderr)
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / "models").mkdir(exist_ok=True)
    os.chmod(path.parent, 0o700)
    shutil.copyfile(tpl, path)
    os.chmod(path, 0o600)                       # hasła kamer, klucze API
    print(f"Utworzono {path} (z {tpl}).")
    legacy = Path(os.path.expanduser("~/szukaj_cpu/szukaj_cpu.toml"))
    if legacy.is_file():
        print(f"Uwaga: istnieje stara instalacja ({legacy.parent}). Przenieś ją skryptem uninstall.sh – "
              "skopiuje konfigurację i modele do ~/.szukaj_cpu.")
    print("Uzupełnij adresy kamer i ścieżki, potem: szukaj-cpu --test")
    return 0


def rss_mb(pid="self"):
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def log_memory(cams):
    """Pamięć procesu, procesów ffmpeg i bufora segmentów w /dev/shm (liczony do pamięci usługi)."""
    ff = sum(rss_mb(p.pid) for c in cams for p in (c.proc, c.rec_proc) if p is not None and p.poll() is None)
    buf, n = 0, 0
    for c in cams:
        for _, p in list_segments(c.buf):
            try:
                buf += p.stat().st_size
                n += 1
            except OSError:
                pass
    log.info("Pamięć: szukaj_cpu %.0f MB, ffmpeg %.0f MB, bufor klipów %.0f MB (%d segmentów)",
             rss_mb(), ff, buf / 1048576, n)


def hard_exit(code):
    """Zakończenie bez sprzątania wątków C++ OpenVINO przy zamykaniu interpretera – inaczej
    kończy się to 'terminate called without an active exception' (SIGABRT)."""
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def main():
    ap = argparse.ArgumentParser(description="Wykrywanie osób na żywo z kamer RTSP (YOLO na CPU).")
    ap.add_argument("-c", "--config", default=DEFAULT_CONF, help=f"plik konfiguracji (domyślnie {DEFAULT_CONF})")
    ap.add_argument("--test", action="store_true", help="sprawdź kamery i model, zapisz test_<kamera>.jpg i zakończ")
    ap.add_argument("--init", action="store_true", help="utwórz plik konfiguracji z szablonu i zakończ")
    ap.add_argument("--version", action="version", version=f"szukaj_cpu {__version__}")
    a = ap.parse_args()
    if a.init:
        sys.exit(init_config(expand(a.config)))
    if not expand(a.config).exists():
        sys.exit(f"Brak pliku konfiguracji {expand(a.config)}. Utwórz go: szukaj-cpu --init")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout)
    cfg = load_config(expand(a.config))
    if cfg.logfile:
        cfg.logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(cfg.logfile, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"))
        log.addHandler(fh)

    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, lambda *_: STOP.set())

    log.info("ffmpeg: %s (%s), ffprobe: %s", cfg.ffmpeg, tool_version(cfg.ffmpeg), cfg.ffprobe)
    log.info("Start: %d kamer, galeria %s, Shinobi API: %s", len(cfg.cams), cfg.galeria,
             cfg.sh_url if cfg.shinobi else "wyłączone")
    gallery = Gallery(cfg)
    det = Detector(cfg)
    cams = [Camera(s, cfg) for s in cfg.cams]
    for cam in cams:                       # stare segmenty z poprzedniego uruchomienia są bezużyteczne
        for _, p in list_segments(cam.buf):
            p.unlink(missing_ok=True)
    threads = [threading.Thread(target=cam.loop, name=cam.name, daemon=True) for cam in cams]
    if cfg.clips:
        threads += [threading.Thread(target=cam.rec_loop, name=f"{cam.name}-rec", daemon=True)
                    for cam in cams if cam.separate]
    for th in threads:
        th.start()

    if a.test:
        ok = run_test(cfg, det, cams, gallery)
        STOP.set()
        for cam in cams:
            cam.stop()
        hard_exit(0 if ok else 1)

    clipper = Clipper(cfg, cams, gallery, det)
    threading.Thread(target=clipper.loop, name="klipy", daemon=True).start()
    start_http(cfg, cams)
    load_static(cfg, cams)
    gallery.cleanup()
    last_cleanup = last_stats = time.time()

    while not STOP.is_set():
        busy = False
        for cam in cams:
            item = cam.take()
            if item is not None:
                busy = True
                try:
                    process(cfg, det, cam, *item)
                except Exception:
                    log.exception("[%s] błąd analizy klatki", cam.name)
        now = time.time()
        for cam in cams:
            if cam.event is not None and now - cam.event["end"] > cfg.gap:
                close_event(cfg, cam, gallery, clipper)
            cam.watchdog(now)
        if now - last_cleanup > 3600:
            gallery.cleanup()
            last_cleanup = now
        if now - last_stats > 300:
            el = now - last_stats
            for cam in cams:
                every = f"co {el / cam.frames:.1f} s" if cam.frames else "brak"
                log.info("[%s] %s, %dx%d, klatki do analizy: %s, pominięte: %d", cam.name, cam.mode,
                         *cam.size, every, cam.dropped)
                cam.frames = cam.dropped = 0
            if det.n:
                log.info("Inferencja: śr. %.0f ms/klatkę (%d klatek)", det.ms_total / det.n, det.n)
                det.ms_total, det.n = 0.0, 0
            log_memory(cams)
            save_static(cfg, cams)
            last_stats = now
        if not busy:
            STOP.wait(0.05)

    log.info("Zatrzymywanie…")
    for cam in cams:
        cam.stop()
        if cam.event is not None:
            close_event(cfg, cam, gallery, clipper)
    clipper.run_due(force=True)
    save_static(cfg, cams)
    log.info("Zatrzymano.")
    hard_exit(0)


if __name__ == "__main__":
    main()
