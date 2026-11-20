"""Uruchomienie: argumenty, pętla główna, tryb testowy, --init."""

import argparse
import logging
import os
import shutil
import signal
import sys
import threading
import time
from pathlib import Path

import cv2

from . import __version__
from .camera import Camera
from .clips import Clipper
from .common import DEFAULT_CONF, APP_DIR, STOP, expand, list_segments, log
from .config import load_config, tool_version
from .detector import Detector, detect, draw
from .events import close_event, load_static, process, save_static
from .go2rtc import Go2rtc
from .recorder import Recorder
from .gallery import Gallery
from .notify import shinobi_event
from .web.server import start_http


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


def find_template():
    for p in (APP_DIR / "szukaj_cpu.toml.example",              # instalacja /opt/szukaj_cpu
              APP_DIR.parent / "config" / "szukaj_cpu.toml"):     # repozytorium
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
    g2r = Go2rtc(cfg, cams) if cfg.g2r else None
    if g2r is not None and not g2r.start():
        g2r = None
    recorder = Recorder(cfg, cams)
    if not a.test:
        recorder.check()
    else:
        for cam in cams:
            cam.record = False
    threads = [threading.Thread(target=cam.loop, name=cam.name, daemon=True) for cam in cams]
    threads += [threading.Thread(target=cam.rec_loop, name=f"{cam.name}-rec", daemon=True)
                for cam in cams if cam.rec_buffer or cam.record]
    for th in threads:
        th.start()

    if a.test:
        ok = run_test(cfg, det, cams, gallery)
        STOP.set()
        for cam in cams:
            cam.stop()
        if g2r is not None:
            g2r.stop()
        hard_exit(0 if ok else 1)

    clipper = Clipper(cfg, cams, gallery, det)
    threading.Thread(target=clipper.loop, name="klipy", daemon=True).start()
    if recorder.cams:
        threading.Thread(target=recorder.loop, name="nagrania", daemon=True).start()
    start_http(cfg, cams, det, gallery, g2r)
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
    recorder.shutdown()
    if g2r is not None:
        g2r.stop()
    save_static(cfg, cams)
    log.info("Zatrzymano.")
    hard_exit(0)
