"""Kamery: polecenia ffmpeg, odbiór klatek, bufor segmentów, watchdog."""

import json
import os
import subprocess
import threading
import time
from collections import deque

import numpy as np

from .common import STOP, list_segments, log, mask_url


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


def record_output(cfg, cam):
    """Zapis ciągły na dysk: pliki co segment_minutes (pełne minuty zegara), nazwy jak w Shinobi.
    Plik w trakcie zapisu jest fragmentowanym MP4 w katalogu .nagrywanie (odporny na przerwanie),
    po zamknięciu Recorder przepisuje go (bez kodowania) do zwykłego MP4 obok."""
    out = ["-map", "0:v:0"]
    out += ["-map", "0:a:0?", "-c:a", "aac", "-b:a", "64k"] if cfg.rec_audio else ["-an"]
    out += ["-c:v", "copy", "-f", "segment", "-segment_time", str(cfg.rec_min * 60), "-segment_atclocktime", "1",
            "-reset_timestamps", "1", "-strftime", "1", "-segment_format", "mp4",
            "-segment_format_options", "movflags=+frag_keyframe+empty_moov+default_base_moof",
            str(cam.staging / "%Y-%m-%dT%H-%M-%S.mp4")]
    return out


def build_rec_cmd(cfg, cam):
    """Osobny proces na strumień główny (tylko kopia, bez dekodowania): bufor segmentów w RAM
    (klipy, pełne klatki) i/lub zapis ciągły na dysk – jedno połączenie, dwa wyjścia."""
    c = ([cfg.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
         + src_opts(cam.url) + ["-i", cam.url])
    if cam.rec_buffer:
        c += segment_output(cfg, cam)
    if cam.record:
        c += record_output(cfg, cam)
    return c


class Camera:
    def __init__(self, spec, cfg):
        self.name, self.url, self.zone, self.cfg = spec.name, spec.url, spec.zone, cfg
        self.monitor = spec.monitor
        self.analysis_url = spec.analysis_url
        self.crop = spec.crop
        self.snap = None            # ostatnia klatka analizy (podgląd w panelu WWW)
        self.separate = bool(spec.analysis_url) and spec.analysis_url != spec.url
        self.src_url, self.src_analysis = spec.url, spec.analysis_url     # adresy kamery (go2rtc zmienia url)
        self.record = getattr(spec, "record", False)
        self.record_dir = getattr(spec, "record_dir", None)
        self.staging = self.record_dir / ".nagrywanie" if self.record_dir else None
        self.rec_buffer = cfg.clips and self.separate      # bufor klipów w procesie nagrywania
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
            if STOP.wait(0.5):
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
        """Strumień główny – osobny proces ffmpeg (tylko kopia): bufor klipów i/lub zapis na dysk."""
        unpin()
        backoff = 5
        while not STOP.is_set():
            self.buf.mkdir(parents=True, exist_ok=True)
            if self.record:
                self.staging.mkdir(parents=True, exist_ok=True)
            started = time.time()
            self.rec_err.clear()
            proc = subprocess.Popen(build_rec_cmd(self.cfg, self), stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE)
            self.rec_proc = proc
            for line in proc.stderr:
                self.rec_err.append(line.decode(errors="replace").strip())
            proc.wait()
            self.rec_proc = None
            if STOP.wait(1):
                break
            if time.time() - started > 60:
                backoff = 5
            log.warning("[%s] zapis strumienia głównego przerwany po %.0f s – ponownie za %d s: %s",
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
            try:
                if self.rec_buffer:
                    newest = max((p.stat().st_mtime for _, p in list_segments(self.buf)), default=0)
                    limit = max(self.cfg.watchdog, 4 * self.cfg.seg)
                else:
                    newest = max((p.stat().st_mtime for p in self.staging.glob("*.mp4")), default=0)
                    limit = max(self.cfg.watchdog, 60)        # fragmenty MP4 dopisywane co klatkę kluczową
            except OSError:
                newest, limit = now, 1
            if newest and now - newest > limit:
                log.warning("[%s] zapis strumienia głównego nie rośnie od %.0f s – restart", self.name, now - newest)
                r.terminate()

    def stop(self):
        p = self.proc
        if p is not None and p.poll() is None:
            p.kill()
        r = self.rec_proc
        if r is not None and r.poll() is None:
            r.terminate()                 # ffmpeg zamyka bieżący plik nagrania
            try:
                r.wait(5)
            except subprocess.TimeoutExpired:
                r.kill()
