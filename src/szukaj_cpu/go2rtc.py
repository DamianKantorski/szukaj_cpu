"""go2rtc: jedno połączenie z kamerą na strumień, dalej lokalny restream RTSP.

szukaj_cpu uruchamia go2rtc jako proces potomny z konfiguracją wygenerowaną z szukaj_cpu.toml.
Analiza, bufor klipów i zapis na dysk czytają z rtsp://127.0.0.1:8554/<kamera> (strumień
główny) i …/<kamera>_sub (substream) zamiast łączyć się z kamerą kilka razy.
Kolejne etapy (podgląd w przeglądarce, HomeKit) korzystają z tego samego procesu.
"""

import json
import os
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from .common import APP_DIR, CONF_DIR, STOP, log, mask_url


def find_go2rtc(setting):
    if setting not in ("", "auto"):
        p = Path(setting).expanduser()
        return p if p.is_file() and os.access(p, os.X_OK) else None
    for p in (CONF_DIR / "go2rtc", APP_DIR / "go2rtc"):
        if p.is_file() and os.access(p, os.X_OK):
            return p
    w = shutil.which("go2rtc")
    return Path(w) if w else None


def stream_names(cam):
    return cam.name, f"{cam.name}_sub"


class Go2rtc:
    def __init__(self, cfg, cams):
        # tylko kamery RTSP (pliki testowe czytamy bezpośrednio)
        self.cfg = cfg
        self.cams = [c for c in cams if c.src_url.lower().startswith(("rtsp://", "rtsps://"))]
        self.bin = find_go2rtc(cfg.g2r_bin)
        self.proc = None
        self.err = []
        self.conf = CONF_DIR / "go2rtc.yaml"
        self.host = "127.0.0.1"
        self.audio = {}             # strumień -> kodek dźwięku ("" = brak)

    def local_url(self, stream):
        return f"rtsp://{self.host}:{self.cfg.g2r_rtsp}/{stream}"

    def write_config(self):
        streams = {}
        for cam in self.cams:
            main, sub = stream_names(cam)
            streams[main] = self._sources(main, cam.src_url)
            if self._sub(cam):
                streams[sub] = self._sources(sub, cam.src_analysis)
        conf = {
            "api": {"listen": f"127.0.0.1:{self.cfg.g2r_api}"},
            "rtsp": {"listen": f"{self.cfg.g2r_listen}:{self.cfg.g2r_rtsp}"},
            "webrtc": {"listen": ""},          # podgląd w przeglądarce – kolejny etap
            "ffmpeg": {"bin": str(self.cfg.ffmpeg)},
            "log": {"level": "warn", "format": "text"},
            "streams": streams,
        }
        # JSON jest poprawnym YAML-em; plik zawiera hasła kamer -> tylko dla właściciela
        self.conf.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.conf.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(conf, f, indent=1)
        tmp.replace(self.conf)

    def _sources(self, name, url):
        """Źródło kamery, a gdy ma dźwięk inny niż AAC – drugie źródło z dźwiękiem przekodowanym na AAC.
        go2rtc uruchamia ten ffmpeg tylko na żądanie (podgląd w Safari/iOS, który nie odtwarza PCM/G.711
        jako FLAC). Bez dźwięku w kamerze NIE dodajemy go – go2rtc próbowałby go uruchamiać na próżno."""
        a = self.audio.get(name)
        return [url, f"ffmpeg:{name}#audio=aac"] if a and a != "aac" else url

    def probe_audio(self):
        """Kodek dźwięku każdego strumienia (ffprobe, równolegle, max ~8 s); wynik zapamiętany na wypadek,
        gdyby kamera była chwilowo niedostępna przy starcie."""
        cache = CONF_DIR / "go2rtc_audio.json"
        try:
            old = json.loads(cache.read_text())
        except (OSError, ValueError):
            old = {}
        jobs = {}
        for cam in self.cams:
            main, sub = stream_names(cam)
            jobs[main] = cam.src_url
            if self._sub(cam):
                jobs[sub] = cam.src_analysis
        res = {}

        def one(name, url):
            try:
                r = subprocess.run([str(self.cfg.ffprobe), "-v", "error", "-rtsp_transport", "tcp", "-timeout", "6000000",
                                    "-show_entries", "stream=codec_type,codec_name", "-of", "json", url],
                                   capture_output=True, text=True, timeout=10)
                streams = json.loads(r.stdout or "{}").get("streams", [])
                if r.returncode == 0 and streams:
                    res[name] = next((st.get("codec_name", "?") for st in streams if st.get("codec_type") == "audio"), "")
            except (OSError, ValueError, subprocess.SubprocessError):
                pass

        ths = [threading.Thread(target=one, args=kv, daemon=True) for kv in jobs.items()]
        for t in ths:
            t.start()
        for t in ths:
            t.join(12)
        self.audio = {k: res.get(k, old.get(k, "")) for k in jobs}
        if res:
            try:
                cache.write_text(json.dumps(self.audio))
            except OSError:
                pass
        aac = [k for k, v in self.audio.items() if v and v != "aac"]
        if aac:
            log.info("go2rtc: dźwięk dla podglądu przekodowywany na AAC (na żądanie): %s",
                     ", ".join(f"{k} ({self.audio[k]})" for k in aac))

    @staticmethod
    def _sub(cam):
        return bool(cam.src_analysis) and cam.src_analysis != cam.src_url and \
            cam.src_analysis.lower().startswith(("rtsp://", "rtsps://"))

    def api(self, path, timeout=2):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.cfg.g2r_api}{path}", timeout=timeout) as r:
            return json.load(r)

    def _spawn(self):
        self.proc = subprocess.Popen([str(self.bin), "-config", str(self.conf)], stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        threading.Thread(target=self._drain, args=(self.proc,), name="go2rtc-log", daemon=True).start()

    def _drain(self, proc):
        for line in proc.stdout:
            s = mask_url(line.decode(errors="replace").strip())
            if s and not (STOP.is_set() and "exit with signal" in s):
                self.err = (self.err + [s])[-20:]
                log.warning("go2rtc: %s", s)

    def start(self):
        """Uruchom go2rtc i przełącz kamery na lokalny restream. False = łączenie bezpośrednie."""
        if not self.cams:
            return False
        if self.bin is None:
            log.warning("go2rtc: nie znaleziono programu (pakiet: /opt/szukaj_cpu/go2rtc) – kamery łączą się "
                        "bezpośrednio, każdy proces osobno")
            return False
        self.probe_audio()
        self.write_config()
        self._spawn()
        for _ in range(50):
            try:
                self.api("/api/streams")
                break
            except OSError:
                if self.proc.poll() is not None:
                    break
                time.sleep(0.2)
        else:
            self.proc.kill()
        if self.proc.poll() is not None:
            log.error("go2rtc nie wystartował (%s) – kamery łączą się bezpośrednio",
                      self.err[-1] if self.err else f"kod {self.proc.returncode}")
            return False
        for cam in self.cams:
            main, sub = stream_names(cam)
            cam.url = self.local_url(main)
            cam.analysis_url = self.local_url(sub) if self._sub(cam) else ""
        log.info("go2rtc %s: restream rtsp://%s:%d/<kamera>, API 127.0.0.1:%d (%d strumieni)",
                 self.version(), self.cfg.g2r_listen, self.cfg.g2r_rtsp, self.cfg.g2r_api,
                 sum(2 if c.analysis_url else 1 for c in self.cams))
        threading.Thread(target=self._supervise, name="go2rtc", daemon=True).start()
        return True

    def version(self):
        try:
            return self.api("/api").get("version", "?")
        except (OSError, ValueError):
            return "?"

    def _supervise(self):
        backoff = 2
        while not STOP.is_set():
            code = self.proc.wait()
            if STOP.wait(1):            # zatrzymywanie usługi (sygnał dotarł też do go2rtc)
                break
            log.error("go2rtc zakończył się (kod %s) – ponowne uruchomienie za %d s", code, backoff)
            STOP.wait(backoff)
            backoff = min(backoff * 2, 60)
            if not STOP.is_set():
                self._spawn()

    def stop(self):
        p = self.proc
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(3)
            except subprocess.TimeoutExpired:
                p.kill()
