"""HomeKit: kamery w aplikacji Dom i czujniki ruchu z detekcji (osoba, samochód…).

Każda kamera z `homekit = true` jest osobnym akcesorium HomeKit (HAP-python, własny port i kod
parowania). Obraz idzie do iPhone'a / Apple TV bez przekodowania: ffmpeg kopiuje H.264 ze
strumienia głównego (przez go2rtc) do SRTP. Klasy z `homekit_motion` są czujnikami ruchu
w tym samym akcesorium – Dom pokazuje wtedy powiadomienia ze zdjęciem z kamery.
Stan (klucze, sparowane urządzenia) i kody parowania są w ~/.szukaj_cpu/homekit/.
"""

import asyncio
import ipaddress
import json
import logging
import secrets
import threading
import time

import cv2

from . import __version__
from .common import CONF_DIR, STOP, log
from .detector import draw

try:
    from pyhap.accessory_driver import AccessoryDriver
    from pyhap.camera import (Camera, VIDEO_CODEC_PARAM_LEVEL_TYPES as LEVELS,
                              VIDEO_CODEC_PARAM_PROFILE_ID_TYPES as PROFILES)
    HAP_ERROR = None
except ImportError as e:                       # pakiet bez HAP-python
    HAP_ERROR = str(e)
    Camera = object

HK_DIR = CONF_DIR / "homekit"
LABELS_PL = {"person": "Osoba", "car": "Samochód", "truck": "Ciężarówka", "bus": "Autobus",
             "motorcycle": "Motocykl", "bicycle": "Rower", "dog": "Pies", "cat": "Kot"}
# kody, których HomeKit nie przyjmuje (za proste)
BAD_PINS = {"00000000", "11111111", "22222222", "33333333", "44444444", "55555555", "66666666",
            "77777777", "88888888", "99999999", "12345678", "87654321"}
SNAP_EVENT_S = 60          # tyle sekund po wykryciu zdjęcie z ramkami zdarzenia (do powiadomienia)


def _new_pin():
    while True:
        p = f"{secrets.randbelow(10 ** 8):08d}"
        if p not in BAD_PINS:
            return f"{p[:3]}-{p[3:5]}-{p[5:]}"


def xhm_uri(category, pin, setup_id):
    """Adres X-HM:// do kodu QR parowania (jak pyhap, bez pakietu base36)."""
    payload = ((category & 0xFF) << 31) | (2 << 27) | (int(pin.replace("-", "")) & 0x7FFFFFF)
    digits, n = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", payload
    out = ""
    while n:
        n, r = divmod(n, 36)
        out = digits[r] + out
    return "X-HM://" + (out or "0").rjust(9, "0") + setup_id


def _setup_id():
    abc = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "".join(secrets.choice(abc) for _ in range(4))


class HKCamera(Camera):
    def __init__(self, hk, cam, driver):
        options = {
            "video": {
                "codec": {
                    "profiles": [PROFILES["BASELINE"], PROFILES["MAIN"], PROFILES["HIGH"]],
                    "levels": [LEVELS["TYPE3_1"], LEVELS["TYPE3_2"], LEVELS["TYPE4_0"]],
                },
                # rozdzielczości, o które może prosić Dom; i tak wysyłamy strumień kamery bez zmian
                "resolutions": [[1920, 1080, 30], [1280, 720, 30], [640, 360, 30], [480, 270, 30],
                                [320, 180, 30], [1280, 960, 30], [1024, 768, 30], [640, 480, 30],
                                [480, 360, 30], [320, 240, 15]],
            },
            "audio": {"codecs": [{"type": "OPUS", "samplerate": 24}, {"type": "OPUS", "samplerate": 16}]},
            "srtp": True,
            "address": hk.address,
        }
        super().__init__(options, driver, cam.name)
        self.hk, self.cam = hk, cam
        self.set_info_service(firmware_revision=__version__, manufacturer="szukaj_cpu",
                              model="Kamera", serial_number=cam.name)
        self.motion = {}
        for cls in cam.homekit_motion:
            svc = self.add_preload_service("MotionSensor", chars=["Name"], unique_id=cls)
            svc.configure_char("Name", value=f"{cam.name} {LABELS_PL.get(cls, cls).lower()}")
            self.motion[cls] = svc.configure_char("MotionDetected", value=False)

    async def start_stream(self, session_info, stream_config):
        addr = stream_config["address"]
        try:
            ipaddress.ip_address(addr)
        except ValueError:
            log.warning("HomeKit [%s]: nieprawidłowy adres odbiorcy %r", self.cam.name, addr)
            return False
        cfg = self.hk.cfg
        v_port = int(stream_config["v_port"])
        mtu = int(stream_config.get("v_max_mtu") or 1316)
        audio = self.hk.has_audio(self.cam) and stream_config.get("a_port") and \
            str(stream_config.get("a_codec", "")).upper() == "OPUS"
        args = [str(cfg.ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin",
                "-rtsp_transport", "tcp", "-timeout", "10000000", "-i", self.hk.source(self.cam, audio),
                "-map", "0:v:0", "-an", "-c:v", "copy", "-bsf:v", "dump_extra",
                "-payload_type", str(int(stream_config.get("v_payload_type") or 99)),
                "-ssrc", str(int(stream_config["v_ssrc"])),
                "-f", "rtp", "-srtp_out_suite", "AES_CM_128_HMAC_SHA1_80",
                "-srtp_out_params", str(stream_config["v_srtp_key"]),
                f"srtp://{addr}:{v_port}?rtcpport={v_port}&localrtpport={v_port}&pkt_size={min(mtu, 1378)}"]
        if audio:
            # dźwięk kamery (G.711/L16/AAC) -> Opus, jak w integracji HomeKit Home Assistanta
            a_port = int(stream_config["a_port"])
            rate = int(stream_config.get("a_sample_rate") or 16)
            ptime = int(stream_config.get("a_packet_time") or 20)
            br = int(stream_config.get("a_max_bitrate") or 24)
            args += ["-map", "0:a:0", "-vn", "-c:a", "libopus", "-application", "lowdelay",
                     "-frame_duration", str(ptime if ptime in (10, 20, 40, 60) else 20),
                     "-ac", "1", "-ar", f"{rate}k", "-b:a", f"{br}k", "-bufsize", f"{br * 4}k",
                     "-payload_type", str(int(stream_config.get("a_payload_type") or 110)),
                     "-ssrc", str(int(stream_config["a_ssrc"])),
                     "-f", "rtp", "-srtp_out_suite", "AES_CM_128_HMAC_SHA1_80",
                     "-srtp_out_params", str(stream_config["a_srtp_key"]),
                     f"srtp://{addr}:{a_port}?rtcpport={a_port}&localrtpport={a_port}&pkt_size=188"]
        try:
            proc = await asyncio.create_subprocess_exec(*args, stdin=asyncio.subprocess.DEVNULL,
                                                        stdout=asyncio.subprocess.DEVNULL,
                                                        stderr=asyncio.subprocess.PIPE)
        except OSError as e:
            log.error("HomeKit [%s]: nie można uruchomić ffmpeg: %s", self.cam.name, e)
            return False
        session_info["process"] = proc
        log.info("HomeKit [%s]: podgląd dla %s (%sx%s zamówione, wysyłany strumień kamery bez zmian)",
                 self.cam.name, addr, stream_config.get("width"), stream_config.get("height"))
        asyncio.ensure_future(self._watch(proc))
        return True

    async def _watch(self, proc):
        err = await proc.stderr.read()
        rc = await proc.wait()
        if rc not in (0, 255, -15) and err:
            log.warning("HomeKit [%s]: ffmpeg zakończony (%s): %s", self.cam.name, rc,
                        err.decode(errors="replace").strip().splitlines()[-1][:300])

    async def stop_stream(self, session_info):
        proc = session_info.get("process")
        if proc is None or proc.returncode is not None:
            return
        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), 3)
        except ProcessLookupError:
            pass
        except asyncio.TimeoutError:
            proc.kill()
        log.info("HomeKit [%s]: koniec podglądu", self.cam.name)

    async def reconfigure_stream(self, session_info, stream_config):
        return True              # kopiujemy strumień – zmiana rozdzielczości/bitrate nas nie dotyczy

    def get_snapshot(self, image_size):
        frame = self.hk.snapshot(self.cam)
        if frame is None:
            return b""
        w = int(image_size.get("image-width") or 640)
        h0, w0 = frame.shape[:2]
        if w0 > w:
            frame = cv2.resize(frame, (w, max(1, round(h0 * w / w0))), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return jpg.tobytes() if ok else b""


class HomeKit:
    def __init__(self, cfg, cams, g2r=None):
        self.cfg, self.g2r = cfg, g2r
        self.cams = [c for c in cams if c.homekit]
        self.address = cfg.hk_address
        self.items = {}          # nazwa kamery -> {"driver", "acc", "thread", "pin", "port", "file"}
        self.events = {}         # nazwa kamery -> (czas, klatka z ramkami)
        self.motion_until = {}   # (kamera, klasa) -> czas wyłączenia czujnika
        self.lock = threading.Lock()

    def source(self, cam, audio=False):
        """Strumień główny: z go2rtc (bez nowego połączenia z kamerą) albo prosto z kamery."""
        if self.g2r is not None and cam.name in [c.name for c in self.g2r.cams]:
            # ?video – go2rtc nie uruchamia wtedy przekodowania dźwięku na AAC (podgląd w przeglądarce)
            return f"rtsp://127.0.0.1:{self.cfg.g2r_rtsp}/{cam.name}?video" + ("&audio" if audio else "")
        return cam.src_url

    def has_audio(self, cam):
        """Czy kamera wysyła dźwięk (wg sprawdzenia go2rtc przy starcie; bez go2rtc – zakładamy, że tak)."""
        if self.g2r is None:
            return True
        return bool(self.g2r.audio.get(cam.name))

    def snapshot(self, cam):
        ev = self.events.get(cam.name)
        if ev and time.time() - ev[0] < SNAP_EVENT_S:
            return ev[1]
        return cam.snap

    # ── start/stop ──
    def start(self):
        if not self.cams:
            return
        if HAP_ERROR:
            log.error("HomeKit: brak biblioteki HAP-python (%s) – HomeKit wyłączony", HAP_ERROR)
            return
        logging.getLogger("pyhap").setLevel(logging.WARNING)
        HK_DIR.mkdir(parents=True, exist_ok=True)
        if self.address == "auto":
            from pyhap import util
            self.address = util.get_local_address()
        for i, cam in enumerate(self.cams):
            try:
                self._start_one(cam, self.cfg.hk_port + i)
            except Exception:
                log.exception("HomeKit [%s]: nie udało się uruchomić akcesorium", cam.name)
        missing = sorted({k for c in self.cams for k in c.homekit_motion} - set(self.cfg.klasy))
        if missing:
            log.warning("HomeKit: czujniki %s nie zadziałają – tych klas nie ma w [detection] classes",
                        ", ".join(missing))
        threading.Thread(target=self._motion_loop, name="homekit-ruch", daemon=True).start()

    def _start_one(self, cam, port):
        meta_f = HK_DIR / f"{cam.name}.json"
        try:
            meta = json.loads(meta_f.read_text())
        except (OSError, ValueError):
            meta = {}
        if "pin" not in meta or "setup_id" not in meta:
            meta = {"pin": _new_pin(), "setup_id": _setup_id()}
            meta_f.write_text(json.dumps(meta))
            meta_f.chmod(0o600)
        state_f = HK_DIR / f"{cam.name}.state"
        driver = AccessoryDriver(port=port, address=self.address, persist_file=str(state_f),
                                 pincode=meta["pin"].encode())
        driver.state.setup_id = meta["setup_id"]
        acc = HKCamera(self, cam, driver)
        acc.setup_message = lambda: None          # kod jest w naszym logu i w panelu
        driver.add_accessory(acc)
        t = threading.Thread(target=driver.start, name=f"homekit-{cam.name}", daemon=True)
        t.start()
        self.items[cam.name] = {"driver": driver, "acc": acc, "thread": t, "pin": meta["pin"],
                                "port": port, "state": state_f, "meta": meta_f}
        log.info("HomeKit [%s]: akcesorium na %s:%d%s%s", cam.name, self.address, port,
                 f", czujniki: {', '.join(LABELS_PL.get(x, x) for x in cam.homekit_motion)}" if cam.homekit_motion else "",
                 "" if driver.state.paired else f" – NIESPAROWANE, kod {meta['pin']} (panel → HomeKit)")

    def stop(self):
        for it in self.items.values():
            d = it["driver"]
            try:
                d.loop.call_soon_threadsafe(d.loop.create_task, d.async_stop())
            except RuntimeError:
                pass
        for it in self.items.values():
            it["thread"].join(3)

    # ── czujniki ruchu ──
    def on_detection(self, cam, labels, frame, dets):
        """Wywoływane przy ogłoszeniu zdarzenia i przy każdej nowej klasie w trwającym zdarzeniu."""
        it = self.items.get(cam.name)
        if it is None:
            return
        hit = [x for x in labels if x in it["acc"].motion]
        if not hit:
            return
        try:
            self.events[cam.name] = (time.time(), draw(frame, dets, cam.zone, cam.crop))
        except Exception:
            self.events[cam.name] = (time.time(), frame)
        until = time.time() + self.cfg.hk_motion_s
        with self.lock:
            for cls in hit:
                was = self.motion_until.get((cam.name, cls), 0) > time.time()
                self.motion_until[(cam.name, cls)] = until
                if not was:
                    self._set(it, cls, True)
                    log.info("HomeKit [%s]: czujnik %s – ruch", cam.name, LABELS_PL.get(cls, cls))

    def keep_alive(self, cam, labels):
        """Obiekt nadal widoczny – czujnik zostaje włączony."""
        it = self.items.get(cam.name)
        if it is None:
            return
        until = time.time() + self.cfg.hk_motion_s
        with self.lock:
            for cls in labels:
                k = (cam.name, cls)
                if k in self.motion_until and self.motion_until[k] > time.time():
                    self.motion_until[k] = until

    def _set(self, it, cls, value):
        d = it["driver"]
        char = it["acc"].motion[cls]
        d.loop.call_soon_threadsafe(char.set_value, value)

    def _motion_loop(self):
        while not STOP.wait(1):
            now = time.time()
            with self.lock:
                for (name, cls), until in list(self.motion_until.items()):
                    if until <= now:
                        del self.motion_until[(name, cls)]
                        it = self.items.get(name)
                        if it:
                            self._set(it, cls, False)

    # ── panel WWW ──
    def status(self):
        out = []
        for cam in self.cams:
            it = self.items.get(cam.name)
            if it is None:
                out.append({"name": cam.name, "running": False})
                continue
            d = it["driver"]
            uri = xhm_uri(it["acc"].category, it["pin"], d.state.setup_id)
            out.append({"name": cam.name, "running": True, "paired": bool(d.state.paired),
                        "pin": it["pin"], "uri": uri, "port": it["port"], "address": self.address,
                        "motion": [{"cls": c, "label": LABELS_PL.get(c, c),
                                    "on": self.motion_until.get((cam.name, c), 0) > time.time()}
                                   for c in cam.homekit_motion],
                        "clients": len(d.state.paired_clients)})
        return {"enabled": self.cfg.hk, "error": HAP_ERROR, "cams": out}

    def reset(self, name):
        """Usuwa parowanie (i nowy kod) – działa po restarcie usługi."""
        cam = next((c for c in self.cams if c.name == name), None)
        if cam is None:
            raise ValueError("nie ma takiej kamery w HomeKit")
        for f in (HK_DIR / f"{name}.state", HK_DIR / f"{name}.json"):
            try:
                f.unlink()
            except FileNotFoundError:
                pass
        log.info("HomeKit [%s]: parowanie usunięte z panelu WWW – nowy kod po restarcie", name)
