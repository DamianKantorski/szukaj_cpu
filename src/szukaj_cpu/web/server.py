"""Serwer WWW: galeria, panel obszaru detekcji (/admin/) i konfiguracji (/admin/config)."""

import base64
import functools
import hmac
import json
import os
import threading
import time
import tomllib
import urllib.parse
from http.server import SimpleHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path

import cv2

from ..common import CONFIG_LOCK, START_TIME, STOP, log
from .models import delete_model, install_zip, models_view
from .live import allowed_streams, check_token, live_info, player_script, ws_tunnel
from .recordings import day_view, days_view, resolve_file, send_file
from .config_form import find_record_dirs, config_version, config_view, plan_config, probe_stream, save_camera_area, validate_config_text, write_config


_HERE = Path(__file__).resolve().parent
ADMIN_HTML = (_HERE / "area.html").read_text(encoding="utf-8")
CONFIG_HTML = (_HERE / "config.html").read_text(encoding="utf-8")
REC_HTML = (_HERE / "nagrania.html").read_text(encoding="utf-8")
LIVE_HTML = (_HERE / "live.html").read_text(encoding="utf-8")


def homekit_view(hk, cfg):
    if hk is None:
        return {"enabled": cfg.hk, "cams": [], "error": None}
    st = hk.status()
    for c in st["cams"]:
        if c.get("uri") and not c.get("paired"):
            try:
                import segno
                c["qr"] = segno.make(c["uri"], error="m").svg_inline(scale=5, border=2, dark="#000", light="#fff")
            except Exception:
                c["qr"] = ""
        c.pop("uri", None)
    return st


def start_http(cfg, cams=(), det=None, gallery=None, g2r=None, hk=None):
    if not cfg.port:
        return
    cams_by_name = {c.name: c for c in cams}
    admin_on = bool(cfg.http_user and cfg.http_password)
    under_systemd = bool(os.environ.get("INVOCATION_ID"))

    fails = {}                       # adres klienta -> [czasy nieudanych logowań]
    fails_lock = threading.Lock()
    window = cfg.login_lockout * 60

    class Handler(SimpleHTTPRequestHandler):
        server_version = "szukaj_cpu"
        sys_version = ""

        def _client(self):
            """Adres klienta; za lokalnym proxy (Tailscale Serve, nginx) – z X-Forwarded-For."""
            ip = self.client_address[0]
            if ip in ("127.0.0.1", "::1"):
                if self.headers.get("X-Forwarded-For"):
                    return self.headers["X-Forwarded-For"].split(",")[0].strip()
                if self.headers.get("Tailscale-User-Login"):
                    return "ts:" + self.headers["Tailscale-User-Login"]
            return ip

        def _locked(self, ip):
            """Ile sekund blokady zostało (0 = brak)."""
            if not cfg.login_max_fail:
                return 0
            now = time.monotonic()
            with fails_lock:
                t = [x for x in fails.get(ip, []) if now - x < window]
                if t:
                    fails[ip] = t
                else:
                    fails.pop(ip, None)
                return int(window - (now - t[-cfg.login_max_fail])) + 1 if len(t) >= cfg.login_max_fail else 0

        def _failed(self, ip):
            if not cfg.login_max_fail:
                return
            with fails_lock:
                fails.setdefault(ip, []).append(time.monotonic())
                n = len(fails[ip])
                if len(fails) > 1000:                    # ochrona pamięci
                    fails.pop(next(iter(fails)))
            if n == cfg.login_max_fail:
                log.warning("Panel WWW: %d nieudanych logowań z %s – logowanie zablokowane na %g min",
                            n, ip, cfg.login_lockout)

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
            if not admin_on:
                return False
            ip = self._client()
            left = self._locked(ip)
            if left:
                self._send(429, f"Za dużo nieudanych prób logowania. Spróbuj ponownie za {max(1, round(left / 60))} min.",
                           "text/plain; charset=utf-8", {"Retry-After": str(left)})
                return True
            if not self._authorized():
                if self.headers.get("Authorization"):
                    self._failed(ip)
                self._send(401, "Wymagane logowanie", "text/plain; charset=utf-8",
                           {"WWW-Authenticate": 'Basic realm="szukaj_cpu", charset="UTF-8"'})
                return True
            return False

        def _config(self, body):
            dry = body.get("dry") is True
            with CONFIG_LOCK:
                try:
                    old, new, changes = plan_config(cfg.path, body)
                except (ValueError, tomllib.TOMLDecodeError) as e:
                    return self._send(400, {"error": str(e)})
                except OSError as e:
                    return self._send(500, {"error": f"odczyt konfiguracji nieudany: {e}"})
                if body.get("version") and body["version"] != config_version(old):
                    return self._send(409, {"error": "plik konfiguracji zmienił się w międzyczasie (np. zapis obszaru "
                                                     "detekcji lub edycja ręczna) – odśwież stronę i wprowadź zmiany "
                                                     "ponownie"})
                err, warns = validate_config_text(new, cfg.path) if new != old else (None, [])
                try:
                    nd = tomllib.loads(new).get("gallery", {})
                except tomllib.TOMLDecodeError as e:
                    nd, err = {}, err or str(e)
                notes = []
                new_port = nd.get("http_port", 0) if isinstance(nd, dict) else 0
                if new_port != cfg.port:
                    notes.append("Port panelu zmieni się na " + (str(new_port) if new_port else
                                 "0 – galeria WWW i ten panel zostaną WYŁĄCZONE") + " po restarcie.")
                if not (str(nd.get("http_user", "")).strip() and nd.get("http_password", "")):
                    notes.append("Bez loginu i hasła panel ustawień będzie wyłączony po restarcie "
                                 "(galeria bez logowania).")
                if dry:
                    return self._send(200, {"changes": changes, "error": err, "warnings": warns, "notes": notes})
                if err:
                    return self._send(400, {"error": err})
                if new == old:
                    return self._send(200, {"ok": True, "changes": [], "restarting": False})
                try:
                    write_config(cfg.path, old, new)
                except OSError as e:
                    return self._send(500, {"error": f"zapis pliku nieudany: {e}"})
            log.info("Konfiguracja zapisana z panelu WWW (%s): %s", self.client_address[0],
                     ", ".join(c["where"] for c in changes) or "bez zmian")
            restart = body.get("restart") is True and under_systemd
            self._send(200, {"ok": True, "changes": changes, "restarting": restart,
                             "new_port": new_port if new_port != cfg.port else None})
            if restart:
                log.info("Restart usługi po zmianie konfiguracji")
                threading.Timer(0.5, STOP.set).start()

        def do_HEAD(self):
            if self._gate():
                return
            if urllib.parse.urlparse(self.path).path.startswith("/rec/"):
                return self._rec_file()
            super().do_HEAD()

        def _rec_file(self):
            u = urllib.parse.urlparse(self.path)
            if not admin_on:
                return self._send(403, {"error": "nagrania dostępne tylko z włączonym logowaniem panelu"})
            parts = u.path.split("/")
            if len(parts) != 4:
                return self._send(404, {"error": "nie ma takiego nagrania"})
            p = resolve_file(cams, urllib.parse.unquote(parts[2]), urllib.parse.unquote(parts[3]))
            if p is None:
                return self._send(404, {"error": "nie ma takiego nagrania"})
            send_file(self, p, download="dl=1" in u.query)

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            if u.path == "/api/live/ws":                  # WebSocket podglądu – dostęp tokenem
                q = urllib.parse.parse_qs(u.query)
                src = q.get("src", [""])[0]
                if not admin_on or not check_token(q.get("token", [""])[0]):
                    return self._send(403, {"error": "brak dostępu – odśwież stronę podglądu"})
                if g2r is None or src not in allowed_streams(g2r):
                    return self._send(404, {"error": "nie ma takiego strumienia"})
                if self.headers.get("Upgrade", "").lower() != "websocket":
                    return self._send(400, {"error": "wymagane połączenie WebSocket"})
                return ws_tunnel(self, g2r, src)
            if self._gate():
                return
            if u.path.startswith("/rec/"):
                return self._rec_file()
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
            if u.path in ("/admin/config", "/admin/config/"):
                return self._send(200, CONFIG_HTML, "text/html; charset=utf-8")
            if u.path in ("/admin/live", "/admin/live/"):
                return self._send(200, LIVE_HTML, "text/html; charset=utf-8")
            if u.path == "/admin/live/video-rtc.js":
                if g2r is None:
                    return self._send(404, {"error": "go2rtc wyłączony"})
                try:
                    return self._send(200, player_script(g2r), "text/javascript; charset=utf-8")
                except OSError as e:
                    return self._send(502, {"error": f"go2rtc niedostępny: {e}"})
            if u.path == "/api/homekit":
                return self._send(200, homekit_view(hk, cfg))
            if u.path == "/api/live/info":
                return self._send(200, live_info(cams, g2r))
            if u.path in ("/admin/nagrania", "/admin/nagrania/"):
                return self._send(200, REC_HTML, "text/html; charset=utf-8")
            if u.path == "/api/rec/days":
                return self._send(200, days_view(cams, gallery))
            if u.path == "/api/rec/day":
                try:
                    return self._send(200, day_view(cfg, cams, gallery,
                                                    urllib.parse.parse_qs(u.query).get("date", [""])[0]))
                except ValueError as e:
                    return self._send(400, {"error": str(e)})
            if u.path == "/api/recdirs":
                try:
                    return self._send(200, find_record_dirs(cfg.path))
                except (OSError, tomllib.TOMLDecodeError) as e:
                    return self._send(500, {"error": str(e)})
            if u.path == "/api/models":
                return self._send(200, models_view(cfg, det))
            if u.path == "/api/config":
                try:
                    with CONFIG_LOCK:
                        view = config_view(cfg.path)
                except (OSError, tomllib.TOMLDecodeError) as e:
                    return self._send(500, {"error": f"nie można odczytać konfiguracji: {e}"})
                view["restart"] = under_systemd
                return self._send(200, view)
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
            if u.path == "/api/models/upload":            # plik .zip modelu (nie JSON)
                if self.headers.get("X-Szukaj") != "1":
                    return self._send(400, {"error": "nieprawidłowe zapytanie"})
                try:
                    n = int(self.headers.get("Content-Length", "0"))
                    name = install_zip(cfg, self.rfile, n)
                except ValueError as e:
                    self.close_connection = True
                    return self._send(400, {"error": str(e)})
                except OSError as e:
                    return self._send(500, {"error": f"zapis modelu nieudany: {e}"})
                return self._send(200, {"ok": True, "name": name, **models_view(cfg, det)})
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
            if u.path == "/api/config":
                return self._config(body)
            if u.path == "/api/models/delete":
                try:
                    delete_model(cfg, det, str(body.get("name", "")))
                except ValueError as e:
                    return self._send(400, {"error": str(e)})
                except OSError as e:
                    return self._send(500, {"error": str(e)})
                return self._send(200, {"ok": True, **models_view(cfg, det)})
            if u.path == "/api/probe":
                try:
                    return self._send(200, probe_stream(cfg, str(body.get("url", "")).strip(), body.get("camera")))
                except (ValueError, OSError, tomllib.TOMLDecodeError) as e:
                    return self._send(400, {"error": str(e)})
            if u.path == "/api/homekit/reset":
                try:
                    if hk is None:
                        raise ValueError("HomeKit jest wyłączony")
                    hk.reset(str(body.get("camera", "")))
                except ValueError as e:
                    return self._send(400, {"error": str(e)})
                return self._send(200, {"ok": True, "restart": True})
            if u.path == "/api/restart":
                if not under_systemd:
                    return self._send(409, {"error": "restart działa tylko przy uruchomieniu przez systemd"})
                log.info("Restart usługi zlecony z panelu WWW (%s)", self.client_address[0])
                self._send(202, {"ok": True})
                # wyjście procesu – systemd (Restart=always) uruchomi usługę ponownie
                threading.Timer(0.5, STOP.set).start()
                return
            return self._send(404, {"error": "nie ma takiej operacji"})

    import socket

    class Server6(ThreadingHTTPServer):
        address_family = socket.AF_INET6

    handler = functools.partial(Handler, directory=str(cfg.galeria))
    bound = []
    for addr in cfg.http_listen:
        try:
            srv = (Server6 if ":" in addr else ThreadingHTTPServer)((addr, cfg.port), handler)
        except OSError as e:
            log.error("Galeria WWW: nie można nasłuchiwać na %s:%d (%s)", addr, cfg.port, e)
            continue
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        bound.append(addr)
    if not bound:
        return
    log.info("Galeria WWW: http://%s:%d/%s", bound[0] if bound != ["0.0.0.0"] else "<adres-serwera>", cfg.port,
             "  (panel: /admin/live, /admin/nagrania, /admin/, /admin/config – logowanie)" if admin_on
             else "  (panel /admin/ wyłączony – ustaw http_user i http_password)")
