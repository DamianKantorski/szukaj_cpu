"""Serwer WWW: galeria, panel obszaru detekcji (/admin/) i konfiguracji (/admin/config)."""

import base64
import functools
import hmac
import json
import os
import threading
import tomllib
import urllib.parse
from http.server import SimpleHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path

import cv2

from ..common import CONFIG_LOCK, START_TIME, STOP, log
from .models import delete_model, install_zip, models_view
from .config_form import config_version, config_view, plan_config, probe_stream, save_camera_area, validate_config_text, write_config


_HERE = Path(__file__).resolve().parent
ADMIN_HTML = (_HERE / "area.html").read_text(encoding="utf-8")
CONFIG_HTML = (_HERE / "config.html").read_text(encoding="utf-8")


def start_http(cfg, cams=(), det=None):
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
            if u.path in ("/admin/config", "/admin/config/"):
                return self._send(200, CONFIG_HTML, "text/html; charset=utf-8")
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
             "  (panel ustawień: /admin/ i /admin/config, logowanie)" if admin_on
             else "  (panel /admin/ wyłączony – ustaw http_user i http_password)")
