"""Podgląd na żywo: przekazywanie WebSocket przeglądarki do go2rtc (MSE) z kontrolą dostępu.

go2rtc słucha tylko na 127.0.0.1 – jego API pozwala m.in. dodawać źródła (także polecenia
exec), więc NIE wystawiamy go w sieci. Przeglądarka łączy się z serwerem szukaj_cpu, który
przepuszcza wyłącznie /api/ws?src=<znany strumień kamery> i statyczny odtwarzacz video-rtc.js.
Gniazdo WebSocket nie zawsze dostaje dane logowania Basic (np. Safari), dlatego zalogowana
strona pobiera krótki token (HMAC, ważny 24 h) i dołącza go do adresu.
"""

import hashlib
import hmac
import secrets
import select
import socket
import threading
import time
import urllib.parse
import urllib.request

from ..common import STOP, log

_SECRET = secrets.token_bytes(32)
TOKEN_TTL = 24 * 3600
_script = {}
_script_lock = threading.Lock()


def make_token():
    exp = str(int(time.time()) + TOKEN_TTL)
    sig = hmac.new(_SECRET, exp.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{exp}.{sig}"


def check_token(tok):
    exp, _, sig = (tok or "").partition(".")
    if not exp.isdigit() or int(exp) < time.time():
        return False
    good = hmac.new(_SECRET, exp.encode(), hashlib.sha256).hexdigest()[:32]
    return hmac.compare_digest(sig, good)


def allowed_streams(g2r):
    """{nazwa strumienia go2rtc: (kamera, 'hd'|'sd')}"""
    out = {}
    if g2r is None:
        return out
    for cam in g2r.cams:
        out[cam.name] = (cam.name, "hd")
        if g2r._sub(cam):
            out[f"{cam.name}_sub"] = (cam.name, "sd")
    return out


def _codecs(g2r):
    """Kodek obrazu każdego aktywnego strumienia wg go2rtc (np. H264, H265)."""
    res = {}
    try:
        streams = g2r.api("/api/streams")
    except (OSError, ValueError):
        return res
    for name, st in (streams or {}).items():
        for p in (st or {}).get("producers") or []:
            for m in p.get("medias") or []:
                parts = [x.strip() for x in m.split(",")]
                if parts and parts[0] == "video" and len(parts) > 2:
                    res[name] = parts[2].split("/")[0]
                    break
    return res


def live_info(cams, g2r):
    if g2r is None:
        return {"enabled": False, "cams": [],
                "reason": "Podgląd na żywo wymaga go2rtc (Konfiguracja → go2rtc → włączone) i kamer RTSP."}
    codecs = _codecs(g2r)
    allowed = allowed_streams(g2r)
    out = []
    for cam in cams:
        if cam.name not in allowed:
            continue
        sub = f"{cam.name}_sub"
        out.append({"name": cam.name, "hd": cam.name, "sd": sub if sub in allowed else None,
                    "hd_codec": codecs.get(cam.name), "sd_codec": codecs.get(sub),
                    "recording": bool(cam.record)})
    return {"enabled": True, "cams": out, "token": make_token(), "ttl": TOKEN_TTL}


def player_script(g2r):
    """video-rtc.js z go2rtc (wbudowany w program) – pobrany raz i trzymany w pamięci."""
    with _script_lock:
        if "js" not in _script:
            with urllib.request.urlopen(f"http://127.0.0.1:{g2r.cfg.g2r_api}/video-rtc.js", timeout=5) as r:
                _script["js"] = r.read()
        return _script["js"]


FORWARD = ("Upgrade", "Connection", "Sec-WebSocket-Key", "Sec-WebSocket-Version",
           "Sec-WebSocket-Extensions", "Sec-WebSocket-Protocol", "User-Agent")


def ws_tunnel(h, g2r, src):
    """Przełącza połączenie HTTP w surowy tunel do go2rtc /api/ws (go2rtc robi uzgodnienie WebSocket)."""
    try:
        up = socket.create_connection(("127.0.0.1", g2r.cfg.g2r_api), timeout=5)
    except OSError as e:
        return h._send(502, {"error": f"go2rtc niedostępny: {e}"})
    # bez nagłówka Origin – go2rtc odrzuca obce Origin; dostęp sprawdziliśmy sami
    lines = [f"GET /api/ws?src={urllib.parse.quote(src)} HTTP/1.1", f"Host: 127.0.0.1:{g2r.cfg.g2r_api}"]
    lines += [f"{k}: {h.headers[k]}" for k in FORWARD if h.headers.get(k)]
    h.close_connection = True
    cl = h.connection
    try:
        up.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1"))
        up.settimeout(None)
        cl.settimeout(None)
        socks = [cl, up]
        while not STOP.is_set():
            r, _, x = select.select(socks, [], socks, 30)
            if x:
                break
            for s in r:
                data = s.recv(65536)
                if not data:
                    return
                (up if s is cl else cl).sendall(data)
    except OSError as e:
        log.debug("podgląd %s: koniec połączenia (%s)", src, e)
    finally:
        try:
            up.close()
        except OSError:
            pass
