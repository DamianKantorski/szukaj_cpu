"""Konfigurator WWW: opis formularza, podgląd i zapis zmian, test strumienia."""

import json
import os
import re
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from ..common import COCO80, CONFIG_LOCK, NAME_DATE, expand, log, mask_url
from ..config import check_crop, check_zone, default_record_dir, load_config
from .toml_edit import CAMERA_KEYS, toml_apply, toml_set_camera, toml_value


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


COCO = COCO80


def F(key, label, typ, default, help="", **kw):
    return dict(key=key, label=label, type=typ, default=default, help=help, **kw)


FORM = [
    {"id": "detection", "title": "Wykrywanie", "desc": "Co i jak dokładnie wykrywa model.", "fields": [
        F("classes", "Wykrywane obiekty", "classes", ["person"],
          "Klasy COCO, o których usługa powiadamia. Zwykle wystarczy osoba."),
        F("confidence", "Minimalna pewność", "float", 0.5,
          "Wyżej = mniej fałszywych alarmów, ale łatwiej przeoczyć osobę w nocy.", min=0.05, max=0.95, step=0.05,
          slider=True),
        F("frame_selection", "Które klatki analizować", "select", "interval",
          "Klatki kluczowe = najmniej pracy, wymagają substreamu z I-frame co 1 s. "
          "Interwał = iGPU dekoduje cały strumień i co analysis_interval s wybiera klatkę.",
          options=[["keyframes", "tylko klatki kluczowe (I-frame)"], ["interval", "co określony czas"]]),
        F("analysis_interval", "Odstęp analizy", "float", 1.0, "Tylko w trybie „co określony czas”.",
          min=0.1, max=60, step=0.1, unit="s", show_if=["frame_selection", "interval"]),
        F("min_hits", "Klatki potrzebne do zdarzenia", "int", 1,
          "Ile klatek z wykryciem trzeba, aby zapisać zdarzenie. 1 = najczulej; 2 odfiltruje pojedyncze pomyłki.",
          min=1, max=20),
        F("event_gap", "Koniec zdarzenia po", "float", 10.0,
          "Zdarzenie kończy się po tylu sekundach bez wykrycia.", min=1, max=600, unit="s"),
        F("max_height", "Maksymalna wysokość klatki", "int", 1080,
          "Wyższe klatki są zmniejszane (np. 4K → 1080p). 0 = bez zmniejszania.", min=0, max=4320,
          options=[0, 720, 1080, 1440]),
        F("watchdog", "Ponowne połączenie po", "float", 30.0,
          "Gdy z kamery nie przychodzi klatka przez tyle sekund, usługa łączy się ponownie.", min=5, max=600, unit="s"),
    ]},
    {"id": "model", "section": "detection", "title": "Model",
     "desc": "Model YOLO (YOLO11 albo nowszy YOLO26) i jego wersja. INT8 jest szybszy; jeśli wykrywanie "
             "pogorszy się (np. w nocy), przełącz na FP32 albo na drugi model i zrestartuj usługę.",
     "fields": [
        F("precision", "Wersja modelu", "select", "fp32",
          "INT8 = model skwantyzowany (NNCF): szybsza analiza, minimalnie niższa dokładność. "
          "Gdy brak modelu INT8 dla wybranego rozmiaru, usługa użyje FP32 i zapisze ostrzeżenie w logu.",
          options=[["fp32", "standardowy (FP32) – najdokładniejszy"], ["int8", "INT8 – szybszy"]]),
        F("_models", "Dostępne modele", "models", None, ""),
        F("imgsz", "Rozmiar wejścia modelu", "int", 640,
          "Dłuższy bok obrazu dla modelu. 640 = szybko; 960 = lepiej widzi małe, dalekie postacie (~2× wolniej). "
          "Musi istnieć model o tym rozmiarze (lista wyżej) – najprościej wybrać go przyciskiem „Użyj”.",
          min=320, max=1920, step=32, options=[640, 768, 960, 1280]),
        F("aspect_ratio", "Proporcje wejścia modelu", "select", "16:9",
          "16:9 pasuje do kamer – bez szarych pasów, szybciej niż kwadrat.",
          options=[["16:9", "16:9 (zalecane)"], ["4:3", "4:3"], ["square", "kwadrat"]]),
        F("engine", "Silnik", "select", "openvino", "",
          options=[["openvino", "OpenVINO (zalecany na Intel)"], ["pytorch", "PyTorch (wymaga narzędzi eksportu)"]]),
        F("threads", "Wątki CPU na analizę", "int", 2, "0 = wszystkie rdzenie. Na i5-7500 najlepiej 4.",
          min=0, max=64),
    ]},
    {"id": "static", "section": "detection", "title": "Stojące pojazdy",
     "desc": "Zaparkowane auto nie wywołuje ciągle nowych zdarzeń. Przyjazd, odjazd i ruch nadal są zgłaszane.",
     "fields": [
        F("static_classes", "Dotyczy klas", "classes", ["car", "truck", "bus", "motorcycle", "bicycle"],
          "Pusta lista = wyłączone. Nie dodawaj osoby – stojący człowiek przestałby wywoływać alarm."),
        F("static_after", "Nieruchomy po", "float", 60.0,
          "Po tylu sekundach w tym samym miejscu obiekt jest uznawany za nieruchomy.", min=5, max=3600, unit="s"),
        F("static_forget", "Zapomnij po", "float", 21600.0,
          "Nieruchomy obiekt niewidziany przez ten czas jest zapominany (21600 s = 6 h).",
          min=60, max=604800, unit="s"),
    ]},
    {"id": "recording", "title": "Nagrywanie",
     "desc": "Zapis ciągły strumienia głównego na dysk (zamiast Shinobi), bez przekodowania. Pliki jak w "
             "Shinobi: <nagrania>/<klucz grupy>/<ID monitora>/RRRR-MM-DDTGG-MM-SS.mp4. Nagrywanie każdej "
             "kamery włączysz w zakładce Kamery.",
     "fields": [
        F("enabled", "Nagrywaj na dysk", "bool", False,
          "Ścieżka: Ścieżki → Nagrania Shinobi. Usługa musi mieć prawo zapisu w tym katalogu."),
        F("segment_minutes", "Długość pliku", "int", 15, "Pliki zaczynają się o pełnych minutach zegara.",
          min=1, max=60, unit="min"),
        F("retention_days", "Przechowuj przez", "float", 14.0, "Starsze nagrania są usuwane. 0 = bez limitu dni.",
          min=0, max=3650, unit="dni"),
        F("min_free_gb", "Minimalne wolne miejsce", "float", 50.0,
          "Gdy na dysku zostanie mniej, usuwane są najstarsze nagrania wszystkich kamer.",
          min=0, max=100000, unit="GB"),
        F("audio", "Nagrywaj dźwięk", "bool", False, "Dźwięk z kamery zapisywany jako AAC (lekkie kodowanie na CPU)."),
    ]},
    {"id": "homekit", "title": "HomeKit",
     "desc": "Kamery w aplikacji Dom na iPhonie, iPadzie i Apple TV (podgląd bez przekodowania) oraz czujniki "
             "ruchu z wykrywania (np. osoba, samochód) z powiadomieniami ze zdjęciem. Każdą kamerę i jej czujniki "
             "włączysz w zakładce Kamery. Zastępuje integrację HomeKit z Home Assistanta.",
     "fields": [
        F("enabled", "HomeKit", "bool", False, "Wymaga restartu usługi."),
        F("port", "Pierwszy port", "int", 51826,
          "Każda kamera to osobne akcesorium na kolejnym porcie (51826, 51827…). Port nie może być zajęty, "
          "np. przez Home Assistanta.", min=1024, max=65000, restart_note=True),
        F("address", "Adres serwera w sieci", "text", "auto",
          "Adres IP, pod którym iPhone i Apple TV widzą serwer. auto = adres z trasą domyślną.",
          restart_note=True),
        F("motion_seconds", "Czujnik ruchu aktywny przez", "float", 10.0,
          "Po ostatnim wykryciu czujnik wraca do „brak ruchu” po tylu sekundach.", min=2, max=600, unit="s"),
        F("_homekit", "Akcesoria", "homekit", None, ""),
    ]},
    {"id": "go2rtc", "title": "Strumienie (go2rtc)",
     "desc": "go2rtc łączy się z każdą kamerą raz na strumień i udostępnia obraz lokalnie analizie, nagrywaniu "
             "i innym programom. Bez niego każdy proces łączy się z kamerą osobno.",
     "fields": [
        F("enabled", "Używaj go2rtc", "bool", True, "Program jest w pakiecie (/opt/szukaj_cpu/go2rtc)."),
        F("rtsp_listen", "Restream RTSP dostępny z", "select", "127.0.0.1",
          "„tylko ten serwer” jest najbezpieczniejsze. „sieć lokalna” pozwala np. otworzyć "
          "rtsp://serwer:8554/brama w VLC albo w innym programie (bez hasła!).",
          options=[["127.0.0.1", "tylko ten serwer"], ["0.0.0.0", "sieć lokalna"]]),
        F("rtsp_port", "Port RTSP", "int", 8554, "", min=1, max=65535),
        F("api_port", "Port API (lokalny)", "int", 1984, "Dostępny tylko z tego serwera.", min=1, max=65535),
        F("binary", "Program go2rtc", "text", "auto", "„auto” = ~/.szukaj_cpu/go2rtc, potem pakiet, potem PATH."),
    ]},
    {"id": "clips", "title": "Klipy", "desc": "Krótkie nagrania zdarzeń wycinane ze strumienia głównego (bez kodowania).",
     "fields": [
        F("enabled", "Twórz klipy", "bool", True, "Wymagane także do pełnych klatek w galerii."),
        F("pre_seconds", "Przed zdarzeniem", "float", 3.0, "", min=0, max=60, unit="s"),
        F("post_seconds", "Po zdarzeniu", "float", 5.0, "", min=0, max=120, unit="s"),
        F("max_length", "Maksymalna długość klipu", "float", 300.0, "", min=10, max=3600, unit="s"),
        F("segment_length", "Długość segmentu bufora", "int", 4,
          "Dokładność początku i końca klipu. Bufor jest w RAM (/dev/shm).", min=2, max=30, unit="s"),
    ]},
    {"id": "gallery", "title": "Galeria i panel WWW", "desc": "Strona z wykryciami i ten panel.", "fields": [
        F("retention_days", "Przechowuj przez", "float", 5.0, "Starsze wpisy, miniatury i klipy są usuwane.",
          min=0.5, max=365, unit="dni"),
        F("refresh_seconds", "Odświeżanie galerii", "int", 60, "0 = wyłączone.", min=0, max=3600, unit="s"),
        F("http_port", "Port serwera WWW", "int", 0,
          "0 = wyłącza galerię WWW i ten panel. Zmiana portu = panel pod nowym adresem po restarcie.",
          min=0, max=65535, restart_note=True),
        F("http_listen", "Adres nasłuchu", "text", "0.0.0.0",
          "0.0.0.0 = wszystkie interfejsy (LAN i Tailscale). 127.0.0.1 = tylko lokalnie, np. za Tailscale Serve "
          "z HTTPS. Można podać kilka adresów po przecinku, np. 127.0.0.1, 100.101.102.103.", restart_note=True),
        F("http_user", "Login", "text", "", "Login i hasło włączają ten panel. Bez nich panel się wyłączy."),
        F("http_password", "Hasło", "secret", "", "Zostaw puste, aby nie zmieniać."),
        F("login_max_failures", "Blokada po nieudanych logowaniach", "int", 5,
          "Po tylu błędnych hasłach z jednego adresu logowanie jest czasowo zablokowane. 0 = bez blokady.",
          min=0, max=100),
        F("login_lockout_minutes", "Czas blokady", "float", 15.0, "", min=1, max=1440, unit="min"),
        F("full_frame", "Pełna klatka ze strumienia głównego", "bool", True,
          "Dla każdego zdarzenia zapisuje kadr w pełnej rozdzielczości (wymaga klipów)."),
        F("full_format", "Format pełnej klatki", "select", "jxl",
          "JPEG XL jest najmniejszy, ale pokazuje go tylko Safari.",
          options=[["jxl", "JPEG XL (najmniejszy)"], ["jpg", "JPEG (działa wszędzie)"], ["webp", "WebP"]]),
        F("full_quality", "Jakość pełnej klatki", "int", 90, "Dla JPEG XL 90 = wizualnie bezstratnie.",
          min=1, max=100, slider=True),
        F("thumb_width", "Szerokość miniatury", "int", 1280, "0 = pełny rozmiar.", min=0, max=3840, unit="px"),
    ]},
    {"id": "notifications", "title": "Powiadomienia", "desc": "Powiadomienia ze zdjęciem przez ntfy.", "fields": [
        F("ntfy_url", "Adres tematu ntfy", "text", "",
          "Np. http://127.0.0.1:8092/szukaj (własny serwer) lub https://ntfy.sh/temat. Puste = wyłączone.",
          placeholder="http://127.0.0.1:8092/szukaj"),
        F("ntfy_token", "Token ntfy", "secret", "", "Token użytkownika publikującego. Zostaw puste, aby nie zmieniać."),
        F("priority", "Priorytet", "select", "high", "",
          options=[["min", "min"], ["low", "niski"], ["default", "domyślny"], ["high", "wysoki"],
                   ["max", "maksymalny (alarm)"]]),
        F("click_url", "Adres po stuknięciu", "text", "", "Np. adres galerii.", placeholder="http://serwer:8091/"),
        F("min_interval", "Odstęp między powiadomieniami", "float", 120.0, "Z jednej kamery.",
          min=0, max=3600, unit="s"),
    ]},
    {"id": "shinobi", "title": "Shinobi", "desc": "Znaczniki zdarzeń na osi czasu Shinobi (API /motion). "
                                                   "Puste klucze = wyłączone.", "fields": [
        F("url", "Adres panelu Shinobi", "text", "", "Na tym samym serwerze zwykle http://127.0.0.1:8080",
          placeholder="http://127.0.0.1:8080"),
        F("api_key", "Klucz API", "secret", "", "Panel Shinobi → menu użytkownika → API. Puste = bez zmian."),
        F("group_key", "Klucz grupy", "secret", "", "Panel Shinobi → Ustawienia → Group Key. Puste = bez zmian."),
        F("repeat_seconds", "Ponawiaj co", "float", 30.0,
          "Dopóki obiekt jest widoczny (podtrzymuje nagrywanie Watch-Only). 0 = raz na zdarzenie.",
          min=0, max=3600, unit="s"),
    ]},
    {"id": "paths", "title": "Ścieżki i narzędzia", "desc": "Katalogi, model i programy.", "fields": [
        F("recordings", "Nagrania Shinobi", "text", "",
          "Gdy galeria jest pusta, powstaje obok jako wyniki_osoby.", placeholder="/home/Shinobi/videos"),
        F("gallery", "Katalog galerii", "text", "", "Użytkownik usługi musi mieć prawo zapisu."),
        F("buffer", "Bufor segmentów", "text", "/dev/shm/szukaj_cpu", "W RAM – nie zużywa dysku."),
        F("model", "Model YOLO", "text", "~/.szukaj_cpu/models/yolo11s.pt",
          "Plik .pt; wersja OpenVINO powstaje obok przy eksporcie."),
        F("ffmpeg", "ffmpeg", "text", "auto", "„auto” = statyczny build z ~/.szukaj_cpu/, inaczej systemowy."),
        F("ffprobe", "ffprobe", "text", "auto", ""),
        F("vaapi_device", "Urządzenie VAAPI (iGPU)", "text", "/dev/dri/renderD128", "Puste = dekodowanie na CPU."),
        F("log_file", "Dodatkowy plik logu", "text", "", "Puste = tylko dziennik systemd."),
    ]},
]


SECRET_MASK = "********"

_URL_PW = re.compile(r"^([a-zA-Z][\w+.-]*://[^:/@\s]*:)([^/\s]*)(@)")   # hasło do ostatniego @


def form_fields():
    for grp in FORM:
        for f in grp["fields"]:
            if f["type"] not in ("models", "homekit"):  # widżety panelu, nie klucze pliku
                yield grp.get("section", grp["id"]), f


def mask_secret_url(url):
    return _URL_PW.sub(lambda m: m.group(1) + (SECRET_MASK if m.group(2) else "") + m.group(3), url)


def url_password(url):
    m = _URL_PW.match(url or "")
    return m.group(2) if m else None


def config_view(path):
    """Wartości do formularza: z pliku albo domyślne; sekrety bez treści."""
    text = Path(path).read_text(encoding="utf-8")
    data = tomllib.loads(text)
    vals, secrets = {}, {}
    for sec, f in form_fields():
        cur = data.get(sec, {}) if isinstance(data.get(sec), dict) else {}
        v = cur.get(f["key"], f["default"])
        if f["type"] == "secret":
            secrets[f"{sec}.{f['key']}"] = bool(v)
            v = ""
        vals.setdefault(sec, {})[f["key"]] = v
    cams = []
    rec_root = str((data.get("paths") or {}).get("recordings", "")).strip()
    group = str((data.get("shinobi") or {}).get("group_key", "")).strip()
    for i, c in enumerate(data.get("camera", []) if isinstance(data.get("camera"), list) else [], 1):
        name = re.sub(r"[^\w-]", "_", str(c.get("name", f"camera{i}")).strip()) or f"camera{i}"
        mon = str(c.get("shinobi_monitor", "")).strip()
        dflt = default_record_dir(expand(rec_root), group, mon, name)
        cams.append({"orig": name, "name": str(c.get("name", name)), "enabled": c.get("enabled", True) is True,
                     "record": c.get("record", True) is True, "record_dir": str(c.get("record_dir", "")).strip(),
                     "record_dir_default": str(dflt) if dflt else "",
                     "url": mask_secret_url(str(c.get("url", ""))),
                     "analysis_url": mask_secret_url(str(c.get("analysis_url", ""))),
                     "shinobi_monitor": str(c.get("shinobi_monitor", "")),
                     "homekit": c.get("homekit", False) is True,
                     "homekit_motion": [str(x) for x in c.get("homekit_motion", []) if isinstance(x, str)]
                     if isinstance(c.get("homekit_motion", []), list) else [],
                     "has_zone": bool(c.get("zone")), "has_crop": bool(c.get("crop"))})
    return {"form": FORM, "coco": COCO, "values": vals, "secrets": secrets, "cameras": cams,
            "version": config_version(text)}


def find_record_dirs(path):
    """Katalogi z nagraniami w stylu Shinobi (RRRR-MM-DDTGG-MM-SS.mp4) pod [paths] recordings – podpowiedzi
    do ustawienia katalogu nagrań kamery w panelu."""
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    root = expand(str((data.get("paths") or {}).get("recordings", "")).strip())
    if root is None or not root.is_dir():
        return {"root": str(root) if root else "", "dirs": []}
    out, seen = [], 0
    stack = [(root, 0)]
    while stack and seen < 2000:
        d, depth = stack.pop()
        seen += 1
        try:
            entries = list(os.scandir(d))
        except OSError:
            continue
        mp4 = [e for e in entries if e.is_file() and NAME_DATE.search(e.name) and e.name.endswith(".mp4")]
        if mp4:
            newest = max(mp4, key=lambda e: e.name)
            out.append({"path": str(d), "files": len(mp4), "newest": newest.name})
        if depth < 3:
            stack += [(Path(e.path), depth + 1) for e in entries
                      if e.is_dir(follow_symlinks=False) and not e.name.startswith(".")]
    return {"root": str(root), "dirs": sorted(out, key=lambda x: x["path"])}


def config_version(text):
    import hashlib
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _coerce(f, v):
    t, lab = f["type"], f["label"]
    if t == "bool":
        if not isinstance(v, bool):
            raise ValueError(f"{lab}: oczekiwano tak/nie")
        return v
    if t in ("int", "float"):
        try:
            x = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"{lab}: to nie jest liczba")
        if t == "int":
            if x != int(x):
                raise ValueError(f"{lab}: musi być liczbą całkowitą")
            x = int(x)
        if "min" in f and x < f["min"] or "max" in f and x > f["max"]:
            raise ValueError(f"{lab}: dozwolony zakres {f.get('min')}–{f.get('max')}")
        return x
    if t == "classes":
        if not isinstance(v, list):
            raise ValueError(f"{lab}: oczekiwano listy")
        out = []
        for x in v:
            x = str(x).strip()
            if x and x not in out:
                out.append(x)
        return out
    if t == "select":
        if v not in [o[0] for o in f["options"]]:
            raise ValueError(f"{lab}: niedozwolona wartość")
        return v
    if not isinstance(v, str):
        raise ValueError(f"{lab}: oczekiwano tekstu")
    if "\n" in v or "\r" in v:
        raise ValueError(f"{lab}: tekst nie może mieć nowych linii")
    return v if t == "secret" else v.strip()


def _restore_pw(new_url, old_urls):
    """Adres z hasłem zamaskowanym w formularzu → hasło z dotychczasowego adresu tej kamery."""
    if url_password(new_url) != SECRET_MASK:
        return new_url
    def who(u):          # użytkownik i host – hasło wraca tylko do tej samej kamery
        m = _URL_PW.match(u)
        return m.group(1), u[m.end():].split("/", 1)[0].lower()
    want = who(new_url)
    for old in old_urls:
        m = _URL_PW.match(old or "")
        if m and m.group(2) != SECRET_MASK and who(old) == want:
            return _URL_PW.sub(lambda mm: mm.group(1) + m.group(2) + mm.group(3), new_url)
    raise ValueError("podaj hasło w adresie kamery – zmieniono użytkownika lub adres IP, więc zapisane "
                     "hasło nie zostanie użyte")


def plan_config(path, body):
    """Zbuduj nowy tekst pliku z danych formularza. Zwraca (stary, nowy, lista zmian)."""
    old = Path(path).read_text(encoding="utf-8")
    data = tomllib.loads(old)
    vals = body.get("values") or {}
    sections, changes = {}, []
    for sec, f in form_fields():
        if f["key"] not in (vals.get(sec) or {}):
            continue
        new = _coerce(f, vals[sec][f["key"]])
        cur_tbl = data.get(sec, {}) if isinstance(data.get(sec), dict) else {}
        cur = cur_tbl.get(f["key"], f["default"])
        if f["type"] == "secret" and new == "" and not body.get("clear", {}).get(f"{sec}.{f['key']}"):
            continue                                   # puste = bez zmian
        if f["type"] == "float" and new == int(new):
            new = int(new)                             # 10 zamiast 10.0 – TOML i load_config przyjmą oba
        if cur == new and type(cur) is not bool or cur is new:
            continue
        sections.setdefault(sec, {})[f["key"]] = new
        if f["type"] == "secret":
            cur, new_show = ("•••" if cur else ""), ("(nowa wartość)" if new else "")
        else:
            new_show = new
        changes.append({"where": f"{sec}.{f['key']}", "label": f["label"], "old": cur, "new": new_show})

    old_cams = {}
    for i, c in enumerate(data.get("camera", []) if isinstance(data.get("camera"), list) else [], 1):
        old_cams[re.sub(r"[^\w-]", "_", str(c.get("name", f"camera{i}")).strip()) or f"camera{i}"] = c
    cams_in = body.get("cameras")
    cams = []
    if cams_in is None:
        cams = [{"orig": n} for n in old_cams]
    else:
        names = set()
        for c in cams_in:
            orig = c.get("orig") if c.get("orig") in old_cams else None
            name = str(c.get("name", "")).strip()
            if not name:
                raise ValueError("każda kamera musi mieć nazwę")
            safe = re.sub(r"[^\w-]", "_", name)
            if safe in names:
                raise ValueError(f"powtórzona nazwa kamery: {name}")
            names.add(safe)
            oc = old_cams.get(orig, {})
            olds = [str(oc.get("url", "")), str(oc.get("analysis_url", ""))]
            entry = {"orig": orig}
            for k in CAMERA_KEYS:
                v = c.get(k)
                if k in ("enabled", "record"):
                    v = v is not False
                elif k == "homekit":
                    v = v is True
                elif k == "homekit_motion":
                    v = [str(x) for x in (v or []) if str(x) in COCO]
                else:
                    v = str(v or "").strip()
                    if "\n" in v:
                        raise ValueError(f"{name}: nowa linia w polu {k}")
                if k in ("url", "analysis_url"):
                    if v and not re.match(r"^(rtsps?|https?|file)://|^/", v):
                        raise ValueError(f"{name}: adres musi zaczynać się od rtsp://")
                    v = _restore_pw(v, olds) if v else v
                if k == "record_dir" and v and not (v.startswith("/") or v.startswith("~")):
                    raise ValueError(f"{name}: katalog nagrań musi być pełną ścieżką (np. /srv/…)")
                if k == "url" and not v:
                    raise ValueError(f"{name}: brak adresu strumienia głównego")
                prev = oc.get(k, {"enabled": True, "record": True, "homekit": False, "homekit_motion": []}.get(k, "")) \
                    if orig else None
                if orig is None or prev != v:
                    entry[k] = v
                    if orig is not None:
                        show = mask_secret_url if k in ("url", "analysis_url") else (lambda x: x)
                        changes.append({"where": f"camera.{orig}.{k}", "label": f"{orig}: {k}",
                                        "old": show(prev) if isinstance(prev, str) else prev,
                                        "new": show(v) if isinstance(v, str) else v})
            cams.append(entry)
            if orig is None:
                changes.append({"where": f"camera.{name}", "label": f"nowa kamera {name}", "old": "",
                                "new": mask_secret_url(entry["url"])})
        for n in old_cams:
            if n not in {c["orig"] for c in cams}:
                changes.append({"where": f"camera.{n}", "label": f"usunięta kamera {n}", "old": n, "new": ""})
        if [c["orig"] for c in cams if c["orig"]] != [n for n in old_cams if n in {c["orig"] for c in cams}]:
            changes.append({"where": "camera", "label": "kolejność kamer", "old": "", "new": ""})
    new = toml_apply(old, sections, cams) if (sections or cams_in is not None) else old
    return old, new, changes


def validate_config_text(text, path):
    """Pełna kontrola jak przy starcie usługi (load_config) na kopii pliku."""
    tmp = Path(path).with_name(f".{Path(path).name}.check")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.chmod(tmp, 0o600)
        err = []
        real = sys.stderr
        import io
        sys.stderr = io.StringIO()
        try:
            load_config(tmp)
        except SystemExit as e:
            err.append(str(e.code).replace(str(tmp), "konfiguracja"))
        finally:
            warn = sys.stderr.getvalue()
            sys.stderr = real
        warns = [w.strip().lstrip("! ").replace(str(tmp), "konfiguracja") for w in warn.splitlines() if w.strip()]
        return (err[0] if err else None), warns
    finally:
        tmp.unlink(missing_ok=True)


def write_config(path, old, new):
    path = Path(path)
    mode = path.stat().st_mode & 0o777
    bak = path.with_name(path.name + ".bak")
    bak.write_text(old, encoding="utf-8")
    os.chmod(bak, mode)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(new, encoding="utf-8")
    os.chmod(tmp, mode)
    tmp.replace(path)


def probe_stream(cfg, url, cam_orig=None):
    """Test strumienia z panelu: kodek, rozdzielczość, kl/s i odstęp klatek kluczowych."""
    if not re.match(r"^rtsps?://", url or ""):
        raise ValueError("test działa tylko dla adresów rtsp://")
    if url_password(url) == SECRET_MASK:
        data = tomllib.loads(Path(cfg.path).read_text(encoding="utf-8"))
        olds = [str(c.get(k, "")) for c in data.get("camera", []) for k in ("url", "analysis_url")
                if cam_orig is None or re.sub(r"[^\w-]", "_", str(c.get("name", "")).strip()) == cam_orig]
        url = _restore_pw(url, olds)
    cmd = [cfg.ffprobe, "-v", "error", "-rtsp_transport", "tcp", "-timeout", "8000000",
           "-select_streams", "v:0", "-read_intervals", "%+6",
           "-show_entries", "stream=codec_name,width,height,avg_frame_rate,r_frame_rate:packet=pts_time,flags",
           "-of", "json", url]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
    except subprocess.TimeoutExpired:
        raise ValueError("brak odpowiedzi kamery w 25 s")
    try:
        j = json.loads(r.stdout or "{}")
        st = j["streams"][0]
    except (ValueError, KeyError, IndexError):
        msg = (r.stderr or "").strip().splitlines()
        raise ValueError(mask_url(msg[-1]) if msg else "nie udało się odczytać strumienia")

    def rate(s):
        try:
            a, b = s.split("/")
            return float(a) / float(b) if float(b) else 0.0
        except (ValueError, AttributeError):
            return 0.0
    fps = rate(st.get("avg_frame_rate")) or rate(st.get("r_frame_rate"))
    keys = sorted(float(p["pts_time"]) for p in j.get("packets", [])
                  if "K" in p.get("flags", "") and p.get("pts_time") not in (None, "N/A"))
    gop = None
    if len(keys) >= 2:
        d = sorted(b - a for a, b in zip(keys, keys[1:]))
        gop = round(d[len(d) // 2], 2)
    return {"codec": st.get("codec_name", "?"), "width": st.get("width"), "height": st.get("height"),
            "fps": round(fps, 2) if fps else None, "gop": gop, "keyframes": len(keys),
            "ms": int((time.time() - t0) * 1000)}
