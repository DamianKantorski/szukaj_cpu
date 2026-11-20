"""Galeria zdarzeń: index.html, wyniki.csv, miniatury, sprzątanie starych wpisów."""

import csv
import html
import os
import subprocess
import sys
import threading
import urllib.parse
from datetime import datetime
from datetime import timedelta

import cv2
import numpy as np

from .common import CSV_FIELDS, NAME_DATE, TS_FMT, log


def _sod(ts):
    """Sekunda doby z 'RRRR-MM-DD GG:MM:SS'."""
    try:
        h, m, sec = (int(x) for x in ts[11:19].split(":"))
        return h * 3600 + m * 60 + sec
    except ValueError:
        return 0


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
        admin = bool(self.cfg.port and self.cfg.http_user and self.cfg.http_password)
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
                + (f' · <a href="/admin/nagrania#date={r["poczatek"][:10]}&amp;cam={urllib.parse.quote(r["kamera"])}'
                   f'&amp;t={max(0, _sod(r["poczatek"]) - 5)}">nagranie</a>' if admin else "")
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
{'<p><a href="/admin/nagrania">▶ Nagrania</a> · <a href="/admin/">⚙ Obszar detekcji</a> · <a href="/admin/config">⚙ Konfiguracja</a></p>' if self.cfg.port and self.cfg.http_user and self.cfg.http_password else ''}
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
