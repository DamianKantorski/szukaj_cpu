"""Wycinanie klipów i pełnych klatek ze strumienia głównego (bufor segmentów w RAM)."""

import subprocess
import threading
import time

import cv2
import numpy as np

from .camera import probe, unpin
from .common import STOP, list_segments, log
from .detector import detect, draw
from .gallery import save_image


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
