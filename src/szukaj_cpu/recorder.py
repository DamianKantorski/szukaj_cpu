"""Zapis ciągły: domykanie plików i usuwanie starych nagrań.

Proces nagrywania kamery (camera.build_rec_cmd) pisze fragmentowane MP4 do <katalog>/.nagrywanie –
taki plik da się odtworzyć nawet po przerwaniu zapisu. Po zamknięciu segmentu przepisujemy go
(bez kodowania, z niskim priorytetem) do zwykłego MP4 z indeksem na początku w <katalog>/,
bo takie pliki przeglądarki odtwarzają i przewijają najlepiej. Nazwy jak w Shinobi:
2026-11-19T17-30-00.mp4 (czas rozpoczęcia).
"""

import os
import shutil
import subprocess
import time
from datetime import datetime, timedelta

from .common import NAME_DATE, STOP, log


def low_prio():
    pre = ["nice", "-n", "19"] if shutil.which("nice") else []
    if shutil.which("ionice"):
        pre = ["ionice", "-c", "3"] + pre
    return pre


def file_start(p):
    m = NAME_DATE.search(p.name)
    if not m:
        return None
    try:
        return datetime.strptime(f"{m.group(1)}T{m.group(2)}-{m.group(3)}-{m.group(4)}", "%Y-%m-%dT%H-%M-%S")
    except ValueError:
        return None


def check_writable(d):
    """Czy usługa może pisać w katalogu nagrań (katalogi Shinobi bywają własnością innego użytkownika)."""
    try:
        d.mkdir(parents=True, exist_ok=True)
        t = d / f".zapis-test-{os.getpid()}"
        t.write_bytes(b"")
        t.unlink()
        return None
    except OSError as e:
        return str(e)


def has_encoder(ffmpeg, name):
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return any(len(f := ln.split()) > 1 and f[1] == name for ln in out.splitlines())


class Recorder:
    def __init__(self, cfg, cams):
        self.cfg = cfg
        self.cams = [c for c in cams if c.record]
        self.pre = low_prio()
        self.last_retention = 0.0

    def check(self):
        """Przed startem: katalogi zapisywalne? Kamery bez dostępu nie nagrywają (z jasnym komunikatem)."""
        if self.cams and self.cfg.rec_audio and not has_encoder(self.cfg.ffmpeg, "aac"):
            log.warning("nagrywanie: %s nie ma kodera AAC – zapis bez dźwięku "
                        "(zaktualizuj ffmpeg albo usuń stary z ~/.szukaj_cpu/)", self.cfg.ffmpeg)
            self.cfg.rec_audio = False
        for cam in self.cams:
            err = check_writable(cam.staging)
            if err:
                log.error("[%s] brak zapisu w %s (%s) – nagrywanie wyłączone. Nadaj uprawnienia, np.:  "
                          "sudo chown -R %s: %s", cam.name, cam.record_dir, err,
                          os.environ.get("USER", "$USER"), cam.record_dir)
                cam.record = False
            else:
                log.info("[%s] nagrywanie: %s (pliki %d min, %s, przechowywanie %g dni, min. %g GB wolnego)",
                         cam.name, cam.record_dir, self.cfg.rec_min, "z dźwiękiem" if self.cfg.rec_audio else "bez dźwięku",
                         self.cfg.rec_days, self.cfg.rec_free)
        self.cams = [c for c in self.cams if c.record]

    def loop(self):
        while not STOP.is_set():
            for cam in self.cams:
                try:
                    self.finalize(cam)
                except Exception:
                    log.exception("[%s] błąd domykania nagrań", cam.name)
            if time.time() - self.last_retention > 600:
                try:
                    self.retention()
                except Exception:
                    log.exception("Błąd usuwania starych nagrań")
                self.last_retention = time.time()
            STOP.wait(15)

    def finalize(self, cam, all_files=False):
        files = sorted(cam.staging.glob("*.mp4"))
        running = cam.rec_proc is not None and cam.rec_proc.poll() is None
        if running and not all_files:
            files = files[:-1]                       # najnowszy jest właśnie zapisywany
        for f in files:
            try:
                if time.time() - f.stat().st_mtime < 5 and not all_files:
                    continue
                size = f.stat().st_size
            except OSError:
                continue
            dst = cam.record_dir / f.name
            if size < 64 * 1024:                     # pusty / urwany po starcie – bezużyteczny
                f.unlink(missing_ok=True)
                continue
            tmp = cam.record_dir / f".{f.name}.tmp"
            r = subprocess.run(self.pre + [self.cfg.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                                           "-i", str(f), "-map", "0", "-c", "copy", "-movflags", "+faststart",
                                           "-f", "mp4", str(tmp)], capture_output=True, text=True)
            if r.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 0:
                tmp.replace(dst)
                f.unlink(missing_ok=True)
            else:
                tmp.unlink(missing_ok=True)
                log.warning("[%s] nie udało się przepisać %s (%s) – zostaje w wersji fragmentowanej",
                            cam.name, f.name, (r.stderr or "").strip().splitlines()[-1:] or "?")
                shutil.move(str(f), dst)

    def recordings(self):
        out = []
        for cam in self.cams:
            for p in cam.record_dir.glob("*.mp4"):
                t = file_start(p)
                if t:
                    out.append((t, cam, p))
        return sorted(out, key=lambda x: x[0])

    def retention(self):
        removed, freed = 0, 0
        limit = datetime.now() - timedelta(days=self.cfg.rec_days)
        files = self.recordings()
        keep = []
        for t, cam, p in files:
            if self.cfg.rec_days > 0 and t < limit:
                try:
                    freed += p.stat().st_size
                    p.unlink()
                    removed += 1
                except OSError:
                    pass
            else:
                keep.append((t, cam, p))
        # mało miejsca: usuwaj najstarsze (wszystkich kamer), zostaw zawsze ostatnią godzinę
        need = self.cfg.rec_free * 1e9
        for t, cam, p in keep:
            if self.free(cam.record_dir) >= need or t > datetime.now() - timedelta(hours=1):
                break
            try:
                freed += p.stat().st_size
                p.unlink()
                removed += 1
            except OSError:
                pass
        if removed:
            log.info("Nagrania: usunięto %d starych plików (%.1f GB); wolne: %.0f GB", removed, freed / 1e9,
                     self.free(self.cams[0].record_dir) / 1e9 if self.cams else 0)

    @staticmethod
    def free(d):
        try:
            st = os.statvfs(d)
            return st.f_bavail * st.f_frsize
        except OSError:
            return float("inf")

    def status(self):
        res = {}
        for cam in self.cams:
            cur = sorted(cam.staging.glob("*.mp4"))
            res[cam.name] = {"dir": str(cam.record_dir), "current": cur[-1].name if cur else None,
                             "running": cam.rec_proc is not None and cam.rec_proc.poll() is None,
                             "free_gb": round(self.free(cam.record_dir) / 1e9, 1)}
        return res

    def shutdown(self):
        """Po zatrzymaniu procesów nagrywania: domknij wszystkie pliki z .nagrywanie."""
        for cam in self.cams:
            try:
                self.finalize(cam, all_files=True)
            except Exception:
                log.exception("[%s] błąd domykania nagrań", cam.name)

