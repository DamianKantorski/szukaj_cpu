"""Model YOLO w OpenVINO (FP32/INT8) lub PyTorch, filtr strefy i przycięcia, rysowanie ramek."""

import os
import re
import sys
import threading
import time

import cv2
import numpy as np

from .common import APP_DIR, COCO80, log
from .export import model_dir_name, parse_model_dir_name

BUNDLED_MODELS = APP_DIR / "models"          # modele dołączone do pakietu .deb


def model_search_dirs(cfg):
    """Kolejność szukania: katalog modelu z konfiguracji (własne, ~/.szukaj_cpu/models), potem pakiet."""
    out = [cfg.model.parent]
    if BUNDLED_MODELS.resolve() != cfg.model.parent.resolve():
        out.append(BUNDLED_MODELS)
    return out


def list_models(cfg):
    """Wszystkie modele OpenVINO w katalogach wyszukiwania (do panelu WWW)."""
    seen, res = set(), []
    for src, d in zip(("user", "bundled"), model_search_dirs(cfg)):
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            info = parse_model_dir_name(p.name) if p.is_dir() else None
            if not info or not any(p.glob("*.xml")):
                continue
            stem, w, h, int8 = info
            res.append({"name": p.name, "stem": stem, "w": w, "h": h, "precision": "int8" if int8 else "fp32",
                        "source": src, "path": str(p), "shadowed": p.name in seen,
                        "mb": round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6, 1),
                        "mtime": int(max(f.stat().st_mtime for f in p.rglob("*") if f.is_file()))})
            seen.add(p.name)
    return res


def find_model(cfg):
    """Katalog modelu OpenVINO dla stem/wejścia/precyzji z konfiguracji.
    -> (ścieżka albo None, użyta precyzja, uwaga do logu/panelu)"""
    h, w = cfg.input_hw
    stem = cfg.model.stem
    order = ["int8", "fp32"] if cfg.precision == "int8" else ["fp32"]
    for prec in order:
        name = model_dir_name(stem, w, h, prec == "int8")
        for d in model_search_dirs(cfg):
            if (d / name).is_dir() and any((d / name).glob("*.xml")):
                note = ""
                if prec != cfg.precision:
                    note = (f"brak modelu INT8 {model_dir_name(stem, w, h, True)} – używam FP32. "
                            f"Dodaj model INT8 (pakiet .deb, panel → Model albo tools/export_models.py).")
                return d / name, prec, note
    return None, cfg.precision, ""


def read_names(model_dir):
    """Nazwy klas z metadata.yaml zapisanego przez eksport Ultralytics (bez zależności od PyYAML)."""
    names, inside = {}, False
    try:
        for line in (model_dir / "metadata.yaml").read_text(encoding="utf-8").splitlines():
            if line.startswith("names:"):
                inside = True
                continue
            if inside:
                m = re.match(r"\s+(\d+):\s*(.+)$", line)
                if m:
                    names[int(m.group(1))] = m.group(2).strip().strip("'\"")
                elif line.strip() and not line.startswith(" "):
                    break
    except OSError:
        pass
    return names or dict(enumerate(COCO80))


class OpenVinoYolo:
    """YOLO (eksport Ultralytics) uruchamiany bezpośrednio w OpenVINO – bez PyTorch i Ultralytics.
    Przygotowanie obrazu (letterbox) i NMS przez OpenCV."""

    def __init__(self, model_dir, threads, iou=0.7, max_det=100, precision_hint=None):
        import openvino as ov
        xml = next(model_dir.glob("*.xml"))
        core = ov.Core()
        props = {"PERFORMANCE_HINT": "LATENCY"}
        if threads > 0:
            props["INFERENCE_NUM_THREADS"] = int(threads)
        if precision_hint:        # np. "f32" – nowsze CPU liczą FP32 w bf16; porównania jak na starszym CPU
            props["INFERENCE_PRECISION_HINT"] = precision_hint
        self.compiled = core.compile_model(core.read_model(xml), "CPU", props)
        self.req = self.compiled.create_infer_request()
        self.out = self.compiled.output(0)
        _, _, self.in_h, self.in_w = [int(d) for d in self.compiled.input(0).shape]
        self.names = read_names(model_dir)
        self.iou, self.max_det = iou, max_det

    def __call__(self, frame, classes, conf):
        h, w = frame.shape[:2]
        r = min(self.in_w / w, self.in_h / h)
        nw, nh = int(round(w * r)), int(round(h * r))
        img = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else frame
        left = (self.in_w - nw) // 2
        top = (self.in_h - nh) // 2
        img = cv2.copyMakeBorder(img, top, self.in_h - nh - top, left, self.in_w - nw - left,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
        blob = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))[None].astype(np.float32) * (1 / 255.0)

        pred = self.req.infer({0: blob})[self.out][0].T          # (N, 4 + liczba klas)
        scores = pred[:, 4:][:, classes]
        k = scores.argmax(1)
        best = scores[np.arange(len(scores)), k]
        keep = best >= conf
        if not keep.any():
            return []
        boxes, best, k = pred[keep, :4], best[keep], k[keep]
        # cx,cy,w,h (wejście modelu) -> x,y,w,h (klatka)
        bw, bh = boxes[:, 2] / r, boxes[:, 3] / r
        bx = (boxes[:, 0] - left) / r - bw / 2
        by = (boxes[:, 1] - top) / r - bh / 2
        rects = np.stack([bx, by, bw, bh], 1)
        cls_ids = np.asarray(classes)[k]
        idx = cv2.dnn.NMSBoxesBatched(rects.tolist(), best.tolist(), cls_ids.tolist(), conf, self.iou)
        res = []
        for i in np.array(idx).reshape(-1)[: self.max_det]:
            x, y, ww, hh = rects[i]
            res.append((int(max(0, x)), int(max(0, y)), int(min(w - 1, x + ww)), int(min(h - 1, y + hh)),
                        float(best[i]), self.names.get(int(cls_ids[i]), str(cls_ids[i]))))
        return res


class Detector:
    def __init__(self, cfg):
        self.cfg = cfg
        pt = cfg.model
        pt.parent.mkdir(parents=True, exist_ok=True)
        os.chdir(pt.parent)   # tu Ultralytics pobierze model, jeśli go brak

        self.model_dir, self.precision, self.note = None, None, ""
        if cfg.silnik == "openvino":
            h, w = cfg.input_hw
            ov, self.precision, self.note = find_model(cfg)
            if ov is None:
                # Ultralytics/PyTorch (+ NNCF dla INT8) potrzebne tylko do jednorazowego eksportu
                int8 = cfg.precision == "int8"
                log.info("Brak modelu %s – eksport %s do OpenVINO (wejście %dx%d%s), jednorazowo, potrwa chwilę…",
                         model_dir_name(pt.stem, w, h, int8), pt.name, w, h, ", INT8" if int8 else "")
                import importlib.util
                if importlib.util.find_spec("ultralytics") is None:
                    sys.exit(f"Brak modelu {model_dir_name(pt.stem, w, h, int8)} (szukano w: "
                             f"{', '.join(str(d) for d in model_search_dirs(cfg))}) i brak narzędzi do eksportu.\n"
                             f"Wybierz rozmiar wejścia dołączony do pakietu (panel → Model) albo zainstaluj "
                             f"narzędzia jednorazowo:  sudo szukaj-cpu-export-tools install")
                from .export import export_one
                if not pt.is_file():
                    from ultralytics import YOLO
                    YOLO(pt.name)                              # pobranie wag do katalogu modeli
                ov = export_one(pt, w, h, int8, pt.parent)
            elif self.note:
                log.warning(self.note)
            self.model_dir = ov
            self.ov = OpenVinoYolo(ov, cfg.watki)
            self.names = self.ov.names
            src = f"{ov.name} ({ov.parent}), wątki: {cfg.watki or 'auto'}"
            wejscie = f"{self.ov.in_w}x{self.ov.in_h}"
        else:
            import torch
            torch.set_num_threads(max(1, cfg.watki))
            from ultralytics import YOLO
            self.ov = None
            self.model = YOLO(pt.name)
            self.imgsz = cfg.imgsz
            dummy = np.zeros((720, 1280, 3), np.uint8)
            self.model.predict(dummy, imgsz=self.imgsz, device="cpu", verbose=False)
            self.names = {int(k): v for k, v in self.model.names.items()}
            src, wejscie = pt.name, str(cfg.imgsz)

        wanted = {n.lower() for n in cfg.klasy}
        self.cls = [i for i, n in self.names.items() if n.lower() in wanted]
        unknown = wanted - {n.lower() for n in self.names.values()}
        if unknown:
            log.warning("Nieznane klasy (pomijam): %s", ", ".join(sorted(unknown)))
        if not self.cls:
            sys.exit("Żadna z klas z konfiguracji nie występuje w modelu.")

        dummy = np.zeros((576, 1024, 3), np.uint8)
        self._run(dummy)                                     # rozgrzewka
        t0 = time.perf_counter()
        for _ in range(5):
            self._run(dummy)
        ms = (time.perf_counter() - t0) / 5 * 1000
        log.info("Model %s (%s), klasy: %s, wejście %s – ok. %.0f ms na klatkę",
                 src, cfg.silnik, ", ".join(self.names[i] for i in self.cls), wejscie, ms)
        self.ms_total, self.n = 0.0, 0
        self.lock = threading.Lock()

    def _run(self, frame):
        lock = getattr(self, "lock", None)
        if lock is not None:
            with lock:
                return self._run_unlocked(frame)
        return self._run_unlocked(frame)

    def _run_unlocked(self, frame):
        if self.ov is not None:
            return self.ov(frame, self.cls, self.cfg.conf)
        r = self.model.predict(frame, classes=self.cls, conf=self.cfg.conf, imgsz=self.imgsz,
                               device="cpu", verbose=False)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return []
        xyxy = b.xyxy.cpu().numpy()
        conf = b.conf.cpu().numpy()
        cls = b.cls.cpu().numpy().astype(int)
        return [(int(x1), int(y1), int(x2), int(y2), float(c), self.names[int(k)])
                for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls)]

    def predict(self, frame):
        t0 = time.perf_counter()
        res = self._run(frame)
        self.ms_total += (time.perf_counter() - t0) * 1000
        self.n += 1
        return res


def in_zone(det, zone, w, h):
    if zone is None:
        return True
    x1, y1, x2, y2 = det[:4]
    pt = ((x1 + x2) / 2 / w, y2 / h)     # środek dolnej krawędzi = "stopy"
    return cv2.pointPolygonTest(zone, pt, False) >= 0


def detect(det, cam, frame):
    """Analiza z przycięciem obrazu (crop) i filtrem strefy (zone); ramki we współrzędnych klatki."""
    h, w = frame.shape[:2]
    crop = cam.crop
    if crop is not None:
        x1, y1 = int(crop[0] * w), int(crop[1] * h)
        x2, y2 = max(x1 + 2, int(round(crop[2] * w))), max(y1 + 2, int(round(crop[3] * h)))
        part = np.ascontiguousarray(frame[y1:y2, x1:x2])
        dets = [(a + x1, b + y1, c + x1, d + y1, p, lbl) for (a, b, c, d, p, lbl) in det.predict(part)]
    else:
        dets = det.predict(frame)
    return [d for d in dets if in_zone(d, cam.zone, w, h)]


def draw(frame, dets, zone=None, crop=None):
    img = frame.copy()
    h, w = img.shape[:2]
    if crop is not None:                       # obszar analizy – przerywana ramka
        x1, y1, x2, y2 = int(crop[0] * w), int(crop[1] * h), int(crop[2] * w) - 1, int(crop[3] * h) - 1
        for a, b in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
            n = max(1, int(np.hypot(b[0] - a[0], b[1] - a[1]) // 12))
            for i in range(0, n, 2):
                p = (int(a[0] + (b[0] - a[0]) * i / n), int(a[1] + (b[1] - a[1]) * i / n))
                q = (int(a[0] + (b[0] - a[0]) * (i + 1) / n), int(a[1] + (b[1] - a[1]) * (i + 1) / n))
                cv2.line(img, p, q, (60, 140, 255), 2)
    if zone is not None:
        pts = (zone * np.array([w, h], np.float32)).astype(np.int32)
        cv2.polylines(img, [pts], True, (255, 180, 0), 2)
    for (x1, y1, x2, y2, c, label) in dets:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(img, f"{label} {c:.2f}", (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
    return img
