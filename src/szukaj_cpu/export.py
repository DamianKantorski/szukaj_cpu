"""Eksport YOLO -> OpenVINO (FP32 i INT8 przez NNCF). Wymaga Ultralytics + PyTorch + NNCF,
które nie są potrzebne do pracy usługi (komputer budujący albo szukaj-cpu-export-tools)."""

import re
import shutil
import sys
import tempfile
from pathlib import Path

MODEL_RE = re.compile(r"^(?P<stem>.+?)_(?P<w>\d+)(?:x(?P<h>\d+))?(?P<int8>_int8)?_openvino_model$")


def model_dir_name(stem, w, h, int8):
    size = f"{w}" if w == h else f"{w}x{h}"
    return f"{stem}_{size}{'_int8' if int8 else ''}_openvino_model"


def calib_yaml(images, names, workdir):
    """Zbiór kalibracyjny z katalogu obrazów bez etykiet (do INT8 etykiety nie są potrzebne)."""
    imgs = [p for p in Path(images).expanduser().rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if len(imgs) < 20:
        sys.exit(f"Za mało obrazów do kalibracji w {images} ({len(imgs)}, potrzeba ≥ 20, najlepiej 100–300)")
    d = Path(workdir) / "calib" / "images"
    d.mkdir(parents=True)
    for i, p in enumerate(sorted(imgs)[:1000]):        # kopie: Ultralytics zapisuje obok pliki .cache
        shutil.copy2(p, d / f"{i:05d}{p.suffix.lower()}")
    y = Path(workdir) / "calib.yaml"
    lines = [f"path: {d.parent}", "train: images", "val: images", "names:"]
    lines += [f"  {k}: {v}" for k, v in sorted(names.items())]
    y.write_text("\n".join(lines) + "\n")
    print(f"   kalibracja INT8: {len(imgs)} obrazów z {images}")
    return str(y)


def export_one(pt, w, h, int8, out_dir, calib=None, force=False):
    from ultralytics import YOLO
    dst = Path(out_dir) / model_dir_name(Path(pt).stem, w, h, int8)
    if dst.exists() and not force:
        print(f"== {dst.name}: już jest – pomijam (--force, aby zbudować ponownie)")
        return dst
    print(f"== {dst.name}: eksport {'INT8 (NNCF)' if int8 else 'FP32'}, wejście {w}x{h}", flush=True)
    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / Path(pt).name
        shutil.copy2(pt, local)
        model = YOLO(str(local))
        kw = {"format": "openvino", "imgsz": [h, w]}
        try:                                       # nowsze Ultralytics: quantize zamiast half/int8
            from ultralytics.cfg import DEFAULT_CFG_DICT
            new_api = "quantize" in DEFAULT_CFG_DICT
        except ImportError:
            new_api = False
        head = model.model.model[-1] if hasattr(model.model, "model") else None
        if getattr(head, "one2one_cv2", None) is not None:
            # YOLO26: głowica bez NMS (end-to-end) – gotowe ramki [x1, y1, x2, y2, pewność, klasa]
            kw["nms"] = False
        if int8:
            kw.update({"quantize": 8} if new_api else {"int8": True})
            kw["data"] = calib_yaml(calib, model.names, td) if calib else "coco128.yaml"
        out = Path(model.export(**kw))
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.move(str(out), tmp)
        shutil.rmtree(dst, ignore_errors=True)
        tmp.rename(dst)
    mb = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1e6
    print(f"   gotowe: {dst} ({mb:.0f} MB)", flush=True)
    return dst


def parse_model_dir_name(name):
    """yolo11s_960x544_int8_openvino_model -> ("yolo11s", 960, 544, True); None gdy nie pasuje."""
    m = MODEL_RE.match(name)
    if not m:
        return None
    w = int(m.group("w"))
    return m.group("stem"), w, int(m.group("h") or w), bool(m.group("int8"))
