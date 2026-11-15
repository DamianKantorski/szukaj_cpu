"""Modele w panelu WWW: lista, wgrywanie paczki .zip z katalogiem modelu OpenVINO, usuwanie."""

import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from ..common import log
from ..detector import list_models
from ..export import parse_model_dir_name

UPLOAD_MAX = 500 * 1024 * 1024


def models_view(cfg, det):
    active = str(det.model_dir) if det is not None and det.model_dir else None
    items = list_models(cfg)
    for m in items:
        m["active"] = m["path"] == active
        del m["path"]
    return {"models": items, "user_dir": str(cfg.model.parent), "stem": cfg.model.stem,
            "running": {"precision": getattr(det, "precision", None), "note": getattr(det, "note", ""),
                        "name": Path(active).name if active else None},
            "upload_max_mb": UPLOAD_MAX // (1024 * 1024)}


def install_zip(cfg, stream, length):
    """Rozpakuj .zip z jednym katalogiem *_openvino_model do katalogu modeli użytkownika."""
    if length <= 0 or length > UPLOAD_MAX:
        raise ValueError(f"plik musi mieć od 1 B do {UPLOAD_MAX // (1024 * 1024)} MB")
    dest_root = cfg.model.parent
    dest_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest_root, prefix=".upload-") as td:
        zpath = Path(td) / "model.zip"
        left = length
        with open(zpath, "wb") as f:
            while left > 0:
                chunk = stream.read(min(left, 1 << 20))
                if not chunk:
                    raise ValueError("przerwane przesyłanie pliku")
                f.write(chunk)
                left -= len(chunk)
        try:
            zf = zipfile.ZipFile(zpath)
        except zipfile.BadZipFile:
            raise ValueError("to nie jest plik .zip")
        with zf:
            names = [i for i in zf.infolist() if not i.is_dir()]
            total = sum(i.file_size for i in names)
            if total > 4 * UPLOAD_MAX:
                raise ValueError("archiwum po rozpakowaniu jest za duże")
            tops = set()
            for i in names:
                pp = PurePosixPath(i.filename)
                if pp.is_absolute() or ".." in pp.parts or not pp.parts:
                    raise ValueError(f"niedozwolona ścieżka w archiwum: {i.filename}")
                tops.add(pp.parts[0] if len(pp.parts) > 1 else "")
            if len(tops) != 1 or "" in tops:
                raise ValueError("archiwum musi zawierać dokładnie jeden katalog modelu, "
                                 "np. yolo11s_960x544_int8_openvino_model/ (tak tworzy go tools/export_models.py --zip)")
            top = tops.pop()
            if not parse_model_dir_name(top):
                raise ValueError(f"nazwa katalogu „{top}” nie pasuje do wzoru NAZWA_SZERxWYS[_int8]_openvino_model")
            files = {PurePosixPath(i.filename).name for i in names}
            if not any(n.endswith(".xml") for n in files) or not any(n.endswith(".bin") for n in files):
                raise ValueError("w katalogu modelu brak pliku .xml albo .bin (eksport OpenVINO)")
            out = Path(td) / "x"
            zf.extractall(out)
        src = out / top
        dst = dest_root / top
        old = dest_root / f".{top}.old"
        shutil.rmtree(old, ignore_errors=True)
        if dst.exists():
            dst.rename(old)
        shutil.move(str(src), dst)
        shutil.rmtree(old, ignore_errors=True)
    log.info("Wgrano model z panelu WWW: %s", dst)
    return top


def delete_model(cfg, det, name):
    if not parse_model_dir_name(name or "") or "/" in name:
        raise ValueError("nieprawidłowa nazwa modelu")
    p = cfg.model.parent / name
    if not p.is_dir():
        raise ValueError("można usuwać tylko własne modele (z katalogu użytkownika), nie modele z pakietu")
    if det is not None and det.model_dir and Path(det.model_dir).resolve() == p.resolve():
        raise ValueError("ten model jest teraz używany – najpierw wybierz inny i zrestartuj usługę")
    shutil.rmtree(p)
    log.info("Usunięto model z panelu WWW: %s", p)
