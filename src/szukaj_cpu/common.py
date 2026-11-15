"""Wspólne stałe, logger, flaga zatrzymania i drobne narzędzia."""

import logging
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path


log = logging.getLogger("szukaj_cpu")

STOP = threading.Event()

CSV_FIELDS = ["kamera", "poczatek", "koniec", "obiekty", "trafien", "max_obiektow", "pelna",
              "max_pewnosc", "miniatura", "klip"]

NAME_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})")   # jak w Shinobi

SEG_RE = re.compile(r"^(\d{8}-\d{6})\.ts$")

TS_FMT = "%Y-%m-%d %H:%M:%S"

FILE_TS = "%Y-%m-%dT%H-%M-%S"

CONF_DIR = Path(os.path.expanduser("~/.szukaj_cpu"))

DEFAULT_CONF = "~/.szukaj_cpu/szukaj_cpu.toml"


def expand(v):
    v = (v or "").strip()
    return Path(os.path.expanduser(v)) if v else None


# katalog programu: /opt/szukaj_cpu (pakiet) albo src/ (repozytorium)
APP_DIR = Path(__file__).resolve().parent.parent


def mask_url(url):
    return re.sub(r"//[^@/]*@", "//***@", url)


COCO80 = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
          "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
          "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
          "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
          "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
          "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
          "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
          "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
          "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
          "teddy bear", "hair drier", "toothbrush"]


def list_segments(d):
    segs = []
    try:
        for f in d.iterdir():
            m = SEG_RE.match(f.name)
            if m:
                try:
                    segs.append((datetime.strptime(m.group(1), "%Y%m%d-%H%M%S").timestamp(), f))
                except ValueError:
                    pass
    except FileNotFoundError:
        pass
    return sorted(segs)


CONFIG_LOCK = threading.Lock()

START_TIME = time.time()
