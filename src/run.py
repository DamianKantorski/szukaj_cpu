#!/usr/bin/env python3
"""Uruchomienie szukaj_cpu bez instalowania pakietu (python3 run.py …); w /opt/szukaj_cpu tak samo."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from szukaj_cpu.main import main  # noqa: E402

main()
