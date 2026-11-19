#!/usr/bin/env bash
# Buduje lokalnie pełny pakiet do wydania (kontener Debiana 13, Docker albo Podman):
#   dist/szukaj-cpu_<wersja>_amd64.deb  (+ .sha256)
# Pamięć podręczna (ffmpeg, modele, koła Pythona, narzędzia eksportu) zostaje w ./build,
# więc kolejne budowania trwają ok. minuty. Opcje jak packaging/build-deb.sh, np.:
#   ./build-local.sh --calib ~/klatki       INT8 kalibrowany na klatkach z kamer
#   ./build-local.sh --rebuild-ffmpeg       świeży ffmpeg
# Potem:  ./release.sh v<wersja>
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p dist
rm -f dist/szukaj-cpu_*.deb dist/szukaj-cpu_*.deb.sha256      # dist/ = tylko bieżąca wersja
bash packaging/build-deb.sh "$@"
echo
ls -lh dist/
echo "Wydanie na GitHubie:  ./release.sh v$(tr -d ' \n' < VERSION)"
