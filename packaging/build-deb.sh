#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  build-deb.sh – buduje dist/szukaj-cpu_<wersja>_<arch>.deb
#
#  Pełny pakiet (domyślnie) zawiera wszystko, co wcześniej instalacja pobierała z sieci:
#    • koła Pythona (OpenVINO, OpenCV, NumPy) dla Pythona z Debiana 13 – instalacja offline,
#    • modele YOLO OpenVINO: FP32 i INT8 (kwantyzacja NNCF) dla MODEL_SIZES,
#    • ffmpeg/ffprobe skompilowane na tym komputerze (tools/build_ffmpeg.sh, -march=$MARCH).
#  Wszystko powstaje w kontenerze Debiana 13 (Docker albo Podman), więc na komputerze
#  budującym (np. CachyOS, WSL) wystarczy kontener – bez Pythona, dpkg ani kompilatorów.
#  Wyniki pośrednie są w build/ (pamięć podręczna): ponowne budowanie trwa chwilę,
#  ffmpeg i modele powstają tylko za pierwszym razem (albo po --rebuild-…).
#
#  Użycie:
#    bash packaging/build-deb.sh                      # pełny pakiet (amd64)
#    bash packaging/build-deb.sh --calib ~/klatki     # INT8 kalibrowany na klatkach z własnych kamer
#    bash packaging/build-deb.sh --slim               # sam program; zależności pobiera instalacja
#  Opcje:
#    --rebuild-ffmpeg | --rebuild-models | --rebuild-wheels | --rebuild-all
#    --no-container      buduj bez kontenera (Debian 13 albo zgodny system z apt)
#    --no-ffmpeg | --no-models | --no-wheels   pomiń składnik
#  Zmienne:
#    MODEL_SIZES="960x544 640x384"  MODEL=yolo11s.pt  MARCH=skylake  JOBS=…  LTO=1
#    FFMPEG_VERSION=n8.0  JXL_VERSION=v0.11.1   (domyślnie najnowsze wydania)
#    IMAGE=debian:trixie   BUILD_CA_CERT=/ścieżka/ca.crt (gdy sieć ma własny certyfikat proxy)
# ════════════════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${IMAGE:-debian:trixie}"
MODEL_SIZES="${MODEL_SIZES:-960x544}"
MODEL="${MODEL:-yolo11s.pt}"
MARCH="${MARCH:-skylake}"

SLIM=0 INSIDE=0 NO_CONTAINER=0 CALIB=""
DO_FFMPEG=1 DO_MODELS=1 DO_WHEELS=1
RB_FFMPEG=0 RB_MODELS=0 RB_WHEELS=0
ARGS=("$@")
while [ $# -gt 0 ]; do
    case "$1" in
        --slim) SLIM=1 ;;
        --inside) INSIDE=1 ;;
        --no-container) NO_CONTAINER=1 ;;
        --calib) CALIB="$2"; shift ;;
        --rebuild-ffmpeg) RB_FFMPEG=1 ;;
        --rebuild-models) RB_MODELS=1 ;;
        --rebuild-wheels) RB_WHEELS=1 ;;
        --rebuild-all) RB_FFMPEG=1; RB_MODELS=1; RB_WHEELS=1 ;;
        --no-ffmpeg) DO_FFMPEG=0 ;;
        --no-models) DO_MODELS=0 ;;
        --no-wheels) DO_WHEELS=0 ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "Nieznana opcja: $1" >&2; exit 2 ;;
    esac
    shift
done
[ "$SLIM" = 1 ] && { DO_FFMPEG=0; DO_MODELS=0; DO_WHEELS=0; }
FULL=$(( DO_FFMPEG || DO_MODELS || DO_WHEELS ))
B="$ROOT/build"

# ───────────────────────── uruchomienie w kontenerze ─────────────────────────
if [ "$INSIDE" = 0 ] && [ "$NO_CONTAINER" = 0 ] && { [ "$FULL" = 1 ] || ! command -v dpkg-deb >/dev/null 2>&1; }; then
    if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then RT=docker
    elif command -v podman >/dev/null 2>&1; then RT=podman
    else
        echo "Potrzebny Docker albo Podman (CachyOS: sudo pacman -S podman; WSL/Ubuntu: sudo apt install podman)," >&2
        echo "albo --no-container na Debianie 13. Sam program bez dodatków: --slim (wymaga dpkg-deb)." >&2
        exit 1
    fi
    mkdir -p "$B"
    VOL=(-v "$ROOT:/src")
    ENVS=(-e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" -e MODEL_SIZES="$MODEL_SIZES" -e MODEL="$MODEL"
          -e MARCH="$MARCH" -e JOBS="${JOBS:-}" -e LTO="${LTO:-0}"
          -e FFMPEG_VERSION="${FFMPEG_VERSION:-}" -e JXL_VERSION="${JXL_VERSION:-}")
    INNER=()
    for a in "${ARGS[@]}"; do INNER+=("$a"); done
    if [ -n "$CALIB" ]; then
        VOL+=(-v "$(realpath "$CALIB"):/calib:ro")
        # --calib KATALOG -> --calib /calib
        for i in "${!INNER[@]}"; do [ "${INNER[$i]}" = --calib ] && INNER[$((i + 1))]=/calib; done
    fi
    if [ -n "${BUILD_CA_CERT:-}" ]; then
        VOL+=(-v "$(realpath "$BUILD_CA_CERT"):/build-ca.crt:ro")
        ENVS+=(-e BUILD_CA_CERT=/build-ca.crt)
    fi
    for v in http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY; do
        [ -n "${!v:-}" ] && ENVS+=(-e "$v=${!v}")
    done
    echo "==> Budowanie w kontenerze $IMAGE ($RT)"
    "$RT" run --rm --network host "${VOL[@]}" "${ENVS[@]}" "$IMAGE" \
        bash /src/packaging/build-deb.sh --inside "${INNER[@]}"
    exit $?
fi

# ───────────────────────── budowanie (kontener albo --no-container) ─────────────────────────
VERSION="$(tr -d ' \n' < "$ROOT/VERSION")"
export DEBIAN_FRONTEND=noninteractive
if [ -n "${BUILD_CA_CERT:-}" ]; then
    export PIP_CERT="$BUILD_CA_CERT" SSL_CERT_FILE="$BUILD_CA_CERT" REQUESTS_CA_BUNDLE="$BUILD_CA_CERT" \
           CURL_CA_BUNDLE="$BUILD_CA_CERT" GIT_SSL_CAINFO="$BUILD_CA_CERT"
fi
own() {   # pliki w build/ i dist/ mają należeć do użytkownika, który uruchomił budowanie
    [ -n "${HOST_UID:-}" ] && chown -R "$HOST_UID:${HOST_GID:-$HOST_UID}" "$@" 2>/dev/null || true
}
trap 'own "$B" "$ROOT/dist"' EXIT

if [ "$FULL" = 1 ] && command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = 0 ]; then
    echo "==> Pakiety do budowania"
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends python3 python3-venv python3-pip ca-certificates curl git \
        libgl1 libglib2.0-0 libva2 libva-drm2 libdrm2 >/dev/null
fi
PY="$(command -v python3)"
PYTAG="cp$("$PY" -c 'import sys; print("%d%d" % sys.version_info[:2])')"
mkdir -p "$B"

# ── 1. koła Pythona (instalacja offline na serwerze) ──
WHEELS="$B/wheels-$PYTAG"
if [ "$DO_WHEELS" = 1 ]; then
    STAMP="$(sha256sum "$ROOT/requirements.txt" | cut -c1-16)"
    if [ "$RB_WHEELS" = 1 ] || [ "$(cat "$WHEELS/.stamp" 2>/dev/null)" != "$STAMP" ]; then
        echo "==> Koła Pythona ($PYTAG): $(grep -vE '^\s*(#|$)' "$ROOT/requirements.txt" | tr '\n' ' ')"
        rm -rf "$WHEELS"; mkdir -p "$WHEELS"
        "$PY" -m venv "$B/venv-pip"
        "$B/venv-pip/bin/pip" download -q --disable-pip-version-check --cache-dir "$B/cache/pip" \
            --only-binary=:all: -d "$WHEELS" -r "$ROOT/requirements.txt"
        echo "$STAMP" > "$WHEELS/.stamp"
    fi
    echo "    koła: $(ls "$WHEELS"/*.whl | wc -l) plików, $(du -sh "$WHEELS" | cut -f1)"
fi

# ── 2. modele YOLO: FP32 + INT8 ──
MODELS="$B/models"
if [ "$DO_MODELS" = 1 ]; then
    VX="$B/venv-export"
    [ "$INSIDE" = 1 ] || VX="$B/venv-export-native"
    if ! "$VX/bin/python" -c "import ultralytics, torch, openvino, nncf" 2>/dev/null; then
        echo "==> Narzędzia eksportu (PyTorch CPU, Ultralytics, OpenVINO, NNCF) – jednorazowo, ~1,5 GB w build/"
        "$PY" -m venv "$VX"
        "$VX/bin/pip" install -q --disable-pip-version-check --cache-dir "$B/cache/pip" \
            torch torchvision --index-url https://download.pytorch.org/whl/cpu
        "$VX/bin/pip" install -q --disable-pip-version-check --cache-dir "$B/cache/pip" ultralytics openvino nncf
    fi
    export HOME="$B/home" YOLO_CONFIG_DIR="$B/home/ultralytics" YOLO_OFFLINE=1
    mkdir -p "$HOME" "$B/datasets"
    "$VX/bin/yolo" settings datasets_dir="$B/datasets" sync=False >/dev/null 2>&1
    # coco128: kalibracja INT8 (gdy brak --calib) i raport porównania
    "$VX/bin/python" -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('coco128.yaml')" \
        >/dev/null 2>&1 || echo "    (nie udało się pobrać coco128 – raport porównania zostanie pominięty)"
    [ "$RB_MODELS" = 1 ] && find "$MODELS" -maxdepth 1 -name '*_openvino_model' -exec rm -rf {} +
    mkdir -p "$MODELS"
    (cd "$MODELS" && "$VX/bin/python" "$ROOT/tools/export_models.py" --model "$MODEL" --out "$MODELS" \
        --sizes $MODEL_SIZES ${CALIB:+--calib "$CALIB"} 2>&1 | grep -vE "^INFO:nncf|^\s*$")
    # porównanie FP32 / INT8 (raport dołączany do pakietu)
    IMGS="${CALIB:-$(find "$B/datasets" -type d -path '*coco128/images/train2017' | head -1)}"
    STEM="$(basename "$MODEL" .pt)"
    : > "$MODELS/RAPORT-INT8.txt"
    for s in $MODEL_SIZES; do
        if [ -n "$IMGS" ] && [ -d "$MODELS/${STEM}_${s}_int8_openvino_model" ]; then
            echo "==> Porównanie FP32 / INT8 ($s) na $(basename "$IMGS")"
            "$VX/bin/python" "$ROOT/tools/compare_models.py" "$MODELS/${STEM}_${s}_openvino_model" \
                "$MODELS/${STEM}_${s}_int8_openvino_model" "$IMGS" --threads 4 | tee -a "$MODELS/RAPORT-INT8.txt"
            echo >> "$MODELS/RAPORT-INT8.txt"
        fi
    done
fi

# ── 3. ffmpeg / ffprobe skompilowane lokalnie ──
FF="$B/ffmpeg"
if [ "$DO_FFMPEG" = 1 ]; then
    if [ "$RB_FFMPEG" = 1 ] || [ ! -x "$FF/ffmpeg" ] || [ ! -x "$FF/ffprobe" ]; then
        echo "==> ffmpeg (-march=$MARCH) – kompilacja, kilka–kilkanaście minut"
        rm -rf "$FF"
        if [ "$INSIDE" = 1 ]; then
            OUT="$FF" bash "$ROOT/tools/build_ffmpeg.sh" --inside
        else
            NATIVE=1 bash "$ROOT/tools/build_ffmpeg.sh" "$FF"
        fi
    fi
    echo "    $("$FF/ffmpeg" -hide_banner -version | head -1)"
fi

# ── 4. pakiet ──
ARCH=all
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"; own "$B" "$ROOT/dist"' EXIT
chmod 755 "$STAGE"
OPT="$STAGE/opt/szukaj_cpu"
DOC="$STAGE/usr/share/doc/szukaj-cpu"
install -d "$OPT/tools" "$OPT/ntfy" "$STAGE/usr/bin" "$STAGE/usr/lib/systemd/system" "$DOC" "$STAGE/DEBIAN"

install -m 755 "$ROOT/src/run.py"                   "$OPT/run.py"
(cd "$ROOT/src" && find szukaj_cpu -type f \( -name '*.py' -o -name '*.html' \) -print0 |
    while IFS= read -r -d '' f; do install -D -m 644 "$f" "$OPT/$f"; done)
install -m 644 "$ROOT/config/szukaj_cpu.toml"       "$OPT/szukaj_cpu.toml.example"
install -m 644 "$ROOT/requirements.txt"             "$OPT/requirements.txt"
for t in build_ffmpeg.sh test_przerwy.py export_models.py compare_models.py; do
    install -m 755 "$ROOT/tools/$t" "$OPT/tools/"
done
install -m 644 "$ROOT/ntfy/server.yml.in"           "$OPT/ntfy/"
install -m 644 "$ROOT/ntfy/docker-compose.yml"      "$OPT/ntfy/"
install -m 755 "$ROOT/ntfy/setup-ntfy.sh"           "$OPT/ntfy/"
install -m 755 "$ROOT/uninstall.sh"                 "$OPT/uninstall.sh"
install -m 755 "$ROOT/bin/szukaj-cpu"               "$STAGE/usr/bin/"
install -m 755 "$ROOT/bin/szukaj-cpu-export-tools"  "$STAGE/usr/bin/"
install -m 644 "$ROOT/systemd/szukaj_cpu@.service"  "$STAGE/usr/lib/systemd/system/"
install -m 644 "$ROOT/README.md"                    "$DOC/"

CONTENTS=()
if [ "$DO_WHEELS" = 1 ]; then
    install -d "$OPT/wheels"
    install -m 644 "$WHEELS"/*.whl "$OPT/wheels/"
    echo "$PYTAG" > "$OPT/wheels/PYTHON"
    CONTENTS+=("Python wheels for $PYTAG (offline install)")
fi
if [ "$DO_MODELS" = 1 ]; then
    install -d "$OPT/models"
    for d in "$MODELS"/*_openvino_model; do
        [ -d "$d" ] || continue
        install -d "$OPT/models/$(basename "$d")"
        install -m 644 "$d"/* "$OPT/models/$(basename "$d")/"
    done
    [ -s "$MODELS/RAPORT-INT8.txt" ] && install -m 644 "$MODELS/RAPORT-INT8.txt" "$DOC/"
    cat > "$DOC/MODELS.txt" <<'EOF'
The YOLO11 models in /opt/szukaj_cpu/models were exported from Ultralytics YOLO11
weights (https://github.com/ultralytics/ultralytics), which are licensed under AGPL-3.0.
INT8 variants were quantized with OpenVINO NNCF.
EOF
    CONTENTS+=("YOLO models: $(cd "$OPT/models" && ls -d *_openvino_model | tr '\n' ' ')")
fi
if [ "$DO_FFMPEG" = 1 ]; then
    install -m 755 "$FF/ffmpeg" "$FF/ffprobe" "$OPT/"
    [ -f "$FF/BUILDINFO.txt" ] && install -m 644 "$FF/BUILDINFO.txt" "$OPT/FFMPEG_BUILDINFO.txt"
    ARCH=amd64
    CONTENTS+=("static ffmpeg/ffprobe ($("$FF/ffmpeg" -hide_banner -version | head -1 | cut -d' ' -f3), -march=$MARCH)")
fi
for f in postinst prerm postrm; do
    install -m 755 "$ROOT/packaging/debian/$f" "$STAGE/DEBIAN/$f"
done
SIZE="$(du -sk --exclude=DEBIAN "$STAGE" | cut -f1)"
sed -e "s/@VERSION@/$VERSION/" -e "s/@SIZE@/$SIZE/" -e "s/^Architecture: .*/Architecture: $ARCH/" \
    "$ROOT/packaging/debian/control.in" > "$STAGE/DEBIAN/control"
if [ "$DO_FFMPEG" = 1 ]; then   # dołączony ffmpeg: libva/libdrm z systemu (sterownik VAAPI ładuje libva)
    sed -i -e 's/^Recommends: ffmpeg, /Recommends: /' -e 's/^Depends: .*/&, libva2, libva-drm2, libdrm2/' "$STAGE/DEBIAN/control"
fi
if [ ${#CONTENTS[@]} -gt 0 ]; then
    { echo " ."; echo " This build also contains:"; for c in "${CONTENTS[@]}"; do echo "  * $c"; done; } >> "$STAGE/DEBIAN/control"
fi

mkdir -p "$ROOT/dist"
OUT="$ROOT/dist/szukaj-cpu_${VERSION}_${ARCH}.deb"
dpkg-deb --root-owner-group -Zxz --build "$STAGE" "$OUT" >/dev/null
echo "Zbudowano: dist/$(basename "$OUT") ($(du -h "$OUT" | cut -f1))"
for c in "${CONTENTS[@]}"; do echo "  • $c"; done
