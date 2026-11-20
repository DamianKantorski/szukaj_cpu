#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  build_ffmpeg.sh – minimalny ffmpeg/ffprobe dla szukaj_cpu
#
#  Buduje najnowsze stabilne FFmpeg (+ libjxl) w kontenerze Debiana 13 (trixie),
#  zoptymalizowane pod Intel Core i5-7500 (Kaby Lake, -march=skylake), tylko ze
#  składnikami potrzebnymi szukaj_cpu:
#    RTSP/RTP, H.264/HEVC (+VAAPI), MJPEG, MPEG-TS/MP4/MKV, segmenty, concat,
#    filtry scale/select/fps/format/hwdownload, zapis JPEG XL (libjxl), ffprobe.
#
#  Biblioteki kodeków są wbudowane statycznie. Dynamicznie linkowane zostają tylko
#  glibc oraz libva/libdrm z systemu – sterownik VAAPI (iHD) jest ładowany przez
#  libva w czasie działania, więc musi pochodzić z systemu docelowego.
#
#  Dlaczego Debian 13, a nie sid: program zbudowany na nowszej glibc nie uruchomi się
#  na starszej. Serwer ma Debiana 13, więc budujemy na tej samej glibc.
#
#  Użycie (na komputerze z Dockerem albo Podmanem, np. CachyOS):
#    bash build_ffmpeg.sh                 # wynik w ./ffmpeg-szukaj/
#    bash build_ffmpeg.sh /ścieżka/wyniku
#
#  Zmienne (opcjonalne):
#    FFMPEG_VERSION=n9.0.2   JXL_VERSION=v0.12.0   – zamiast najnowszych tagów
#    MARCH=skylake           – architektura docelowa (i5-7500 = Kaby Lake = skylake)
#    JOBS=16                 – liczba równoległych zadań kompilacji
#    LTO=1                   – optymalizacja przy linkowaniu (dłużej, ~kilka % szybciej)
#    IMAGE=debian:trixie     – obraz bazowy
#    NATIVE=1                – buduj bez kontenera (na Debianie 13), np. na samym serwerze
#
#  Potem skopiuj ffmpeg i ffprobe na serwer do ~/.szukaj_cpu/ (albo /opt/szukaj_cpu/);
#  w szukaj_cpu.toml: ffmpeg = "auto", ffprobe = "auto".
# ════════════════════════════════════════════════════════════════════════════
set -euo pipefail

MARCH="${MARCH:-skylake}"
IMAGE="${IMAGE:-debian:trixie}"
FF_REPO="https://github.com/FFmpeg/FFmpeg.git"
JXL_REPO="https://github.com/libjxl/libjxl.git"

latest_tag() {   # $1 repo, $2 regex tagu
    git ls-remote --tags --refs "$1" | sed 's|.*refs/tags/||' | grep -E "$2" | sort -V | tail -1
}

# ───────────────────────────── budowanie (w kontenerze) ─────────────────────────────
inside() {
    local OUT="${OUT:-/out}" SRC=/tmp/src PREFIX=/tmp/deps
    JOBS="${JOBS:-$(nproc)}"
    export DEBIAN_FRONTEND=noninteractive

    if [[ "${SKIP_APT:-0}" != 1 ]]; then
        echo "==> Pakiety do budowania"
        apt-get update -qq
        apt-get install -y -qq --no-install-recommends \
            build-essential nasm pkg-config cmake ninja-build git ca-certificates \
            libva-dev libdrm-dev >/dev/null
    fi

    FFMPEG_VERSION="${FFMPEG_VERSION:-$(latest_tag "$FF_REPO" '^n[0-9]+\.[0-9]+(\.[0-9]+)?$')}"
    JXL_VERSION="${JXL_VERSION:-$(latest_tag "$JXL_REPO" '^v[0-9]+\.[0-9]+(\.[0-9]+)?$')}"
    echo "==> FFmpeg $FFMPEG_VERSION, libjxl $JXL_VERSION, -march=$MARCH, $JOBS zadań"

    local CF="-O2 -pipe -march=$MARCH -mtune=$MARCH"
    export CFLAGS="$CF" CXXFLAGS="$CF"
    export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PREFIX/lib/x86_64-linux-gnu/pkgconfig"
    mkdir -p "$SRC" "$PREFIX" "$OUT"

    # ── libjxl (statycznie) z dołączonymi highway i brotli ──
    echo "==> libjxl"
    rm -rf "$SRC/libjxl"
    git -c advice.detachedHead=false clone -q --depth 1 --branch "$JXL_VERSION" "$JXL_REPO" "$SRC/libjxl"
    git -C "$SRC/libjxl" -c advice.detachedHead=false submodule update -q --init --depth 1 \
        third_party/highway third_party/brotli third_party/skcms
    local CM=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX"
              -DCMAKE_INSTALL_LIBDIR=lib -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF
              -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Wno-dev)

    cmake -S "$SRC/libjxl/third_party/highway" -B "$SRC/b-hwy" "${CM[@]}" \
        -DHWY_ENABLE_TESTS=OFF -DHWY_ENABLE_EXAMPLES=OFF -DHWY_ENABLE_CONTRIB=OFF \
        -DHWY_ENABLE_INSTALL=ON >/dev/null
    cmake --build "$SRC/b-hwy" -j "$JOBS" >/dev/null && cmake --install "$SRC/b-hwy" >/dev/null

    cmake -S "$SRC/libjxl/third_party/brotli" -B "$SRC/b-brotli" "${CM[@]}" \
        -DBROTLI_DISABLE_TESTS=ON -DBROTLI_BUILD_TOOLS=OFF >/dev/null
    cmake --build "$SRC/b-brotli" -j "$JOBS" >/dev/null && cmake --install "$SRC/b-brotli" >/dev/null

    cmake -S "$SRC/libjxl" -B "$SRC/b-jxl" "${CM[@]}" \
        -DJPEGXL_FORCE_SYSTEM_HWY=ON -DJPEGXL_FORCE_SYSTEM_BROTLI=ON \
        -DJPEGXL_ENABLE_SKCMS=ON \
        -DJPEGXL_ENABLE_TOOLS=OFF -DJPEGXL_ENABLE_DOXYGEN=OFF -DJPEGXL_ENABLE_MANPAGES=OFF \
        -DJPEGXL_ENABLE_BENCHMARK=OFF -DJPEGXL_ENABLE_EXAMPLES=OFF -DJPEGXL_ENABLE_JNI=OFF \
        -DJPEGXL_ENABLE_SJPEG=OFF -DJPEGXL_ENABLE_OPENEXR=OFF -DJPEGXL_ENABLE_PLUGINS=OFF \
        -DJPEGXL_ENABLE_TRANSCODE_JPEG=OFF \
        -DJPEGXL_ENABLE_BOXES=ON -DJPEGXL_ENABLE_FUZZERS=OFF -DJPEGXL_ENABLE_DEVTOOLS=OFF \
        -DJPEGXL_STATIC=OFF >/dev/null
    cmake --build "$SRC/b-jxl" -j "$JOBS" >/dev/null && cmake --install "$SRC/b-jxl" >/dev/null
    # libstdc++ wbudowana statycznie (pkg-config libjxl dodaje -lstdc++ = wersja dynamiczna)
    sed -i 's/-lstdc++/-l:libstdc++.a/g' "$PREFIX"/lib/pkgconfig/*.pc

    # ── FFmpeg: tylko to, czego używa szukaj_cpu ──
    echo "==> FFmpeg"
    rm -rf "$SRC/ffmpeg"
    git -c advice.detachedHead=false clone -q --depth 1 --branch "$FFMPEG_VERSION" "$FF_REPO" "$SRC/ffmpeg"
    cd "$SRC/ffmpeg"
    local VAAPI=(--enable-vaapi --enable-hwaccel=h264_vaapi,hevc_vaapi --enable-filter=hwdownload)
    [[ "${NO_VAAPI:-0}" == 1 ]] && VAAPI=()
    local LTOF=()
    [[ "${LTO:-0}" == 1 ]] && LTOF=(--enable-lto)
    ./configure \
        --prefix="$PREFIX" --pkg-config-flags=--static \
        --extra-cflags="$CF -I$PREFIX/include" --extra-cxxflags="$CF" \
        --extra-ldflags="-L$PREFIX/lib" \
        --extra-libs="-l:libstdc++.a -lm -lpthread" \
        --extra-ldexeflags="-static-libgcc -static-libstdc++" \
        --disable-shared --enable-static --enable-pic "${LTOF[@]}" \
        --disable-autodetect --disable-everything \
        --disable-doc --disable-debug --disable-ffplay \
        --enable-ffmpeg --enable-ffprobe --enable-network \
        --enable-avcodec --enable-avformat --enable-avfilter --enable-swscale --enable-swresample \
        "${VAAPI[@]}" \
        --enable-libjxl \
        --enable-protocol=file,pipe,tcp,udp,rtp,srtp \
        --enable-demuxer=rtsp,sdp,rtp,mov,mpegts,matroska,concat,rawvideo,h264,hevc,mjpeg,image2 \
        --enable-muxer=rawvideo,segment,mpegts,mp4,mov,matroska,image2,null,rtsp,rtp,adts \
        --enable-decoder=h264,hevc,mjpeg,rawvideo,libjxl,aac,pcm_mulaw,pcm_alaw,pcm_s16be,pcm_s16le,pcm_s24be,pcm_u8,adpcm_g726,adpcm_g726le,adpcm_g722,opus,mp3float \
        --enable-encoder=rawvideo,wrapped_avframe,libjxl,aac \
        --enable-parser=h264,hevc,mjpeg,aac,opus \
        --enable-bsf=h264_mp4toannexb,hevc_mp4toannexb,aac_adtstoasc,extract_extradata,dump_extradata,null \
        --enable-filter=scale,select,fps,format,null,copy,setpts,trim,aresample,aformat,anull,asetpts \
        > "$SRC/configure.log" 2>&1 || { tail -40 "$SRC/configure.log"; tail -60 ffbuild/config.log; exit 1; }
    make -j "$JOBS" ffmpeg ffprobe > "$SRC/make.log" 2>&1 || { grep -E "error|Error" "$SRC/make.log" | tail -30; exit 1; }
    strip ffmpeg ffprobe
    cp ffmpeg ffprobe "$OUT/"

    # ── sprawdzenie i opis ──
    {
        echo "FFmpeg:     $FFMPEG_VERSION"
        echo "libjxl:     $JXL_VERSION"
        echo "Cel:        -march=$MARCH (Intel Core i5-7500 / Kaby Lake)"
        echo "Zbudowano:  $(date -Is) na $(. /etc/os-release; echo "$PRETTY_NAME"), $(ldd --version | head -1)"
        echo
        echo "== Zależności dynamiczne (ldd) =="
        ldd "$OUT/ffmpeg" | sed 's/ (0x[0-9a-f]*)//'
        echo
        echo "== hwaccels ==";  "$OUT/ffmpeg" -hide_banner -hwaccels | tail -n +2
        echo "== enkodery =="; "$OUT/ffmpeg" -hide_banner -encoders | grep -E '^ V|^ A' || true
        echo "== filtry ==";   "$OUT/ffmpeg" -hide_banner -filters 2>/dev/null | awk 'NF>=3 && $2 ~ /^[a-z0-9_]+$/ {printf "%s ", $2}' || true; echo
        echo "== protokoły =="; "$OUT/ffmpeg" -hide_banner -protocols 2>/dev/null | tr '\n' ' ' || true; echo
    } > "$OUT/BUILDINFO.txt"
    "$OUT/ffmpeg" -hide_banner -version | head -1
    echo "==> Gotowe: $OUT/ffmpeg, $OUT/ffprobe ($(du -h "$OUT/ffmpeg" | cut -f1), $(du -h "$OUT/ffprobe" | cut -f1)), opis: $OUT/BUILDINFO.txt"
}

if [[ "${1:-}" == "--inside" ]]; then
    inside
    exit 0
fi

# ───────────────────────────── uruchomienie (komputer budujący) ─────────────────────────────
OUT_HOST="$(realpath -m "${1:-./ffmpeg-szukaj}")"
mkdir -p "$OUT_HOST"
SELF="$(realpath "$0")"

if [[ "${NATIVE:-0}" == 1 ]]; then
    OUT="$OUT_HOST" inside
    exit 0
fi

if command -v docker >/dev/null 2>&1; then RT=docker
elif command -v podman >/dev/null 2>&1; then RT=podman
else
    echo "Brak dockera i podmana. Zainstaluj jeden z nich (CachyOS: sudo pacman -S docker" \
         "&& sudo systemctl start docker) albo uruchom z NATIVE=1 na Debianie 13." >&2
    exit 1
fi

echo "==> Budowanie w kontenerze $IMAGE ($RT), wynik: $OUT_HOST"
"$RT" run --rm \
    -e FFMPEG_VERSION="${FFMPEG_VERSION:-}" -e JXL_VERSION="${JXL_VERSION:-}" \
    -e MARCH="$MARCH" -e JOBS="${JOBS:-}" -e LTO="${LTO:-0}" -e NO_VAAPI="${NO_VAAPI:-0}" \
    -v "$SELF:/build_ffmpeg.sh:ro" -v "$OUT_HOST:/out" \
    "$IMAGE" bash -c "bash /build_ffmpeg.sh --inside && chown -R $(id -u):$(id -g) /out"

echo
echo "Skopiuj na serwer, np.:"
echo "  scp $OUT_HOST/ffmpeg $OUT_HOST/ffprobe UZYTKOWNIK@SERWER:~/.szukaj_cpu/"
