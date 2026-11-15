#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  uninstall.sh – usuwa instalację szukaj_cpu
#
#  Domyślnie: STARA instalacja (install.sh → ~/szukaj_cpu + szukaj_cpu.service).
#  Przed usunięciem przenosi do ~/.szukaj_cpu (nowe miejsce):
#    • szukaj_cpu.toml (ścieżki modelu poprawione na ~/.szukaj_cpu/models),
#    • modele: *.pt i *_openvino_model/  →  ~/.szukaj_cpu/models/,
#    • statyczny ffmpeg/ffprobe (z build_ffmpeg.sh)  →  ~/.szukaj_cpu/.
#  Galerii, nagrań Shinobi ani serwera ntfy nie rusza.
#
#  Uruchom jako zwykły użytkownik (skrypt sam użyje sudo):
#    bash uninstall.sh                # stara instalacja, z przeniesieniem konfiguracji
#    bash uninstall.sh --package      # pakiet .deb (sudo apt remove szukaj-cpu)
#    bash uninstall.sh --all          # stara instalacja i pakiet
#  Opcje:
#    --no-migrate   nie przenoś konfiguracji/modeli (usuń wszystko ze starego katalogu)
#    --purge        dodatkowo usuń ~/.szukaj_cpu (konfiguracja, modele) i pakiet z purge
#    -n, --dry-run  tylko pokaż, co zostałoby zrobione
#    -y, --yes      bez pytań
# ════════════════════════════════════════════════════════════════════════════
set -euo pipefail

DO_LEGACY=1; DO_PKG=0; MIGRATE=1; PURGE=0; DRY=0; YES=0
for a in "$@"; do
    case "$a" in
        --package)    DO_LEGACY=0; DO_PKG=1 ;;
        --all)        DO_LEGACY=1; DO_PKG=1 ;;
        --no-migrate) MIGRATE=0 ;;
        --purge)      PURGE=1 ;;
        -n|--dry-run) DRY=1 ;;
        -y|--yes)     YES=1 ;;
        -h|--help)    sed -n '2,24p' "$0"; exit 0 ;;
        *) echo "Nieznana opcja: $a" >&2; exit 2 ;;
    esac
done

# użytkownik, którego instalację usuwamy (także przy uruchomieniu przez sudo)
USR="${SUDO_USER:-$(id -un)}"
HOME_DIR="$(getent passwd "$USR" | cut -d: -f6)"
OLD="$HOME_DIR/szukaj_cpu"
NEW="$HOME_DIR/.szukaj_cpu"
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"

run() { if [ "$DRY" = 1 ]; then echo "  [dry-run] $*"; else "$@"; fi; }
ask() {
    [ "$YES" = 1 ] && return 0
    [ "$DRY" = 1 ] && return 0
    read -r -p "$1 [t/N] " r; [[ "$r" =~ ^[tTyY]$ ]]
}
as_user() {   # pliki w katalogu domowym mają należeć do użytkownika
    if [ "$(id -u)" -eq 0 ] && [ "$USR" != root ]; then run runuser -u "$USR" -- "$@"; else run "$@"; fi
}

echo "Użytkownik: $USR ($HOME_DIR)"

# ───────────────────────── stara instalacja ─────────────────────────
if [ "$DO_LEGACY" = 1 ]; then
    UNIT=/etc/systemd/system/szukaj_cpu.service
    if [ ! -d "$OLD" ] && [ ! -f "$UNIT" ]; then
        echo "Nie znaleziono starej instalacji ($OLD, $UNIT)."
    else
        echo "==> Stara instalacja: $OLD"
        if [ -f "$UNIT" ] || systemctl list-unit-files szukaj_cpu.service >/dev/null 2>&1; then
            echo "  zatrzymanie i wyłączenie szukaj_cpu.service"
            run $SUDO systemctl disable --now szukaj_cpu.service 2>/dev/null || true
            run $SUDO rm -f "$UNIT"
            run $SUDO rm -rf /etc/systemd/system/szukaj_cpu.service.d
            run $SUDO systemctl daemon-reload 2>/dev/null || true
            run $SUDO systemctl reset-failed szukaj_cpu.service 2>/dev/null || true
        fi

        if [ "$MIGRATE" = 1 ] && [ -d "$OLD" ]; then
            echo "==> Przenoszenie do $NEW"
            as_user mkdir -p "$NEW/models"
            as_user chmod 700 "$NEW"
            if [ -f "$OLD/szukaj_cpu.toml" ]; then
                if [ -f "$NEW/szukaj_cpu.toml" ]; then
                    echo "  $NEW/szukaj_cpu.toml już istnieje – stary plik zapisuję jako szukaj_cpu.toml.old"
                    as_user cp -p "$OLD/szukaj_cpu.toml" "$NEW/szukaj_cpu.toml.old"
                else
                    as_user cp -p "$OLD/szukaj_cpu.toml" "$NEW/szukaj_cpu.toml"
                    # model w starym katalogu -> ~/.szukaj_cpu/models; ffmpeg ze starego katalogu -> auto
                    as_user sed -i \
                        -e "s#^model = \"\(~\|$HOME_DIR\)/szukaj_cpu/#model = \"~/.szukaj_cpu/models/#" \
                        -e "s#^\(ffmpeg\|ffprobe\) = \"\(~\|$HOME_DIR\)/szukaj_cpu/.*\"#\1 = \"auto\"#" \
                        "$NEW/szukaj_cpu.toml"
                    echo "  konfiguracja: $NEW/szukaj_cpu.toml"
                fi
                as_user chmod 600 "$NEW/szukaj_cpu.toml"
            fi
            shopt -s nullglob
            for m in "$OLD"/*.pt "$OLD"/*_openvino_model; do
                if [ -e "$NEW/models/$(basename "$m")" ]; then
                    echo "  model $(basename "$m") już jest w $NEW/models – pomijam"
                else
                    as_user cp -a "$m" "$NEW/models/"
                    echo "  model: $(basename "$m")"
                fi
            done
            shopt -u nullglob
            for b in ffmpeg ffprobe; do
                if [ -f "$OLD/$b" ] && [ ! -L "$OLD/$b" ] && [ ! -e "$NEW/$b" ]; then
                    as_user cp -p "$OLD/$b" "$NEW/$b"
                    echo "  statyczny $b → $NEW/$b"
                fi
            done
        fi

        if [ -d "$OLD" ]; then
            if ask "Usunąć katalog $OLD (skrypt, venv, kopie)?"; then
                run rm -rf "$OLD" 2>/dev/null || run $SUDO rm -rf "$OLD"
                [ "$DRY" = 1 ] || echo "  usunięto $OLD"
            else
                echo "  pozostawiono $OLD"
            fi
        fi
        run $SUDO rm -rf /dev/shm/szukaj_cpu
    fi
fi

# ───────────────────────── pakiet .deb ─────────────────────────
if [ "$DO_PKG" = 1 ]; then
    if dpkg -s szukaj-cpu >/dev/null 2>&1; then
        echo "==> Pakiet szukaj-cpu"
        for u in $(systemctl list-unit-files --plain --no-legend 'szukaj_cpu@*' 2>/dev/null | awk '{print $1}') \
                 $(systemctl list-units --all --plain --no-legend 'szukaj_cpu@*' 2>/dev/null | awk '{print $1}'); do
            [[ "$u" == *@.service ]] && continue
            run $SUDO systemctl disable --now "$u" 2>/dev/null || true
        done
        if [ "$PURGE" = 1 ]; then run $SUDO apt-get purge -y szukaj-cpu; else run $SUDO apt-get remove -y szukaj-cpu; fi
    else
        echo "Pakiet szukaj-cpu nie jest zainstalowany."
    fi
fi

# ───────────────────────── konfiguracja użytkownika ─────────────────────────
if [ "$PURGE" = 1 ] && [ -d "$NEW" ]; then
    if ask "Usunąć także $NEW (konfiguracja z hasłami kamer, modele)?"; then
        run rm -rf "$NEW"
        [ "$DRY" = 1 ] || echo "  usunięto $NEW"
    fi
fi

if [ "$DRY" = 1 ]; then echo "Gotowe (podgląd – nic nie zostało zmienione)."; else echo "Gotowe."; fi
if [ "$DO_LEGACY" = 1 ] && [ "$MIGRATE" = 1 ] && ! dpkg -s szukaj-cpu >/dev/null 2>&1; then
    HERE="$(cd "$(dirname "$0")" && pwd)"
    echo "Dalej:"
    echo "  bash $HERE/packaging/build-deb.sh            # jeśli pakiet nie jest jeszcze zbudowany"
    echo "  sudo apt install $HERE/dist/szukaj-cpu_*.deb"
    echo "  szukaj-cpu --test"
    echo "  sudo systemctl enable --now szukaj_cpu@$USR"
    echo "Nieużywane modele w ~/.szukaj_cpu/models (inne niż ustawione w imgsz) można usunąć."
fi
