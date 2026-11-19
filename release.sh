#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  release.sh – wydanie na GitHubie z paczki zbudowanej lokalnie:
#     ./build-local.sh && ./release.sh v1.9.0
#
#  • tag musi zgadzać się z plikiem VERSION, a w dist/ musi być paczka tej wersji,
#  • bieżący commit musi być już na GitHubie (git push) – tag powstaje dokładnie na nim,
#    więc nie da się otagować starszego main (jak przy ręcznym git tag przed scaleniem PR),
#  • opis wydania z release-notes/<tag>.md (bez pliku – notatki generowane przez GitHub),
#  • istniejące wydanie dostaje pliki podmienione (--clobber), pliki innych wersji są usuwane.
#  Wymaga zalogowanego gh (gh auth login).   --dry-run = tylko pokaż, co zostałoby zrobione.
# ════════════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
DRY=0
ARGS=()
for a in "$@"; do [ "$a" = --dry-run ] && DRY=1 || ARGS+=("$a"); done
TAG="${ARGS[0]:-v$(tr -d ' \n' < VERSION)}"
VER="${TAG#v}"
FILE_VER="$(tr -d ' \n' < VERSION)"
run() { if [ "$DRY" = 1 ]; then echo "  [dry-run] $*"; else "$@"; fi; }

[ "$VER" = "$FILE_VER" ] || { echo "BŁĄD: tag $TAG nie zgadza się z plikiem VERSION ($FILE_VER)" >&2; exit 1; }
command -v gh >/dev/null || { echo "BŁĄD: brak gh (GitHub CLI) – zainstaluj i zaloguj: gh auth login" >&2; exit 1; }
[ "$DRY" = 1 ] || gh auth status >/dev/null 2>&1 || { echo "BŁĄD: gh nie jest zalogowany – gh auth login" >&2; exit 1; }

shopt -s nullglob
FILES=(dist/szukaj-cpu_"$VER"_*.deb dist/szukaj-cpu_"$VER"_*.deb.sha256)
[ ${#FILES[@]} -gt 0 ] || { echo "BŁĄD: brak dist/szukaj-cpu_${VER}_*.deb – najpierw ./build-local.sh" >&2; exit 1; }

# commit z tagiem = bieżący HEAD, który musi już być na GitHubie
[ -z "$(git status --porcelain --untracked-files=no)" ] ||
    echo "UWAGA: masz niezatwierdzone zmiany – wydanie obejmie stan z ostatniego commita, nie z dysku" >&2
HEAD="$(git rev-parse HEAD)"
git fetch -q origin
if ! git branch -r --contains "$HEAD" | grep -q .; then
    echo "BŁĄD: bieżący commit ${HEAD:0:7} nie jest na GitHubie – najpierw git push (albo scal PR i git pull)" >&2
    exit 1
fi
if git ls-remote --exit-code --tags origin "refs/tags/$TAG" >/dev/null 2>&1; then
    REMOTE="$(git ls-remote --tags origin "refs/tags/$TAG^{}" | cut -f1)"
    [ -n "$REMOTE" ] || REMOTE="$(git ls-remote --tags origin "refs/tags/$TAG" | cut -f1)"
    if [ "$REMOTE" != "$HEAD" ]; then
        echo "BŁĄD: tag $TAG na GitHubie wskazuje ${REMOTE:0:7}, a bieżący commit to ${HEAD:0:7}." >&2
        echo "  Usuń go i uruchom ponownie:  git push origin :refs/tags/$TAG && git tag -d $TAG" >&2
        exit 1
    fi
fi

printf 'Wydanie %s (commit %s), pliki:\n' "$TAG" "${HEAD:0:7}"; printf '  %s\n' "${FILES[@]}"
NOTES="release-notes/$TAG.md"
if [ -f "$NOTES" ]; then NOTE_ARGS=(--notes-file "$NOTES"); else NOTE_ARGS=(--generate-notes); fi

if gh release view "$TAG" >/dev/null 2>&1; then
    run gh release upload "$TAG" "${FILES[@]}" --clobber
    [ -f "$NOTES" ] && run gh release edit "$TAG" --notes-file "$NOTES"
else
    run gh release create "$TAG" "${FILES[@]}" --target "$HEAD" --title "szukaj_cpu $VER" "${NOTE_ARGS[@]}"
fi
# wydanie zawiera tylko pliki tej wersji
if [ "$DRY" = 0 ]; then
    while read -r asset; do
        [ -z "$asset" ] || [[ "$asset" == *"$VER"* ]] && continue
        echo "Usuwam plik innej wersji: $asset"
        gh release delete-asset "$TAG" "$asset" --yes
    done < <(gh release view "$TAG" --json assets --jq '.assets[].name')
    git fetch -q origin --tags
    echo "Gotowe: $(gh release view "$TAG" --json url -q .url)"
fi
