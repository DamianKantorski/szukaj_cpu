"""Edycja pliku TOML z zachowaniem komentarzy i układu."""

import json
import re


def toml_value(v):
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    return f"{round(float(v), 4):g}"


def _bracket_depth(line):
    code = re.sub(r'"(?:\\.|[^"\\])*"|\'[^\']*\'', '""', line).split("#", 1)[0]
    return code.count("[") - code.count("]")


def toml_set_camera(text, cam_name, values):
    """Ustaw klucze (już sformatowane wartości) w bloku [[camera]] o danej nazwie."""
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    i = 0
    while i < len(lines):
        if lines[i].strip() != "[[camera]]":
            i += 1
            continue
        # granice bloku: do następnego nagłówka [...] poza wartością wieloliniową
        end, depth = len(lines), 0
        for j in range(i + 1, len(lines)):
            st = lines[j].strip()
            if depth == 0 and st.startswith("["):
                end = j
                break
            depth += _bracket_depth(lines[j])
        block = lines[i + 1:end]
        name = None
        for ln in block:
            m = re.match(r'^\s*name\s*=\s*"([^"]*)"', ln)
            if m:
                name = re.sub(r"[^\w-]", "_", m.group(1).strip())
        if name != cam_name:
            i = end
            continue
        for key, val in values.items():
            k, found = 0, False
            while k < len(block):
                if re.match(rf"^\s*{re.escape(key)}\s*=", block[k]):
                    d, e = _bracket_depth(block[k]), k + 1
                    while d > 0 and e < len(block):          # wartość na kilku liniach
                        d += _bracket_depth(block[e])
                        e += 1
                    block[k:e] = [f"{key} = {val}\n"]
                    found = True
                    break
                k += 1
            if not found:
                last = max((n for n, ln in enumerate(block) if ln.strip() and not ln.strip().startswith("#")),
                           default=-1)
                block.insert(last + 1, f"{key} = {val}\n")
        return "".join(lines[:i + 1] + block + lines[end:])
    raise KeyError(f"brak bloku [[camera]] o nazwie '{cam_name}'")


CAMERA_KEYS = ("name", "url", "analysis_url", "enabled", "shinobi_monitor")


def toml_dump(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(round(v, 6)) if v != int(v) or abs(v) >= 1e15 else f"{v:.1f}"
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(toml_dump(x) for x in v) + "]"
    raise ValueError(f"nieobsługiwany typ {type(v).__name__}")


def _headers(lines):
    """Granice tabel: lista (nagłówek, początek, koniec) z pominięciem nawiasów w wartościach."""
    out, depth, cur = [], 0, None
    for j, ln in enumerate(lines):
        st = ln.strip()
        if depth == 0 and st.startswith("[") and not st.startswith("#"):
            if cur:
                out.append((cur[0], cur[1], j))
            cur = (st.split("#", 1)[0].strip(), j)
            continue
        depth += _bracket_depth(ln)
    if cur:
        out.append((cur[0], cur[1], len(lines)))
    return out


def _inline_comment(line):
    """Komentarz na końcu linii (poza napisami), np. '   # opis'."""
    m = re.match(r'^((?:"(?:\\.|[^"\\])*"|\'[^\']*\'|[^#"\'])*)(\s*#.*)$', line.rstrip("\n"))
    if not m:
        return ""
    code = m.group(1)
    return code[len(code.rstrip()):] + m.group(2).lstrip()


def _set_key(block, key, val):
    """block = linie tabeli bez nagłówka; ustawia lub dopisuje key = val."""
    for k, ln in enumerate(block):
        if re.match(rf"^\s*{re.escape(key)}\s*=", ln):
            d, e = _bracket_depth(ln), k + 1
            while d > 0 and e < len(block):
                d += _bracket_depth(block[e])
                e += 1
            note = _inline_comment(ln) if e == k + 1 else ""
            block[k:e] = [f"{key} = {val}{note}\n"]
            return
    last = max((n for n, ln in enumerate(block) if ln.strip() and not ln.strip().startswith("#")), default=-1)
    if last < 0:      # pusta tabela (same komentarze) – za komentarzami
        last = max((n for n, ln in enumerate(block) if ln.strip()), default=-1)
    block.insert(last + 1, f"{key} = {val}\n")


def toml_apply(text, sections, cameras):
    """sections: {sekcja: {klucz: wartość}}; cameras: lista {"orig": nazwa|None, klucz: wartość}
    w docelowej kolejności (bloki usunięte = nieobecne). Komentarze i reszta pliku zostają."""
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    for sec, vals in sections.items():
        if not vals:
            continue
        hdr = next((h for h in _headers(lines) if h[0] == f"[{sec}]"), None)
        if hdr is None:
            first_cam = next((h[1] for h in _headers(lines) if h[0] == "[[camera]]"), len(lines))
            new = [f"[{sec}]\n"] + [f"{k} = {toml_dump(v)}\n" for k, v in vals.items()] + ["\n"]
            lines[first_cam:first_cam] = new
            continue
        _, s, e = hdr
        block = lines[s + 1:e]
        for k, v in vals.items():
            _set_key(block, k, toml_dump(v))
        lines[s + 1:e] = block

    def cam_name(block):
        for ln in block:
            m = re.match(r'^\s*name\s*=\s*"([^"]*)"', ln)
            if m:
                return re.sub(r"[^\w-]", "_", m.group(1).strip())
        return None

    cam_hdrs = [h for h in _headers(lines) if h[0] == "[[camera]]"]
    existing = {}
    for n, (_, s, e) in enumerate(cam_hdrs):
        existing.setdefault(cam_name(lines[s + 1:e]) or f"camera{n + 1}", (s, e))
    blocks, used = {}, set()
    for cam in cameras:
        if cam.get("orig") in existing:
            s, e = existing[cam["orig"]]
            block = lines[s + 1:e]
            used.add(cam["orig"])
        else:
            block = []
        for k in CAMERA_KEYS:
            if k in cam:
                _set_key(block, k, toml_dump(cam[k]))
        if not cam.get("orig") in existing:
            block += ["zone = []\n", "crop = []\n"]
        while block and not block[-1].strip():
            block.pop()
        blocks[id(cam)] = ["[[camera]]\n"] + block + ["\n"]
    # wszystkie bloki kamer zastąpione w miejscu pierwszego, w kolejności z formularza;
    # usunięte bloki znikają (wraz z ich treścią, bez komentarzy sprzed nich)
    keep = []
    first = cam_hdrs[0][1] if cam_hdrs else None
    removed = {i for h in cam_hdrs for i in range(h[1], h[2])}
    for i, ln in enumerate(lines):
        if i == first:
            for cam in cameras:
                keep += blocks[id(cam)]
        if i not in removed:
            keep.append(ln)
    if first is None:
        if keep and keep[-1].strip():
            keep.append("\n")
        for cam in cameras:
            keep += blocks[id(cam)]
    out = "".join(keep)
    return re.sub(r"\n{3,}(?=\[\[camera\]\])", "\n\n", out).rstrip("\n") + "\n"
