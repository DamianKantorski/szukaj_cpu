# szukaj_cpu

Person detection on live RTSP CCTV streams for small home servers without a dedicated GPU.
A YOLO model runs on the CPU through OpenVINO, video is decoded on the Intel iGPU (VAAPI),
and every detection becomes an event with a thumbnail, a full-resolution frame, a clip cut
from the main stream, an entry in a browsable HTML gallery, a push notification (ntfy) and
a marker on the Shinobi CCTV timeline.

Developed and tested on Debian 13 (OpenMediaVault 8) with an Intel Core i5-7500 (HD 630),
two cameras (Dahua, Tiandy) and Shinobi.

## How it works

```
camera ──► ffmpeg #1: main stream, stream copy ──► RAM segment buffer (/dev/shm)
   │                                               └─► clips, full-resolution frames
   └─────► ffmpeg #2: substream, keyframes only, VAAPI decode ──► YOLO (OpenVINO, CPU)
                                                                   └─► events
events ──► gallery (index.html, wyniki.csv, thumbnails, clips, full frames)
       ──► ntfy notification with photo   ──► Shinobi timeline event (API /motion)
```

* Detection runs on the camera **substream** (1 s keyframe interval, Smart Codec off), so the
  recorded **main stream** can keep Smart Codec / a long GOP and small recordings.
* Substream frames are stretched to the main-stream aspect ratio (e.g. 704×576 → 1024×576).
* The OpenVINO model is exported in 16:9 (e.g. 960×544), so no time is spent on padding.
* Inference uses OpenVINO directly (no PyTorch/Ultralytics at runtime): ~175–195 ms per frame
  with 4 threads on an i5-7500, two cameras at one frame per second each.
* After each event the best moment is re-analysed on the main stream and saved in full
  resolution (JPEG XL, JPEG or WebP).
* Gallery entries, thumbnails, clips and full frames older than `retention_days` are deleted.

## Layout

| Path | Content |
|---|---|
| `/opt/szukaj_cpu/` | program (`run.py`, package `szukaj_cpu/`), Python venv, config template, tools, ntfy setup |
| `/opt/szukaj_cpu/models/` | bundled OpenVINO models: FP32 and INT8 (full package) |
| `/opt/szukaj_cpu/wheels/` | Python wheels for the offline install (full package) |
| `/opt/szukaj_cpu/ffmpeg`, `ffprobe` | static ffmpeg built with the package (full package) |
| `/usr/bin/szukaj-cpu` | launcher (`--init`, `--test`, `--version`) |
| `/usr/bin/szukaj-cpu-export-tools` | optional model-export tools (Ultralytics + PyTorch + NNCF) |
| `/usr/lib/systemd/system/szukaj_cpu@.service` | service, one instance per user |
| `~/.szukaj_cpu/szukaj_cpu.toml` | configuration (cameras, paths, ntfy, Shinobi) |
| `~/.szukaj_cpu/models/` | your own models (uploaded in the panel or exported); used before bundled ones |
| `~/.szukaj_cpu/ffmpeg`, `ffprobe` | optional own static ffmpeg (used before the bundled one) |

### Source layout

```
src/run.py                 launcher (also /opt/szukaj_cpu/run.py)
src/szukaj_cpu/
  common.py    constants, logger, STOP flag, helpers
  config.py    reading and checking szukaj_cpu.toml
  camera.py    ffmpeg commands, frame reader, segment buffer, watchdog
  detector.py  YOLO in OpenVINO (FP32/INT8) or PyTorch, model lookup, zone/crop, drawing
  export.py    YOLO -> OpenVINO export (FP32, INT8 via NNCF)
  events.py    events, parked-vehicle filter
  gallery.py   HTML gallery, CSV, retention
  clips.py     clips and full-resolution frames from the main stream
  notify.py    ntfy and Shinobi
  main.py      arguments, main loop, --test, --init
  web/         web server, detection-area page (area.html), configurator (config.html),
               TOML editing, model upload
tools/         build_ffmpeg.sh, export_models.py, compare_models.py, test_przerwy.py
packaging/     build-deb.sh, debian/ maintainer scripts
```

## Installation

### Building the package

The full package contains everything that used to be downloaded during installation, so the
server needs no internet access to install it:

* Python wheels (OpenVINO, OpenCV, NumPy) for Debian 13's Python – installed offline,
* YOLO11s OpenVINO models, **FP32 and INT8**, for `MODEL_SIZES` (default `960x544`),
* `ffmpeg`/`ffprobe` compiled on the build machine (`tools/build_ffmpeg.sh`, `-march=skylake`).

Everything is built in a Debian 13 container, so the build machine only needs Docker or
Podman (e.g. CachyOS: `sudo pacman -S podman`; WSL/Ubuntu: `sudo apt install podman`):

```bash
bash packaging/build-deb.sh                     # → dist/szukaj-cpu_1.2.0_amd64.deb
bash packaging/build-deb.sh --calib ~/frames    # INT8 calibrated on frames from your cameras
MODEL_SIZES="960x544 640x384" bash packaging/build-deb.sh
bash packaging/build-deb.sh --slim              # program only; install downloads dependencies
```

Intermediate results are cached in `build/` – the first build takes a while (ffmpeg, PyTorch
for the export, quantization), later builds take a minute. `--rebuild-ffmpeg`,
`--rebuild-models`, `--rebuild-wheels` and `--rebuild-all` refresh parts of it. Pushing a
`vX.Y.Z` tag builds the same full package on GitHub Actions and attaches it to the release.

### Installing

```bash
sudo apt install ./szukaj-cpu_1.2.0_amd64.deb
szukaj-cpu --init                       # creates ~/.szukaj_cpu/szukaj_cpu.toml
nano ~/.szukaj_cpu/szukaj_cpu.toml      # cameras, gallery path, Shinobi, ntfy (or use the web panel)
sudo usermod -aG render,video $USER     # iGPU access (log in again)
szukaj-cpu --test                       # checks cameras, model and Shinobi API
sudo systemctl enable --now szukaj_cpu@$USER
journalctl -u szukaj_cpu@$USER -f
```

If the server's Python does not match the bundled wheels (other than Debian 13), the
install falls back to downloading them from PyPI.

### Migrating from the old `install.sh` setup (`~/szukaj_cpu`)

```bash
bash uninstall.sh          # stops szukaj_cpu.service, moves config, models and static
                           # ffmpeg to ~/.szukaj_cpu, removes ~/szukaj_cpu
sudo apt install ./szukaj-cpu_1.2.0_amd64.deb
sudo systemctl enable --now szukaj_cpu@$USER
```

`uninstall.sh --dry-run` shows what would be done; `--package` removes the .deb,
`--purge` also deletes `~/.szukaj_cpu`.

## Configuration

All options are documented in [`config/szukaj_cpu.toml`](config/szukaj_cpu.toml). Most important:

```toml
[detection]
imgsz = 960                  # model input width; height follows aspect_ratio (16:9 → 544)
threads = 4                  # OpenVINO threads per inference
frame_selection = "keyframes"

[[camera]]
name = "brama"
url = "rtsp://USER:PASS@192.168.1.10:554/cam/realmonitor?channel=1&subtype=0"           # recorded
analysis_url = "rtsp://USER:PASS@192.168.1.10:554/cam/realmonitor?channel=1&subtype=1"  # detection
shinobi_monitor = "MONITOR_ID"
```

Camera settings: in the **substream** set the I-frame interval to the frame rate (1 s) and
turn Smart Codec / H.264+ off. The main stream can stay as it is.

The configuration contains camera passwords and API keys – it is created with mode `600`.

### Parked cars

A car parked in view is detected in every frame; at night its confidence hovers around the
threshold, so it used to start a new event each time it reappeared. Objects of
`static_classes` (cars, trucks, … – not people) that stay in the same place for
`static_after` seconds are now remembered as stationary and ignored there; arrival,
departure and movement still raise events. The list survives restarts
(`~/.szukaj_cpu/nieruchome.json`).

### Detection area: `zone` and `crop`

* `zone = [[x, y], …]` – polygon (fractions 0–1 of the frame); a person counts only when the
  bottom-centre of the box (their feet) lies inside it. The whole frame is still analysed.
* `crop = [x1, y1, x2, y2]` – rectangle; only this part of the frame goes to the model.
  Smaller area = people appear larger to the model, so distant figures are detected better.
  `[]` = whole frame.

## Models: FP32 and INT8

`[detection] precision` selects the model version:

* `"fp32"` – the standard export, most accurate;
* `"int8"` – quantized with OpenVINO NNCF (post-training, calibrated on images). It runs
  faster on the CPU at a small accuracy cost. On CPUs without VNNI (e.g. i5-7500) the gain
  is smaller than on new ones – measure it on the server (below).

Models are looked up in the directory of `[paths] model` (`~/.szukaj_cpu/models`, your own)
and then in `/opt/szukaj_cpu/models` (bundled). If the INT8 model for the configured input
size is missing, FP32 is used and a warning is logged. If no model exists at all, the
service exports it (needs `sudo szukaj-cpu-export-tools install`).

Switching in the web panel: **Configuration → Model** – version selector, list of available
models with **Use**, upload of a model `.zip` (model update) and removal of own models;
then save and restart the service.

Own models / calibration on your cameras (on the build machine or the server with the export
tools):

```bash
python3 tools/export_models.py --sizes 960x544 --calib ~/frames --zip --out ./models
# → models/yolo11s_960x544_int8_openvino_model.zip – upload it in the panel
```

Comparing speed and agreement of both versions on the server, on frames from the gallery:

```bash
sudo systemctl stop szukaj_cpu@$USER
/opt/szukaj_cpu/venv/bin/python /opt/szukaj_cpu/tools/compare_models.py \
    /opt/szukaj_cpu/models/yolo11s_960x544_openvino_model \
    /opt/szukaj_cpu/models/yolo11s_960x544_int8_openvino_model  /path/to/gallery/miniatury
sudo systemctl start szukaj_cpu@$USER
```

## Web panel

The gallery server (`http_port` in `[gallery]`) can also edit `zone` and `crop` in the
browser: choose a camera, click the polygon corners or drag the crop rectangle on a live
snapshot, then **Save and apply** – the values are written to `szukaj_cpu.toml` (a `.bak`
copy is kept, file mode preserved) and take effect immediately, without a restart.
**Restart service** restarts the process (systemd brings it back via `Restart=always`).

### Configuration page (`/admin/config`)

Everything else in `szukaj_cpu.toml` can be edited in the browser, grouped in a side menu:
cameras, detection, parked vehicles, clips, gallery and web panel, notifications, Shinobi,
paths and tools.

* **Cameras** – add, rename, disable or remove cameras; main (recorded) and analysis
  (substream) RTSP addresses with a **Test** button (ffprobe: codec, resolution, frame rate
  and keyframe interval, with a warning when the analysis stream has keyframes too rarely
  for `frame_selection = "keyframes"`) and an **RTSP address wizard** (Dahua/Amcrest,
  Hikvision/Tiandy, Reolink, Uniview: IP, port, channel, user, password → both URLs).
* Passwords and keys are never sent to the browser: camera passwords show as `********`
  and are kept unless the user or the IP address changes; secret fields left empty keep
  their value.
* **Save** shows the list of changes and validates the whole file exactly as at service
  start; only changed keys are rewritten, comments and layout stay, the previous file is
  kept as `.bak`. **Save and restart** applies the changes right away. If the file was
  changed meanwhile (e.g. by the detection-area page) saving is refused until reload.

The panel is enabled only when a login is set; the whole site (gallery included) then
requires HTTP Basic authentication:

```toml
[gallery]
http_port = 8091
http_user = "damian"
http_password = "long random password"
```

Open `http://SERVER:8091/admin/`. Basic auth over plain HTTP is fine in the home network;
from outside use Tailscale, e.g. `sudo tailscale serve --bg --https=8443 http://localhost:8091`
→ `https://SERVER.TAILNET.ts.net:8443/admin/`. Do not expose the port to the internet.

## Local ntfy server

The public ntfy.sh keeps messages and attachments only for a limited time, so a notification
read late may come without its photo. `ntfy/setup-ntfy.sh` installs a local
[ntfy](https://ntfy.sh) server that keeps messages and photos for 5 days (like the gallery),
with access control:

```bash
sudo bash /opt/szukaj_cpu/ntfy/setup-ntfy.sh --config ~/.szukaj_cpu/szukaj_cpu.toml
sudo systemctl restart szukaj_cpu@$USER
```

It installs ntfy from the project's apt repository (archive.ntfy.sh, falling back to the Debian package), writes `/etc/ntfy/server.yml` (port 8092),
creates a write-only user `szukaj_cpu` with a token (written to the config) and a read-only
user for the phone. In the ntfy app add the server, log in and subscribe to the topic
(`szukaj` by default).

* **iOS:** the config sets `upstream-base-url: https://ntfy.sh`, which is required for instant
  notifications from a self-hosted server; only a "poll this topic" signal goes through ntfy.sh,
  the message and photo are fetched from your server.
* **Outside the home network** the phone must reach `base-url`, because photos are downloaded
  from it. If Tailscale is logged in on the server, the script uses its
  `https://<server>.<tailnet>.ts.net` address and enables `tailscale serve` for ntfy
  (tailnet only); on the iPhone set Tailscale *VPN On Demand* to *Always*. Without Tailscale
  it falls back to the LAN address and warns. The address is stored in every message when it
  is sent, so older notifications keep the previous one.
* `ntfy/docker-compose.yml` is an alternative for running ntfy in Docker.

## Static ffmpeg

`tools/build_ffmpeg.sh` builds the latest stable FFmpeg with libjxl, optimised for
`-march=skylake` (Skylake/Kaby Lake), containing only what szukaj_cpu uses, in a Debian 13
container (so it runs on the server's glibc). The full package build runs it automatically
and installs the result as `/opt/szukaj_cpu/ffmpeg`. With `ffmpeg = "auto"` the order is
`~/.szukaj_cpu/ffmpeg` (own copy), `/opt/szukaj_cpu/ffmpeg` (bundled), then the system one –
delete an old own copy to use the bundled build. Stand-alone:

```bash
bash tools/build_ffmpeg.sh                      # → ./ffmpeg-szukaj/
```

## Tools

* `tools/export_models.py` – YOLO → OpenVINO export, FP32 and INT8 (`--calib`, `--zip`).
* `tools/compare_models.py` – speed and detection agreement of two models on a folder of images.
* `tools/test_przerwy.py` – measures raw OpenVINO inference time (continuous vs. with pauses,
  2 and 4 threads) to tell model speed apart from service overhead.

## Removal

```bash
bash /opt/szukaj_cpu/uninstall.sh --package         # keeps ~/.szukaj_cpu
bash /opt/szukaj_cpu/uninstall.sh --package --purge # also deletes ~/.szukaj_cpu
```

## Notes

* YOLO11 weights come from Ultralytics and are licensed under AGPL-3.0. They are not part of
  this repository; the full package contains models exported from them (see
  `/usr/share/doc/szukaj-cpu/MODELS.txt`).
* Logs and comments in the program are in Polish.
