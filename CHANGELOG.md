# Changelog — TurboQuant QLauncher

All notable changes to TurboQuant QLauncher are documented here.
The format loosely follows [Keep a Changelog](https://keepachangelog.com/),
and this project uses simple incremental version numbers (`vX.YY`).

License: MIT · © WaveboSF 2026

---

## v1.37

Current release. One source file, `TurboQuant_QLauncher.py`, for Windows
and Linux (platform behaviour is selected at runtime; Linux keeps its own
`*_Linux` config, key, lock and results files).

### Features
- Responsive multi-column model grid with keyboard navigation along the
  grid; up to 3 model directories, optional recursive scan.
- **Other models** section: non-GGUF model folders (Safetensors, PyTorch,
  ONNX) are listed as inventory with size, status and *Open folder*.
- Explicit **Brain** and **Vision** roles. Autoload starts the Brain; the
  Vision model can run as a second server beside it on its own port and
  GPU (`vision_port`, `vision_gpu_key`, `vision_autoload`).
- **LAN** toggle (bind `0.0.0.0` instead of `127.0.0.1`) and optional
  **API key** (`--api-key`), masked in the UI, redacted in the log and
  stored in a separate key file.
- Header **Run** / **Stop** buttons, one **Unload** button per GPU, and
  **💾 Layout** to save window size and position.
- KV and LA options gated by server family (mainline vs. TurboQuant fork,
  detected from the build folder name), `f16-K + turbo4-V` preset and
  LA `auto` state. Bench All and Probe only run valid KV × LA tuples.
- Quick-Switch slot labels derived from dated build folders
  (`llama-server_<fork>_<YYYY-MM-DD>_<build>.<platform>` → `<fork>_<build>`).
- Port-conflict dialog (start on a free port or stop the occupant) and a
  check via `nvidia-smi` that the server landed on the selected GPU.
- CLI: `--autostart`, `--no-autostart`, `--minimized`, `--model` /
  `--gpu`, `--acquire` / `--release` (token-owned model slots),
  `--shutdown`, `--status`, `--version`.

### Fixed
- **Probe ran on the wrong GPU.** The Probe dialog always used `GPU 0`;
  it now uses the highlighted model × GPU row. Q/N/C mode clicks without
  a running server use the highlighted row as well.
- **Stale "Running" status after silent stops.** After a Probe run, a
  Q/N/C restart or a model switch, the status pill and running dot could
  keep showing the stopped server. They are now reset on every stop.
- **Closing with an invalid Port value crashed.** A non-numeric Port field
  raised `ValueError` while saving the config on close; the last valid
  port is now kept.
- **Autoload on a missing GPU.** When the saved GPU no longer exists,
  autoload announced a fallback to the first row but still started on the
  missing GPU; it now really uses the first row.
- *Update Binaries* names the config key `update_binaries_source_path`
  instead of a non-existent Settings dialog.

### Repository
- Source-only developer repository: no pre-built executables or binaries.
  Runtime files (config, API key, lock, benchmark results) are ignored by
  git.
- README rewritten for v1.37. The PDF manual still describes v0.58.

---

## v0.58

### Fixed
- **Autoload selection-rectangle race.** When Autoload restored the last-
  used (model, GPU) pair on startup, the running-indicator dot correctly
  showed the autoloaded cell, but the selection rectangle ended up stuck
  on cell (0, 0). Root cause: the default `_select_cell(0, 0)` queued
  via `after_idle` at the end of card construction fired *after* the
  synchronous autoload `_select_cell(target)` call, clobbering it. Fix
  reroutes `_trigger_autoload_if_eligible()` through `after_idle`, so
  both calls serialise into the same queue and the autoload selection
  wins in the correct order.

### Added
- **User-controlled font zoom.** Global Ctrl+Plus / Ctrl+Minus / Ctrl+0
  keyboard shortcuts scale every font in the UI up or down by 10 %
  steps, clamped to [0.75, 2.0]. Both main-row and numpad `+`/`-` keys
  bound, plus `=` for US/DE layouts where Ctrl+= is what you actually
  press for Ctrl+Plus. Applied on top of (not in place of) the existing
  HiDPI Tk scaling, so 4K setups keep their automatic 2.0× factor.
  Takes effect live via the same theme-rebuild mechanism used for
  dark/light swap — no restart required. Persisted in `cfg["font_scale"]`.
- **New config key `font_scale`** (float, default `1.0`, range
  `[0.75, 2.0]`). Garbage values fall back to 1.0 without crashing.

### Internal
- New constants `FONT_SCALE_MIN = 0.75`, `FONT_SCALE_MAX = 2.00`,
  `FONT_SCALE_STEP = 0.10` alongside the existing `FONT_*` tuples.
- New dict `_FONT_BASE` holding the unscaled base point sizes
  (TITLE=15, SUBTITLE=12, HEADER=11, BODY=11, SMALL=10, DIM=10,
  BRAILLE=12). The module-level `FONT_*` tuples are now derived from
  these by `apply_font_scale()`.
- New function `apply_font_scale(scale: float) -> float` rewrites the
  `FONT_*` module globals in place. 6 pt floor per integer rounding
  step guarantees fonts stay renderable at the minimum zoom level.
  Returns the clamped scale so callers can persist the applied value.
- New method `_on_font_zoom(delta: float)` drives live zoom. Same
  save-rebuild-restore pattern as `_on_toggle_theme`, including the
  Probe / Bench All mid-operation guard.
- Linux-only: `_refresh_fonts(root)` (post-Tk-root font-family upgrade)
  now delegates tuple construction to `apply_font_scale(_FONT_SCALE_CURRENT)`
  so a family upgrade (DejaVu → JetBrains Mono / Fira Code) preserves
  the current user zoom instead of silently resetting sizes to the
  base values. New module-level `_FONT_SCALE_CURRENT` tracks the active
  scale so `_refresh_fonts()` doesn't need launcher-instance access.

---

## v0.48 – v0.57 (summary)

These versions were released without individual changelog entries. The
summary below is reconstructed from the source code.

- **v0.48 – v0.50 — Probe (qnut).** Differential probe of all KV × LA
  combinations against an f16 baseline (three prompts, score 0–3) and
  the Q / N / C mode buttons that apply the stored anchors. Profiles are
  stored per (model, server slot) under `nut_profiles` in the config.
- **v0.51** — `q8_0-K + turbo2-V` ("Boundary V", use with LA=7) added
  to the KV configs and the Bench All matrix (now 7 × 4 × 3 = 84 runs).
- **v0.54** — Vision models: `mmproj-*.gguf` siblings are auto-attached
  via `--mmproj` and hidden from the model list. Windows HiDPI fix
  (DPI awareness + Tk scaling). Footer button groups sized via font
  metrics.
- **v0.55** — Window size adapts to the screen, horizontally scrollable
  settings bar, the Paths dialog no longer sets the active server
  (done by clicking a slot button).
- **v0.56** — Autoload toggle in the footer (unlocked after the first
  successful server start), local IPC socket + lock file, CLI flags
  `--autostart`, `--no-autostart`, `--shutdown`, `--status`,
  `--version`. Blocking "No Thinking" dialog replaced by an inline
  warning.
- **v0.57** — Reasoning-model detection from GGUF metadata
  (architecture and chat template) with filename fallback. Port
  conflict check before starting llama-server.
- Also within this range: live dark/light theme switch from the header.

---

## v0.47

### Added
- **Multi-directory model scan.** The Configure Paths dialog now exposes
  three numbered slots (`#1`, `#2`, `#3`) for LLM model directories. All
  non-empty slots are scanned together and the results are merged into a
  single model list.
- **Recursive scan toggle.** A new global checkbox in the Configure Paths
  dialog controls whether subdirectories of each model path are walked.
  When enabled (default), `os.walk` is used; when disabled, only the top
  level of each directory is scanned via `os.listdir`.
- **Duplicate detection.** Files are deduplicated by canonical realpath,
  so overlapping directories or symlinks no longer list the same file
  twice.

### Changed
- `scan_models()` now accepts either a single path string (for backward
  compatibility) or a list of paths, plus a `recursive: bool` flag.
- Header label above the model list shows all configured directories
  joined with `•`, or `(no directory configured)` when none are set.
- Scan log message now reports the scan mode (`recursive` / `top-level`)
  and the number of directories scanned.

### Migration
- Existing configs with the legacy single-path key `llm_models_path` are
  automatically migrated into `llm_models_paths[0]` on first start. The
  legacy key is removed from the config file. No user action required.

### Internal
- New constant `MAX_MODELS_PATHS = 3`.
- New config keys `llm_models_paths` (list) and `llm_models_recursive`
  (bool, default `True`).

---

## v0.46

### Added
- **Per-run benchmark timeout.** New `Timeout` field in the header (default
  90 seconds) controls how long a single `llama-bench` invocation may run
  before being killed. Replaces the previous hard-coded 90-second limit.
- Timeout value is validated and clamped to `[30, 1800]` seconds. Invalid
  input falls back to 90 with a warning in the log.

### Changed
- Bench All matrices no longer hang indefinitely on a single broken run —
  the per-run timeout now applies to every cell in the matrix.

### Internal
- New config key `bench_timeout` (int, default `90`).

---

## v0.45

### Added
- **Context depth dimension for Bench All.** The benchmark matrix is now
  three-dimensional: `KV × LA × Depth`. Depth is passed to `llama-bench`
  as `-d <n>`, which pre-fills the KV-cache with N tokens before the
  decode test, simulating long-context inference (Madreag methodology).
- Result keys now use the format `{kv}|LA=xx|d=N`.

### Changed
- Outer benchmark loop is now LA (groups runs by LA mode for log
  readability), middle loop is depth, inner loop is KV.
- Context size no longer appears in the benchmark heading — the `Ctx`
  field is for server-start only and is ignored on the benchmark path.

### Fixed
- The broken `-c` path in the benchmark code has been removed. Vanilla
  `llama-bench` has no `-c` / `--ctx-size` switch — every fork tested
  (mainline, gemma4, thetom, madreag) rejects it with
  `error: invalid parameter for argument: -c`. Use `-d <n>` instead.

---

## v0.44

### Added
- **Quick-Switch Slots.** Up to six bookmarked `llama-server.exe` builds
  appear as one-click buttons in the footer next to `Update Binaries`.
  The active slot is highlighted. Configure via the Paths dialog.
- Slot labels are auto-derived from the parent folder name, with the
  common `llama-server_` prefix stripped for readability (e.g.
  `llama-server_thetom_cuda132` → `thetom_cuda132`).
- Engine folder name now disambiguates forks that share the same model
  in benchmark output, so results from `mainline_cuda132` and
  `thetom_cuda132` no longer collide.

### Changed
- Active llama-server slot button is refreshed on every Paths-dialog
  save and after every slot click.

### Internal
- New constant `MAX_SERVER_SLOTS = 6`.
- New config key `server_slots` (list of strings).

---

## v0.43

### Added
- **Reasoning-model awareness.** Known reasoning-model families are
  detected via case-insensitive filename substring match, so the
  launcher can guard against silently disabling thinking mode on
  models that need it.

### Changed
- Footer is now packed *before* the expanding `PanedWindow`, anchored
  to the window bottom edge with `side="bottom"`. Fixes the footer
  occasionally getting clipped on small windows.
- Empty placeholder text replaced with a single space `" "` to keep
  Tk label widgets at a consistent height.
- Loading hint is now shown immediately when the model scan starts,
  not after the first scan result arrives.

---

## v0.42

### Added
- **Ctx field.** New context-window field in the header. Empty = use
  the model's default; any other value is passed to `llama-server` as
  `-c <ctx>` on startup.
- **Run = Start Server.** When no benchmark mode is active, pressing
  Run starts the model server (previously Run only worked in
  benchmark mode).
- Auto-enable Flash Attention whenever K or V cache is compressed (any
  TurboQuant or q8_0 mode). Vanilla f16 stays unchanged.
- Full benchmark environment header is written to `TurboQuant_Benchmark.md`
  on first append (CUDA version, GPU list, build path, etc.).

### Internal
- New config key `ctx_size` (string, default `""`).

---

## v0.41 and earlier

Earlier development focused on:

- Layer-Adaptive modes (off, 1, 5, 7) via the LA buttons, passed to
  `llama-server` as the `TURBO_LAYER_ADAPTIVE` environment variable.
- Hover behaviour fixes for KV / LA buttons and slot buttons.
- Dynamic benchmark output columns (pp512, tg128, deltas) that adapt
  to whichever configurations were actually run.
- Bench All matrix mode (KV × LA, before the v0.45 depth dimension).
- Universal GPU detection: NVIDIA (`nvidia-smi`), AMD (`rocm-smi`),
  Intel (WMI / `lspci`), Apple Silicon (`sysctl`).
- Persistent JSON config for window position, splitter, KV selection,
  port, no-thinking flag, and active llama-server path.
- Idle / Loading / Running status pill, click-to-open-browser when
  running.
- Braille-style VRAM bars per GPU and CPU RAM, KV-aware colour coding
  (green / amber / red).

---

## Notes for contributors

When contributing a change, add a short entry at the top of this file.
Source comments may additionally reference internal work-package tags
(`AP-QL-xx`); this file is the user-facing history.
