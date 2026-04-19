# Changelog — TurboQuant QLauncher

All notable changes to TurboQuant QLauncher are documented here.
The format loosely follows [Keep a Changelog](https://keepachangelog.com/),
and this project uses simple incremental version numbers (`v0.XX`).

License: MIT · © WaveboSF 2026

---

## v0.56

### Added
- **Autoload toggle** (footer button). Single master switch for
  automatic model loading + CLI remote control. Three visual states,
  all rendered with the constant label "Autoload" so the button width
  never jitters:
  - 🔒 locked (grey, disabled) — before the installation is verified.
  - OFF — unlocked but opted out.
  - ✓ ON (green) — next launcher start auto-loads the last-used
    `(model, GPU)` pair, and CLI remote control is live.
- **CLI remote control** for external tools (e.g. MyIDE-style
  integration), available only when Autoload is ON:
  - `--autostart` / `-a` — start the last-used model immediately.
    Forwards via IPC to an already-running instance instead of opening
    a second window.
  - `--shutdown` / `-q` — close a running instance cleanly (no dialog,
    no confirmation). Idempotent: exits `0` even when nothing is running.
  - `--status` — print a JSON status report from the running instance.
  - `--no-autostart` — force-skip autoload even if the toggle is ON.
  - `--version`.
- **Local IPC listener** on `127.0.0.1:<random free port>` when
  Autoload is ON. Lock file `TurboQuant_QLauncher.lock` (JSON: `pid`,
  `port`, `started`, `version`) lives next to the config. Protocol is
  three line-based commands: `SHUTDOWN`, `AUTOSTART [model [gpu]]`,
  `STATUS`.
- **Single-instance forwarding**. A second `--autostart` call while
  the launcher is already running sends the command over IPC and exits
  instead of opening a second window.
- **Inline "reasoning model" warning** next to the No Thinking
  checkbox. A yellow `⚠ reasoning model` label appears whenever No
  Thinking is checked AND the selected model is reasoning-capable
  (Gemma 4 / Qwen3 / DeepSeek-R1 / QwQ). Tooltip carries the full
  accuracy-drop explanation. Updates on checkbox toggle and on
  model-row selection changes.

### Changed
- Server-output parser flips `cfg["install_verified"]` to `True` on
  the first `listening` line. One-way gate that unlocks the Autoload
  button.
- `_start_server` records the `(model, GPU)` pair as `cfg["last_model"]`
  / `cfg["last_gpu_key"]` right after `Popen()` succeeds, so autoload
  has a target even if the user kills the launcher mid-load.
- `_on_close` tears down the IPC listener before destroying the Tk
  root, so in-flight `AUTOSTART` calls can't schedule main-thread work
  on a dying interpreter.

### Removed
- **Blocking "Reasoning Model — Thinking disabled" messagebox** in
  `_start_server`. It popped up before every single server start of a
  reasoning model with No Thinking on — tedious when switching models
  back-to-back. Replaced by the inline warning label next to the
  checkbox plus a single log-line note at server start. The Probe
  run's own one-shot reasoning-model dialog is unaffected (it still
  fires before each Probe run since it describes a different, Probe-
  specific trade-off).

### Migration
- The intermediate two-gate design (a "Safe Settings" footer button +
  a separate "Autoload" checkbox in the settings bar) was merged into
  a single footer toggle. Configs that had `safe_settings=True` are
  auto-migrated to `autoload=True` on first load; the legacy key is
  dropped.
- Forward-compatible with v0.55: four new keys (`install_verified`,
  `autoload`, `last_model`, `last_gpu_key`) default to `False` / `""`,
  so an upgraded config behaves identically until the user opts in.
- v0.55 can still read v0.56 configs (unknown keys are preserved).

### Internal
- New imports: `socket`, `argparse`. New constant: `LOCK_FILE`.
- New helpers: `_pid_is_alive`, `read_lock_file`, `send_ipc_command`.
- New `TurboQuantQLauncher.__init__` parameter: `autostart_override`
  (tri-state: `None` / `True` / `False`).
- Stale lock files (dead PID, corrupt JSON) are purged silently on
  next start.
- Removed dead attribute `_probe_thinking_warning_acknowledged` — was
  only needed as a bypass for the now-removed per-tuple reasoning-model
  messagebox.

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

Inline `# vX.YY:` annotations have been removed from the source for
clarity. This file is now the canonical history. When contributing a
change, please add a new section at the top of this file rather than
annotating the source.
