v0.54 — Vision-model support, HiDPI fix, refined footer layout

## Vision-model support (Windows + Linux)

- Auto-attaches `mmproj-*.gguf` sibling as `--mmproj` argument when
  launching a VL model. Zero config — drop the pair of GGUF files
  into a model directory and llama-server is wired up for both.
- F16 / BF16 projector variants are preferred over quantized ones
  when multiple exist side by side (projector size is negligible
  against main-weight savings, and encoder quality matters).
- Scanner filter updated: `mmproj` added to `_FILENAME_JUNK_HINTS`
  so companion projector files (typically ~1 GB) no longer appear
  as standalone models in the UI. Without this, a user could
  accidentally select the projector as a main model and crash
  llama-server.
- Text-only models are completely unaffected: the mmproj scan is a
  no-op when no projector file is present in the model directory.

## HiDPI fix (Windows)

- `SetProcessDpiAwareness(1)` is declared in `main()` before any Tk
  window is instantiated. Without this, Windows applied DWM bitmap
  upscaling on HiDPI displays, which bilinearly stretched the entire
  window — text appeared blurry (German: "unscharf"). Declaring
  System-DPI-Aware disables the upscale, so text renders at native
  resolution. Fallback to legacy `user32.SetProcessDPIAware()` for
  pre-Windows-8.1.
- Tk's own scaling factor is then applied in `_configure_theme()`
  (2.0x ≥ 4K, 1.5x ≥ QHD) to compensate — otherwise widgets sized
  for 96 DPI would appear tiny on a 4K monitor. Mirrors the logic
  already present in the Linux build.

## Refined footer layout (Windows + Linux)

- Three independent button groups, each with its own uniform width
  computed via `tkfont.measure()` against the widest label in that
  group:
  - Group A: `Rescan Models` + `Update Binaries`
  - Group B: engine slot buttons (`madreag_cuda132`, etc.)
  - Group C: `Paths` + `About`
  - `Save Results...` remains standalone.
- Previous hard-coded pixel widths (130 / 80 / 168 on Windows;
  160 / 100 / 200 on Linux) are gone. `tkfont.measure()` handles
  font-metric differences (Consolas vs DejaVu Sans Mono) and Tk's
  HiDPI scaling factor automatically.
- Horizontal padding reduced from 40 px to 20 px (≈ 1 char per side
  at mono 10).
- Slot-button inter-spacing raised from `padx=2` to `padx=6`, so
  engine names have room to breathe.
- Footer font switched from `FONT_SMALL_B` (bold) to `FONT_SMALL`
  (regular) for a lighter, more filigree appearance.

## File changes

- `TurboQuant_QLauncher.py` (Windows)
- `TurboQuant_QLauncher_Linux.py` (Linux)
