#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TurboQuant QLauncher                            (c) WaveboSF 2026
=================================================================
Model Switcher & Server Manager for llama-server with TurboQuant KV-Cache.

Standalone GUI with zero external dependencies.
Uses only Python stdlib (tkinter/ttk) — runs anywhere Python runs.

Features:
- Auto-scan GGUF models from one or more configurable directories
- Per-model GPU selector with VRAM fit indicator (Braille bars)
- One-click model switching (auto-stops previous server)
- KV-Cache config: f16, q8_0+turbo4, turbo3+turbo3, turbo4+turbo4, etc.
- NVIDIA + AMD + Intel + Apple GPU detection
- Quick-switch slot bookmarks for multiple llama-server.exe builds
- Bench All matrix mode (KV × LA × Depth) with configurable timeout
- Vision model support: auto-attaches mmproj-*.gguf next to VL models (v0.54)
- Server log output with timestamps
- Persistent JSON config

Usage:
    python TurboQuant_QLauncher.py

(c) 2026 — MIT License
"""

import os
import sys
import json
import glob
import time
import signal
import socket
import struct
import argparse
import platform
import subprocess
import threading
import re
import urllib.request
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: Constants
# ═══════════════════════════════════════════════════════════════════════════════

APP_VERSION = "0.58"

def get_launcher_dir() -> Path:
    """Return the directory where the launcher .py or compiled .exe lives.

    Works correctly for three execution modes:
      1. python TurboQuant_QLauncher.py   -> directory of the .py file
      2. Nuitka --standalone              -> directory of the .exe
      3. Nuitka --onefile                 -> directory of the .exe
                                            (NOT the temp extraction dir)

    Per the official Nuitka manual (Onefile: Finding files), the only
    reliable way to locate the user-facing .exe in --onefile mode is
    ``__compiled__.containing_dir`` (Nuitka-specific, also works for
    --standalone). Fallback is ``sys.argv[0]``, which Nuitka also sets
    to the original executable path in onefile mode.
    """
    # Nuitka (standalone or onefile): __compiled__ is injected by Nuitka.
    # .containing_dir is the directory of the user-facing .exe.
    compiled = globals().get("__compiled__")
    if compiled is not None and hasattr(compiled, "containing_dir"):
        return Path(compiled.containing_dir).resolve()
    # Nuitka fallback / PyInstaller: sys.argv[0] holds the original exe path
    if getattr(sys, "frozen", False) or compiled is not None:
        argv0 = sys.argv[0] if sys.argv and sys.argv[0] else sys.executable
        return Path(argv0).resolve().parent
    # Plain Python interpreter
    return Path(__file__).resolve().parent

LAUNCHER_DIR = get_launcher_dir()
CONFIG_FILE  = LAUNCHER_DIR / "TurboQuant_QLauncher.json"
BENCH_FILE   = LAUNCHER_DIR / "TurboQuant_Benchmark_results.md"

# v0.56 — Single-instance lock + IPC for CLI control (MyIDE integration).
#
# GATING: The IPC listener + lock file are ONLY created when the user
# has explicitly enabled "Autoload" in the footer. Autoload itself
# unlocks only after the launcher has proven it can actually start a
# llama-server on this machine (first "listening" line in the server
# log flips cfg["install_verified"] permanently to True). Rationale: a
# freshly installed / misconfigured launcher should never silently
# open a local listening socket or spawn llama-server unattended at
# startup. The user has to earn the "autopilot" state by completing
# one successful manual run first, then explicitly opt in by clicking
# the Autoload button.
#
# When enabled:
#   - Lock file contains {"pid", "port", "started"} as JSON.
#   - IPC server binds 127.0.0.1:<port>, accepts "SHUTDOWN" / "AUTOSTART"
#     / "STATUS" commands. CLI calls like `--shutdown` read the lock
#     file, connect to the control port, send the command, and exit.
#     The running instance performs the action on its main thread
#     (clean teardown + save config, no confirm dialog).
#   - On the next launcher start, _trigger_autoload_if_eligible()
#     fires the last-used (model, GPU) pair as soon as the initial
#     scan completes.
LOCK_FILE    = LAUNCHER_DIR / "TurboQuant_QLauncher.lock"

KV_CACHE_OPTIONS = {
    "f16 (default)":       {"ctk": None,     "ctv": None},
    "q8_0-K + turbo4-V":   {"ctk": "q8_0",   "ctv": "turbo4"},
    "turbo3 / turbo3":     {"ctk": "turbo3",  "ctv": "turbo3"},
    "turbo4 / turbo4":     {"ctk": "turbo4",  "ctv": "turbo4"},
    "q8_0-K + turbo3-V":   {"ctk": "q8_0",   "ctv": "turbo3"},
    "q8_0-K + turbo2-V":   {"ctk": "q8_0",   "ctv": "turbo2"},
    "q8_0 / q8_0":         {"ctk": "q8_0",   "ctv": "q8_0"},
}

# Context depth list for the Bench All matrix.
# Each depth value is passed to llama-bench as `-d <n>`, which pre-fills the
# KV-cache with N tokens before running the tg (decode) test. This reproduces
# Madreag's and TheTom's published benchmark methodology from Discussion #20969
# and directly exposes TurboQuant's long-context decode-speed advantage.
#
# With 7 KV × 4 LA × 3 depths = 84 runs per "Bench All" pass (v0.51: added
# the q8_0-K + turbo2-V Boundary V config from TheTom's 08.04.2026 update).
# Edit here to add/remove depths. Keep 0 as the first entry (short-context
# baseline) — it stays comparable to earlier benchmark entries in the log.
BENCH_DEPTH_LIST = [0, 8192, 32768]

# KV-Cache compression factor relative to f16 (1.0 = no compression).
# K and V keys are compressed independently; factor = mean(K_ratio, V_ratio).
#   f16  = 1.0,  q8_0 = 0.5,  turbo4 = 0.25,  turbo3 = 0.20,  turbo2 = 0.156
# Note: q8_0-K + turbo2-V is the "Boundary V" config from TheTom's
# 08.04.2026 update — first 2 + last 2 layers protected at q8_0-V, rest
# at turbo2-V. Activated via TURBO_LAYER_ADAPTIVE=7 (select LA=7 in the
# launcher). Without LA=7 symmetric turbo2-V catastrophically degrades
# PPL, so the Probe scoring system will correctly mark LA=0 combinations
# for this KV as rotten.
KV_COMPRESSION: Dict[str, float] = {
    "f16 (default)":     1.000,   # no compression
    "q8_0 / q8_0":       0.500,   # both halved
    "q8_0-K + turbo4-V": 0.375,   # mean(0.5, 0.25)
    "q8_0-K + turbo3-V": 0.350,   # mean(0.5, 0.20)
    "q8_0-K + turbo2-V": 0.328,   # mean(0.5, 0.156) — Boundary V via LA=7
    "turbo4 / turbo4":   0.250,   # 4× compression
    "turbo3 / turbo3":   0.200,   # 5× compression
}
# KV-cache at f16 adds roughly 25% of model size (8K context, typical arch).
_KV_BASE_OVERHEAD = 0.25

def kv_effective_gb(model_gb: float, kv_name: str) -> float:
    """Return effective VRAM footprint: model weights + compressed KV overhead."""
    factor = KV_COMPRESSION.get(kv_name, 1.0)
    return model_gb * (1.0 + _KV_BASE_OVERHEAD * factor)

# Known reasoning-model families. Filename substring match (case
# insensitive). When the "No Thinking" checkbox is active and a model whose
# filename contains one of these substrings is started, the launcher warns
# the user before launch. Gemma 4 26B-A4B math accuracy dropped from ~97%
# to ~64% without thinking in our Discussion #20969 runs — hence this guard.
# To extend: add lowercase substrings that reliably appear in the GGUF
# filename of the affected family.
REASONING_MODEL_HINTS = (
    "gemma-4",
    "gemma4",
    "qwen3",
    "qwq",
    "deepseek-r1",
    "deepseek_r1",
    "magistral",
    "-think",
    "reasoning",
)

def _is_reasoning_model(filename: str) -> bool:
    """Return True if the GGUF filename looks like a reasoning-capable model."""
    if not filename:
        return False
    low = filename.lower()
    return any(h in low for h in REASONING_MODEL_HINTS)

MAX_SERVER_SLOTS = 6  # up to 6 quick-switch bookmark slots for llama-server.exe paths
MAX_MODELS_PATHS = 3  # up to 3 LLM model directories scanned together

def slot_label_from_path(server_exe_path: str) -> str:
    """Derive a short button label from a llama-server.exe path.

    Strip the common 'llama-server_' prefix from the parent folder name
    so buttons stay compact. Examples:
      G:\\...\\llama-server_thetom_cuda132\\llama-server.exe → 'thetom_cuda132'
      G:\\...\\llama-server_gemma4_cuda132\\llama-server.exe → 'gemma4_cuda132'
      G:\\...\\my_custom_build\\llama-server.exe            → 'my_custom_build'
    Returns empty string for empty/invalid input.
    """
    if not server_exe_path:
        return ""
    try:
        folder = os.path.basename(os.path.dirname(server_exe_path))
    except Exception:
        return ""
    if not folder:
        return ""
    # Strip the "llama-server_" prefix if present (common pattern in Silvestar's setup)
    low = folder.lower()
    if low.startswith("llama-server_"):
        return folder[len("llama-server_"):]
    return folder

def slot_folder_name(server_exe_path: str) -> str:
    """Return just the parent folder name (no prefix stripping).

    Used for benchmark log entries where we want the full, unambiguous
    folder name — e.g. 'llama-server_thetom_cuda132' instead of 'thetom_cuda132'.
    """
    if not server_exe_path:
        return ""
    try:
        return os.path.basename(os.path.dirname(server_exe_path))
    except Exception:
        return ""

DEFAULT_CONFIG = {
    # Up to MAX_MODELS_PATHS LLM model directories. Empty strings = unused
    # slots. The legacy single-path key "llm_models_path" is migrated into
    # llm_models_paths[0] by load_config() for backward compatibility.
    "llm_models_paths": ["" for _ in range(MAX_MODELS_PATHS)],
    "llm_models_recursive": True,  # scan subdirectories of model paths
    "llama_server_path": "",
    # Up to MAX_SERVER_SLOTS quick-switch bookmark paths to llama-server.exe
    # builds. Empty strings = unused slots. Rendered as buttons in the footer next
    # to "Update Binaries" for one-click switching between forks (mainline/TheTom/
    # Gemma4/spiritbuun/etc.).
    "server_slots": ["" for _ in range(MAX_SERVER_SLOTS)],
    "kv_cache": "q8_0-K + turbo4-V",
    "port": 8080,
    "ctx_size": "",  # empty = use llama-server default, else passed as -c <ctx>
    "bench_timeout": 90,  # per-run timeout for llama-bench (seconds), see Timeout field
    "no_thinking": False,
    "benchmark": False,
    "bench_all": False,
    "layer_adaptive": 0,
    "window_x": None,
    "window_y": None,
    "window_w": 1000,
    "window_h": 800,
    "sash_pos": None,
    # Theme override. None = follow system (auto-detect via
    # detect_system_dark_mode), True = force light, False = force dark.
    # Toggled via the ☀/🌙 checkbox in the header. Takes effect on next
    # launch (widget theming is frozen at __init__ time).
    "light_mode": None,
    # v0.56 — Autoload + CLI control gating. Single user-facing flag:
    # cfg["autoload"] is the master switch. When True, the launcher
    # autoloads the last-used (model, GPU) pair on next start AND
    # accepts CLI remote control (--autostart / --shutdown / --status)
    # via a local IPC socket. When False, none of that happens.
    #
    # install_verified is the silent prerequisite: the Autoload button
    # is locked until the launcher has proven it can actually start a
    # llama-server on this machine (first "listening" line in the
    # server log flips install_verified permanently to True).
    #
    # last_model / last_gpu_key record the most recent (model, GPU)
    # combination that was successfully passed to _start_server, so
    # autoload + `--autostart` know what to launch.
    "install_verified": False,
    "autoload": False,
    "last_model": "",
    "last_gpu_key": "",
    # v0.58 — User-controlled font zoom. Applied on top of the HiDPI
    # Tk scaling (see _configure_theme). Keyboard shortcuts:
    #   Ctrl++ / Ctrl+=   → zoom in  (step +0.1)
    #   Ctrl+-            → zoom out (step -0.1)
    #   Ctrl+0            → reset to 1.0
    # Clamped to [0.75, 2.0]. Takes effect immediately via the same
    # theme-rebuild mechanism used for dark/light swap.
    "font_scale": 1.0,
}

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: Theme Detection & Colors
# ═══════════════════════════════════════════════════════════════════════════════

def detect_system_dark_mode() -> bool:
    system = platform.system()
    if system == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\Themes\Personalize"
            )
            value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            winreg.CloseKey(key)
            return value == 0
        except Exception:
            return False
    elif system == "Darwin":
        try:
            result = subprocess.run(["defaults", "read", "-g", "AppleInterfaceStyle"],
                                    capture_output=True, text=True, timeout=3)
            return "dark" in result.stdout.lower()
        except Exception:
            return False
    elif system == "Linux":
        try:
            result = subprocess.run(
                ["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
                capture_output=True, text=True, timeout=3)
            if "dark" in result.stdout.lower():
                return True
        except Exception:
            pass
        try:
            kde_path = os.path.expanduser("~/.config/kdeglobals")
            if os.path.exists(kde_path):
                with open(kde_path) as f:
                    if "Dark" in f.read():
                        return True
        except Exception:
            pass
    return False

@dataclass
class ThemeColors:
    bg: str; bg_secondary: str; bg_header: str
    fg: str; fg_secondary: str; fg_dim: str
    accent: str; green: str; yellow: str; red: str
    border: str; button_bg: str; button_fg: str
    entry_bg: str; select_bg: str

DARK_THEME = ThemeColors(
    bg="#1a1d23", bg_secondary="#1e2028", bg_header="#12151a",
    fg="#e2e8f0", fg_secondary="#94a3b8", fg_dim="#64748b",
    accent="#3b82f6", green="#4ade80", yellow="#fbbf24", red="#f87171",
    border="#2a2d35", button_bg="#2563eb", button_fg="#ffffff",
    entry_bg="#161920", select_bg="#1e3a5f",
)
LIGHT_THEME = ThemeColors(
    bg="#ffffff", bg_secondary="#f1f5f9", bg_header="#e2e8f0",
    fg="#1e293b", fg_secondary="#475569", fg_dim="#64748b",
    accent="#2563eb", green="#15803d", yellow="#a16207", red="#b91c1c",
    border="#cbd5e1", button_bg="#2563eb", button_fg="#ffffff",
    entry_bg="#f1f5f9", select_bg="#dbeafe",
)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: Font Configuration
# ═══════════════════════════════════════════════════════════════════════════════

_MONO = "Consolas" if sys.platform == "win32" else "DejaVu Sans Mono"

# v0.58 — Font-size tuples are derived from these base sizes via
# apply_font_scale(). The module-level FONT_* constants below are what
# every widget references (by re-reading the tuple at build time), so
# re-running apply_font_scale() + a theme rebuild propagates the new
# sizes to every widget on screen.
_FONT_BASE = {
    "TITLE":    15,
    "SUBTITLE": 12,
    "HEADER":   11,
    "BODY":     11,
    "SMALL":    10,
    "DIM":      10,
    "BRAILLE":  12,
}

FONT_TITLE    = (_MONO, _FONT_BASE["TITLE"], "bold")
FONT_SUBTITLE = (_MONO, _FONT_BASE["SUBTITLE"])
FONT_HEADER   = (_MONO, _FONT_BASE["HEADER"], "bold")
FONT_BODY     = (_MONO, _FONT_BASE["BODY"])
FONT_BODY_B   = (_MONO, _FONT_BASE["BODY"], "bold")
FONT_SMALL    = (_MONO, _FONT_BASE["SMALL"])
FONT_SMALL_B  = (_MONO, _FONT_BASE["SMALL"], "bold")
FONT_DIM      = (_MONO, _FONT_BASE["DIM"])
FONT_BRAILLE  = (_MONO, _FONT_BASE["BRAILLE"])

FONT_SCALE_MIN  = 0.75
FONT_SCALE_MAX  = 2.00
FONT_SCALE_STEP = 0.10


def apply_font_scale(scale: float) -> float:
    """Rescale all FONT_* module globals relative to their base sizes.

    Clamped to [FONT_SCALE_MIN, FONT_SCALE_MAX] so runaway key-repeat
    can't produce illegible or oversized fonts. Integer rounding with
    a floor of 6 guarantees even the smallest fonts remain renderable
    on the lowest zoom level. Returns the actually-applied scale so
    the caller can persist it.
    """
    global FONT_TITLE, FONT_SUBTITLE, FONT_HEADER, FONT_BODY, FONT_BODY_B
    global FONT_SMALL, FONT_SMALL_B, FONT_DIM, FONT_BRAILLE

    scale = max(FONT_SCALE_MIN, min(FONT_SCALE_MAX, float(scale)))

    def _sz(key: str) -> int:
        return max(6, int(round(_FONT_BASE[key] * scale)))

    FONT_TITLE    = (_MONO, _sz("TITLE"), "bold")
    FONT_SUBTITLE = (_MONO, _sz("SUBTITLE"))
    FONT_HEADER   = (_MONO, _sz("HEADER"), "bold")
    FONT_BODY     = (_MONO, _sz("BODY"))
    FONT_BODY_B   = (_MONO, _sz("BODY"), "bold")
    FONT_SMALL    = (_MONO, _sz("SMALL"))
    FONT_SMALL_B  = (_MONO, _sz("SMALL"), "bold")
    FONT_DIM      = (_MONO, _sz("DIM"))
    FONT_BRAILLE  = (_MONO, _sz("BRAILLE"))
    return scale

ACCENT_SOFT = "#60a5fa"
ACCENT_TURBO = "#38bdf8"  # Sky blue — TurboQuant accent

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: Custom Canvas Widgets
# ═══════════════════════════════════════════════════════════════════════════════

_BRAILLE_FILLED = "⣿"
_BRAILLE_HALF   = "⣇"
_BRAILLE_EMPTY  = "⡀"

class BrailleBar(tk.Canvas):
    """Braille-style VRAM usage bar — MyIDE GPU Monitor style."""

    def __init__(self, parent, theme, width=220, height=16, chars=16, **kw):
        canvas_bg = kw.pop("canvas_bg", theme.bg)
        super().__init__(parent, width=width, height=height,
                         bg=canvas_bg, highlightthickness=0, **kw)
        self._theme = theme
        self._chars = chars
        self._font = FONT_BRAILLE
        self._width = width

    def set_value(self, used_gb: float, total_gb: float, canvas_bg: str = None):
        self.delete("all")
        if canvas_bg:
            self.configure(bg=canvas_bg)
        t = self._theme
        if total_gb <= 0:
            return
        pct = min(100, max(0, used_gb / total_gb * 100))
        sub_units = self._chars * 2
        filled_units = int(sub_units * pct / 100)
        full_chars = filled_units // 2
        has_half = filled_units % 2 == 1
        empty_chars = self._chars - full_chars - (1 if has_half else 0)
        if pct > 95:
            fill_color = t.red
        elif pct > 80:
            fill_color = t.yellow
        else:
            fill_color = t.green
        y_mid = int(self.cget("height")) // 2
        if full_chars + (1 if has_half else 0) > 0:
            filled_str = _BRAILLE_FILLED * full_chars + (_BRAILLE_HALF if has_half else "")
            self.create_text(4, y_mid, text=filled_str, anchor="w",
                             font=self._font, fill=fill_color)
        if empty_chars > 0:
            empty_str = _BRAILLE_EMPTY * empty_chars
            offset = (full_chars + (1 if has_half else 0))
            x_offset = 4 + offset * self._char_width()
            self.create_text(x_offset, y_mid, text=empty_str, anchor="w",
                             font=self._font, fill=t.border)

    def _char_width(self):
        try:
            import tkinter.font as tkfont
            f = tkfont.Font(font=self._font)
            return f.measure(_BRAILLE_FILLED)
        except Exception:
            return 9

class HoverButton(tk.Canvas):
    """Colored button with hover effect, drawn on Canvas for full color control.

    v0.48: also supports an optional outer border_color (drawn as a 2px
    rounded frame around the fill) and an optional corner_glyph (one or
    two characters painted in the top-right corner). Both default to
    None which reproduces the original v0.47 appearance exactly.
    """

    # Sentinel for "argument not passed" vs "argument passed as None" in
    # configure_btn(): passing None explicitly clears border_color /
    # corner_glyph, while omitting the argument preserves the current value.
    _UNSET = object()

    def __init__(self, parent, theme, text="", color=None, command=None,
                 width=90, height=26, font=None, **kw):
        canvas_bg = kw.pop("canvas_bg", theme.bg)
        super().__init__(parent, width=width, height=height,
                         bg=canvas_bg, highlightthickness=0, cursor="hand2", **kw)
        self._theme = theme
        self._text = text
        self._color = color or theme.button_bg
        self._command = command
        self._width = width
        self._height = height
        self._hover = False
        self._disabled = False
        self._font = font or FONT_SMALL_B
        self._border_color = None   # v0.48: outer ring color (None = none)
        self._corner_glyph = None   # v0.48: single char in top-right corner
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonRelease-1>", self._on_click)
        self._draw()

    def configure_btn(self, text=None, color=None, state=None, command=None,
                      border_color=_UNSET, corner_glyph=_UNSET):
        """Update any subset of the button's visual properties.

        border_color and corner_glyph use a sentinel default so that
        omitting them preserves the current value (unlike passing None,
        which explicitly clears the border or glyph).
        """
        if text is not None:
            self._text = text
        if color is not None:
            self._color = color
        if state is not None:
            self._disabled = (state == "disabled")
            self.configure(cursor="" if self._disabled else "hand2")
        if command is not None:
            self._command = command
        if border_color is not HoverButton._UNSET:
            self._border_color = border_color
        if corner_glyph is not HoverButton._UNSET:
            self._corner_glyph = corner_glyph
        self._draw()

    def _draw(self):
        self.delete("all")
        t = self._theme
        w, h = self._width, self._height
        r = 4
        if self._disabled:
            bg = t.bg_secondary
            fg = t.fg_dim
        elif self._hover:
            bg = self._lighten(self._color, 0.15)
            fg = "#ffffff"
        else:
            bg = self._color
            fg = "#ffffff"
        # Outer border ring (v0.48) — drawn first so the fill sits on top.
        # A 2px ring is painted by drawing a slightly larger rounded rect
        # in the border colour, then the normal fill rect on top of it.
        if self._border_color:
            self._rounded_rect(0, 0, w, h, r + 1, fill=self._border_color,
                               outline="")
            self._rounded_rect(2, 2, w - 2, h - 2, r, fill=bg, outline="")
        else:
            self._rounded_rect(1, 1, w - 1, h - 1, r, fill=bg, outline="")
        self.create_text(w // 2, h // 2, text=self._text, fill=fg,
                         font=self._font, anchor="center")
        # Anchor glyph in the top-right corner (v0.48).
        if self._corner_glyph:
            self.create_text(w - 6, 5, text=self._corner_glyph,
                             fill="#ffffff", anchor="ne",
                             font=("Consolas", 7, "bold"))

    def _rounded_rect(self, x1, y1, x2, y2, r, **kw):
        self.create_arc(x1, y1, x1+2*r, y1+2*r, start=90, extent=90, style="pieslice", **kw)
        self.create_arc(x2-2*r, y1, x2, y1+2*r, start=0, extent=90, style="pieslice", **kw)
        self.create_arc(x2-2*r, y2-2*r, x2, y2, start=270, extent=90, style="pieslice", **kw)
        self.create_arc(x1, y2-2*r, x1+2*r, y2, start=180, extent=90, style="pieslice", **kw)
        self.create_rectangle(x1+r, y1, x2-r, y2, **kw)
        self.create_rectangle(x1, y1+r, x2, y2-r, **kw)

    @staticmethod
    def _lighten(hex_color, factor):
        try:
            r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
            return f"#{int(r+(255-r)*factor):02x}{int(g+(255-g)*factor):02x}{int(b+(255-b)*factor):02x}"
        except Exception:
            return hex_color

    def _on_enter(self, e):
        if not self._disabled:
            self._hover = True
            self._draw()

    def _on_leave(self, e):
        self._hover = False
        self._draw()

    def _on_click(self, e):
        if not self._disabled and self._command:
            self._command()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: Tooltip
# ═══════════════════════════════════════════════════════════════════════════════

class ToolTip:
    """Hover tooltip for any widget."""
    def __init__(self, widget, text: str, theme=None, delay=400):
        self._widget = widget
        self._text = text
        self._theme = theme
        self._delay = delay
        self._tip_window = None
        self._after_id = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def update_text(self, text: str):
        self._text = text

    def _schedule(self, event):
        self._after_id = self._widget.after(self._delay, self._show)

    def _hide(self, event=None):
        if self._after_id:
            self._widget.after_cancel(self._after_id)
            self._after_id = None
        if self._tip_window:
            self._tip_window.destroy()
            self._tip_window = None

    def _show(self):
        if self._tip_window or not self._text:
            return
        x = self._widget.winfo_rootx() + 8
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        bg = "#1e293b" if self._theme and self._theme.bg.startswith("#1") else "#fefce8"
        fg = "#e2e8f0" if self._theme and self._theme.bg.startswith("#1") else "#1e293b"
        label = tk.Label(tw, text=self._text, font=FONT_SMALL, bg=bg, fg=fg,
                         relief="solid", borderwidth=1, padx=6, pady=3, justify="left")
        label.pack()
        self._tip_window = tw

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: GPU Detection (NVIDIA + AMD)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GPUInfo:
    index: int
    name: str
    vendor: str
    vram_mb: int = 0
    driver: str = ""

def _run_cmd(cmd: List[str], timeout: int = 5) -> str:
    try:
        kw = {"capture_output": True, "text": True, "timeout": timeout, "stdin": subprocess.DEVNULL}
        if sys.platform == "win32":
            kw["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        r = subprocess.run(cmd, **kw)
        return r.stdout if r.returncode == 0 else ""
    except Exception:
        return ""

def detect_all_gpus() -> List[GPUInfo]:
    gpus = []
    idx = 0
    out = _run_cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                     "--format=csv,noheader,nounits"])
    if out:
        for line in out.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append(GPUInfo(index=idx, name=parts[0], vendor="nvidia",
                                    vram_mb=int(float(parts[1])), driver=parts[2]))
                idx += 1
    out = _run_cmd(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--csv"])
    if out:
        for line in out.strip().split("\n")[1:]:  # skip header
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                name = parts[0] if parts[0] else "AMD GPU"
                vram = 0
                try:
                    vram = int(float(parts[1])) // (1024 * 1024) if len(parts) > 1 else 0
                except (ValueError, IndexError):
                    pass
                gpus.append(GPUInfo(index=idx, name=name, vendor="amd", vram_mb=vram))
                idx += 1
    if sys.platform == "win32":
        out = _run_cmd(["wmic", "path", "win32_VideoController", "get",
                         "Name,AdapterRAM", "/format:csv"])
        for line in out.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3 and parts[1]:
                name = parts[1]
                if "Microsoft" in name:
                    continue
                if any(name.lower() in g.name.lower() for g in gpus):
                    continue
                nl = name.lower()
                vendor = "amd" if ("radeon" in nl or "amd" in nl) else \
                         "intel" if ("intel" in nl or "arc" in nl) else None
                if vendor:
                    try:
                        vram = int(parts[2]) // (1024 * 1024) if parts[2].isdigit() else 0
                    except (ValueError, IndexError):
                        vram = 0
                    cleaned = _clean_wmi_gpu_name(name)
                    gpus.append(GPUInfo(index=idx, name=cleaned, vendor=vendor, vram_mb=vram))
                    idx += 1
    elif sys.platform == "linux":
        out = _run_cmd(["lspci"])
        for line in out.split("\n"):
            if any(x in line for x in ["VGA", "3D", "Display"]):
                nl = line.lower()
                if any(g.vendor == "nvidia" for g in gpus) and "nvidia" in nl:
                    continue
                if "radeon" in nl or "amd" in nl:
                    gpus.append(GPUInfo(index=idx, name=line.split(":")[-1].strip(),
                                        vendor="amd"))
                    idx += 1
                elif "intel" in nl:
                    gpus.append(GPUInfo(index=idx, name=line.split(":")[-1].strip(),
                                        vendor="intel"))
                    idx += 1
    return gpus

def _clean_wmi_gpu_name(raw: str) -> str:
    """Turn WMI's verbose VideoController Name into something short and
    readable for the model-card label column.

    WMI Name field examples:
      "NVIDIA GeForce RTX 5090"              -> "GeForce RTX 5090"
      "AMD Radeon RX 7900 XTX"               -> "Radeon RX 7900 XTX"
      "AMD Radeon(TM) Graphics"              -> "Radeon iGPU"       (APU)
      "Intel(R) UHD Graphics 770"            -> "Intel UHD 770"     (iGPU)
      "Intel(R) Iris(R) Xe Graphics"         -> "Intel Iris Xe"     (iGPU)
      "Intel(R) Arc(TM) A770 Graphics"       -> "Intel Arc A770"    (discrete)
    """
    import re
    s = raw
    # Strip trademark marks
    s = s.replace("(TM)", "").replace("(R)", "")
    s = re.sub(r"[™®]", "", s)
    # Collapse spaces
    s = re.sub(r"\s+", " ", s).strip()
    low = s.lower()

    # AMD integrated (APU) — name is just "AMD Radeon Graphics" with no model
    if "radeon graphics" in low and not any(
            c.isdigit() for c in low.split("radeon")[-1].split("graphics")[0]):
        return "Radeon iGPU"

    # Intel iGPU family: UHD/HD/Iris Xe with integrated silicon
    if "intel" in low:
        # Strip "Intel " prefix, then re-prefix "Intel " for consistency
        tail = re.sub(r"^Intel\s+", "", s, flags=re.IGNORECASE)
        # Drop trailing "Graphics" — it's redundant with the category
        tail = re.sub(r"\s+Graphics\s*$", "", tail, flags=re.IGNORECASE)
        tail = tail.strip()
        if tail:
            return f"Intel {tail}"
        return "Intel iGPU"

    # AMD discrete (has a model number in the Radeon line)
    if "amd" in low or "radeon" in low:
        s = re.sub(r"^AMD\s+", "", s, flags=re.IGNORECASE)
        return s.strip()

    # NVIDIA — strip "NVIDIA " prefix, keep the rest
    if "nvidia" in low or "geforce" in low:
        s = re.sub(r"^NVIDIA\s+", "", s, flags=re.IGNORECASE)
        return s.strip()

    # Fallback — cap at 28 chars
    if len(s) > 28:
        s = s[:26] + "…"
    return s or raw[:28]

def detect_cpu_ram_gb() -> float:
    """Detect total system RAM in GB."""
    system = platform.system()
    if system == "Windows":
        out = _run_cmd(["wmic", "ComputerSystem", "get", "TotalPhysicalMemory", "/format:value"])
        for line in out.strip().split("\n"):
            if "TotalPhysicalMemory" in line:
                try:
                    return int(line.split("=")[1].strip()) / (1024**3)
                except (ValueError, IndexError):
                    pass
    elif system == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        return int(line.split()[1]) / (1024**2)
        except Exception:
            pass
    elif system == "Darwin":
        out = _run_cmd(["sysctl", "-n", "hw.memsize"])
        if out.strip():
            try:
                return int(out.strip()) / (1024**3)
            except ValueError:
                pass
    return 0.0

def detect_cuda_version() -> str:
    """Detect CUDA toolkit version via nvidia-smi or nvcc."""
    out = _run_cmd(["nvidia-smi"])
    if out:
        for line in out.split("\n"):
            if "CUDA Version" in line:
                # Format: "| NVIDIA-SMI 572.83    Driver Version: 572.83    CUDA Version: 12.8 |"
                parts = line.split("CUDA Version:")
                if len(parts) > 1:
                    ver = parts[1].strip().rstrip("|").strip()
                    return ver
    out = _run_cmd(["nvcc", "--version"])
    if out:
        for line in out.split("\n"):
            if "release" in line.lower():
                # "Cuda compilation tools, release 12.4, V12.4.131"
                parts = line.split("release")
                if len(parts) > 1:
                    ver = parts[1].strip().split(",")[0].strip()
                    return ver
    return ""

REQUIRED_DLLS_CORE = [
    "ggml-cuda.dll", "ggml-base.dll", "ggml-cpu.dll",
    "ggml.dll", "llama.dll",
]

# cuBLAS DLLs: check for 12.x OR 13.x (whichever matches the build)
CUBLAS_VARIANTS = [
    ("cublas64_12.dll", "cublasLt64_12.dll"),    # CUDA 12.x
    ("cublas64_13.dll", "cublasLt64_13.dll"),    # CUDA 13.x
]

def check_required_dlls(server_dir: str) -> List[Dict]:
    """Check for required DLLs next to llama-server.exe. Returns list of {name, found, size}.
    Detects cuBLAS version automatically (12.x or 13.x)."""
    results = []

    # Find which cuBLAS variant is present
    cublas_found = False
    for cublas, cublaslt in CUBLAS_VARIANTS:
        if os.path.isfile(os.path.join(server_dir, cublas)):
            cublas_found = True
            for dll in (cublas, cublaslt):
                path = os.path.join(server_dir, dll)
                found = os.path.isfile(path)
                size_mb = round(os.path.getsize(path) / (1024 * 1024), 1) if found else 0
                results.append({"name": dll, "found": found, "size_mb": size_mb})
            break

    if not cublas_found:
        # Neither version found — report 12.x as missing (default expectation)
        for dll in CUBLAS_VARIANTS[0]:
            results.append({"name": dll, "found": False, "size_mb": 0})

    # Core DLLs
    for dll in REQUIRED_DLLS_CORE:
        path = os.path.join(server_dir, dll)
        found = os.path.isfile(path)
        size_mb = 0
        if found:
            try:
                size_mb = round(os.path.getsize(path) / (1024 * 1024), 1)
            except OSError:
                pass
        results.append({"name": dll, "found": found, "size_mb": size_mb})
    return results

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: Model Scanner
# ═══════════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────────
# GGUF metadata parser (v0.57).
#
# Replaces the v0.56 filename-substring check as the primary source of truth
# for "is this a reasoning model?". Reads only the header of each GGUF file
# (first 1 MB max) — metadata sits BEFORE the tensor data in GGUF, so we
# never touch the multi-GB weight section. Per file this is typically
# 5–50 KB of disk IO + a small amount of parsing, i.e. sub-millisecond on
# any modern SSD.
#
# We intentionally do NOT pull in the `gguf-py` package (part of llama.cpp)
# to avoid a 3 MB dependency for what amounts to reading three string
# values. The parser understands enough of the GGUF v2/v3 spec to walk the
# metadata KV section and extract `general.architecture`, `general.name`,
# and `tokenizer.chat_template`. All other keys are skipped by advancing
# the file pointer past their serialized length.
#
# On malformed / truncated / non-GGUF files the parser returns None and
# the caller falls back to the legacy filename-substring check. No crash,
# no log spam — a broken file is just silently filename-matched.
#
# Reasoning detection rules (applied to the parsed metadata):
#   1. Architecture in {"qwen3", "deepseek2"} → reasoning by default.
#   2. Chat template contains <think> / <|think|> / <|thinking|> markers
#      → the model was trained to emit thinking blocks, i.e. reasoning.
#   3. Neither matches → fall through to filename substring (legacy).

# GGUF value-type enum widths for fixed-width scalars. Used to skip values
# we don't care about.
_GGUF_SCALAR_WIDTHS = {
    0: 1,   # uint8
    1: 1,   # int8
    2: 2,   # uint16
    3: 2,   # int16
    4: 4,   # uint32
    5: 4,   # int32
    6: 4,   # float32
    7: 1,   # bool
    10: 8,  # uint64
    11: 8,  # int64
    12: 8,  # float64
}
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9

# Only these metadata keys are worth keeping. Everything else is skipped.
_GGUF_KEYS_OF_INTEREST = frozenset({
    "general.architecture",
    "general.name",
    "tokenizer.chat_template",
})

# Architectures where reasoning / thinking is the default behaviour.
# Expand as new reasoning-native architectures ship.
_GGUF_REASONING_ARCHITECTURES = frozenset({
    "qwen3",
    "deepseek2",
})

# Substrings in the chat template that indicate a thinking-capable model.
# Lowercased comparison. If any of these appear in tokenizer.chat_template,
# the model was trained to emit <think> blocks.
_GGUF_THINKING_TEMPLATE_MARKERS = (
    "<think>",
    "</think>",
    "<|think|>",
    "<|thinking|>",
)


def _read_gguf_value(buf: bytes, pos: int, val_type: int, depth: int = 0):
    """Read (or skip) one GGUF metadata value. Returns (value, new_pos).

    Returns (None, None) on any parse error. For non-string types the
    value is returned as None — we only care about strings here, so
    scalars and arrays are advanced past without materialising.
    """
    if depth > 2:
        # Nested arrays of arrays of arrays — pathological, bail out.
        return None, None

    if val_type in _GGUF_SCALAR_WIDTHS:
        w = _GGUF_SCALAR_WIDTHS[val_type]
        if pos + w > len(buf):
            return None, None
        return None, pos + w

    if val_type == _GGUF_TYPE_STRING:
        if pos + 8 > len(buf):
            return None, None
        s_len = struct.unpack_from("<Q", buf, pos)[0]
        pos += 8
        # chat_template can reach multiple KB of Jinja — cap at 4 MB for
        # sanity but accept anything within the buffer we already read.
        if s_len > 4 * 1024 * 1024 or pos + s_len > len(buf):
            return None, None
        try:
            s = buf[pos:pos + s_len].decode("utf-8", errors="replace")
        except Exception:
            return None, None
        return s, pos + s_len

    if val_type == _GGUF_TYPE_ARRAY:
        if pos + 12 > len(buf):
            return None, None
        inner_type = struct.unpack_from("<I", buf, pos)[0]
        pos += 4
        count = struct.unpack_from("<Q", buf, pos)[0]
        pos += 8
        if count > 100_000_000:
            return None, None
        # Scalar-typed arrays: bulk-skip count * width.
        if inner_type in _GGUF_SCALAR_WIDTHS:
            total = count * _GGUF_SCALAR_WIDTHS[inner_type]
            if pos + total > len(buf):
                return None, None
            return None, pos + total
        # Otherwise walk element by element (strings and nested arrays).
        for _ in range(count):
            _, pos = _read_gguf_value(buf, pos, inner_type, depth + 1)
            if pos is None:
                return None, None
        return None, pos

    # Unknown type — can't skip safely.
    return None, None


def _read_gguf_metadata(path: str, max_bytes: int = 1024 * 1024):
    """Read GGUF header metadata. Returns dict of selected string keys or None.

    Only keys in _GGUF_KEYS_OF_INTEREST are kept. All other keys are
    parsed just enough to advance past them. Reads at most `max_bytes`
    from disk — enough for the metadata section on every real-world
    GGUF we've tested (typically 5–50 KB; chat templates push a few
    specific models toward a few hundred KB).

    Supports GGUF v2 and v3. Returns None on v1 (deprecated Jun 2023,
    not seen in practice) and on any parse / IO error.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(max_bytes)
    except OSError:
        return None
    if len(head) < 24 or head[:4] != b"GGUF":
        return None
    try:
        version = struct.unpack_from("<I", head, 4)[0]
    except struct.error:
        return None
    if version < 2:
        # v1 used u32 counts (not u64) and is effectively extinct.
        return None
    try:
        kv_count = struct.unpack_from("<Q", head, 16)[0]
    except struct.error:
        return None
    # Sanity cap — real GGUFs have at most a few hundred metadata keys.
    if kv_count > 10_000:
        return None

    pos = 24
    result = {}
    for _ in range(kv_count):
        # Key: u64 length + utf-8 bytes.
        if pos + 8 > len(head):
            break
        try:
            key_len = struct.unpack_from("<Q", head, pos)[0]
        except struct.error:
            break
        pos += 8
        if key_len > 1024 or pos + key_len > len(head):
            break
        try:
            key = head[pos:pos + key_len].decode("utf-8", errors="replace")
        except Exception:
            break
        pos += key_len

        # Value type: u32.
        if pos + 4 > len(head):
            break
        try:
            val_type = struct.unpack_from("<I", head, pos)[0]
        except struct.error:
            break
        pos += 4

        # Value: read & advance (or skip).
        value, new_pos = _read_gguf_value(head, pos, val_type)
        if new_pos is None:
            # Parse failure — whatever we have in `result` so far is
            # still usable, but we can't continue the walk reliably.
            break
        pos = new_pos
        if key in _GGUF_KEYS_OF_INTEREST and isinstance(value, str):
            result[key] = value

    return result


def _detect_is_reasoning(path: str, filename: str):
    """Return (is_reasoning, architecture) for a GGUF at `path`.

    Tries GGUF metadata first (architecture + chat template), falls back
    to the legacy filename substring check if the GGUF is malformed or
    the metadata is inconclusive. This guarantees v0.57 is at worst as
    accurate as v0.56 (filename-only) and typically much more accurate
    (handles renamed files, exotic quants, fine-tunes with different
    names but intact templates).
    """
    meta = _read_gguf_metadata(path) or {}
    arch = (meta.get("general.architecture") or "").strip().lower()
    tmpl = meta.get("tokenizer.chat_template") or ""

    is_reasoning = False
    if arch in _GGUF_REASONING_ARCHITECTURES:
        is_reasoning = True
    elif tmpl:
        low_tmpl = tmpl.lower()
        if any(m in low_tmpl for m in _GGUF_THINKING_TEMPLATE_MARKERS):
            is_reasoning = True

    # Legacy fallback — if GGUF parse failed or model has no thinking
    # template, still catch it via the filename-substring heuristic.
    if not is_reasoning:
        is_reasoning = _is_reasoning_model(filename)

    return is_reasoning, arch


@dataclass
class ModelInfo:
    filename: str
    path: str
    size_bytes: int
    size_gb: float
    # llama-bench's argument parser truncates paths at the first comma even
    # when properly quoted (it uses commas as list separators for params like
    # `-d 0,8192,32768` and is not context-aware). Models in such paths look
    # like normal entries to llama-server but produce silent FAILED runs in
    # llama-bench. Detected at scan time, surfaced in the UI with a warning
    # marker, and refused with a clear error in _exec_bench. See investigation
    # 2026-04-07: TheTom v2 + Qwen 27B from "Q:\AI, Deeplearning, ...".
    has_unsafe_path: bool = False
    # v0.57: populated from GGUF metadata at scan time. See
    # _detect_is_reasoning() for the decision rules. `architecture` is
    # the lowercased value of `general.architecture` or "" if the GGUF
    # was unreadable / malformed.
    is_reasoning: bool = False
    architecture: str = ""

# Directory name patterns to skip during recursive scanning. These are
# places where .gguf files may exist but are NOT user-selectable LLMs:
# Ollama's blob store (opaque hash filenames), conda/pip caches,
# virtualenvs, build caches, hidden config dirs, Windows system dirs.
# Match is case-insensitive on the basename of the directory.
_SCAN_EXCLUDE_DIRS = frozenset({
    # Python / package manager / env dirs (cross-platform)
    ".cache", ".conda", "conda-meta", ".venv", "venv", "venvs",
    "miniconda3", "anaconda3", "__pycache__", "site-packages",
    "pip-cache", ".npm", "node_modules",
    # Ollama blob store (hash filenames, useless in UI)
    ".ollama", "ollama",
    # Windows-specific junk
    "appdata", "$recycle.bin", "system volume information",
    "programdata", "windows",
    # Browser / IDE / misc caches
    "mozilla", ".mozilla", ".config", ".wine",
    # Compile caches produced by unsloth etc.
    "unsloth_compiled_cache",
    # Build dirs
    "build", "dist",
    # System trash
    ".trash",
})

# Minimum GGUF file size to be considered a "real" model. Anything
# smaller is overwhelmingly vocab-only files (ggml-vocab-*.gguf are
# typically <5 MB), stub test fixtures, or corrupted downloads. Real
# quantized LLMs start around 500 MB even for tiny 1-2B models. We
# pick 50 MB as a conservative lower bound.
_MIN_MODEL_SIZE_BYTES = 50 * 1024 * 1024

# Filename substrings that mark a GGUF as NOT a user-selectable model
# regardless of size. These are files llama.cpp and some model
# distributions ship for internal testing / tokenizer shipping.
_FILENAME_JUNK_HINTS = (
    "ggml-vocab-",   # vocabulary-only dumps from llama.cpp tests
    "-vocab.gguf",   # same pattern, alt naming
    ".tmp.gguf",     # partial downloads
    "mmproj",        # v0.54: companion vision projector, auto-attached
)

def _is_junk_filename(fn: str) -> bool:
    low = fn.lower()
    return any(h in low for h in _FILENAME_JUNK_HINTS)

def scan_models(models_dirs, recursive: bool = True) -> List[ModelInfo]:
    """Scan one or more directories for *.gguf files.

    Accepts either a single path (str) or a list of paths. When
    ``recursive`` is True (default), descends into subdirectories via
    ``os.walk``; otherwise only the top level of each directory is scanned.

    Applies three filters so pointing the launcher at a broad directory
    (e.g. C:\\Users\\wavebo) still yields a clean list of actual LLMs:
      1. Skip common junk directories (venvs, .ollama blob store,
         AppData, conda envs, __pycache__, unsloth caches, etc.)
      2. Skip hidden directories (starting with "."), except the root
         of a user-provided path (which may legitimately be hidden).
      3. Skip files smaller than _MIN_MODEL_SIZE_BYTES (50 MB) and
         files matching _FILENAME_JUNK_HINTS (ggml-vocab-*, etc.).

    Results are deduplicated by canonical (realpath) location so
    overlapping directories or symlinks won't list the same file twice.
    """
    # Normalize input to a list of non-empty strings
    if isinstance(models_dirs, str):
        dirs = [models_dirs] if models_dirs else []
    else:
        dirs = [d for d in (models_dirs or []) if d]

    seen: Dict[str, ModelInfo] = {}
    for models_dir in dirs:
        if not os.path.isdir(models_dir):
            continue
        if recursive:
            # Filter subdirs in-place so os.walk doesn't descend into them.
            for root, subdirs, files in os.walk(models_dir, topdown=True):
                pruned = []
                for d in subdirs:
                    low = d.lower()
                    if low in _SCAN_EXCLUDE_DIRS:
                        continue
                    if d.startswith(".") and d not in (".",):
                        continue
                    pruned.append(d)
                subdirs[:] = pruned
                for f in files:
                    _maybe_add_model(seen, root, f)
        else:
            try:
                entries = os.listdir(models_dir)
            except OSError:
                continue
            for f in entries:
                full = os.path.join(models_dir, f)
                if os.path.isfile(full):
                    _maybe_add_model(seen, models_dir, f)

    models = list(seen.values())
    models.sort(key=lambda m: m.size_bytes, reverse=True)
    return models

def _maybe_add_model(seen: Dict[str, "ModelInfo"], root: str, filename: str) -> None:
    """Helper for scan_models — size/filename filter + dedup insertion."""
    if not filename.lower().endswith(".gguf"):
        return
    if _is_junk_filename(filename):
        return
    full_path = os.path.join(root, filename)
    try:
        size = os.path.getsize(full_path)
    except OSError:
        return
    if size < _MIN_MODEL_SIZE_BYTES:
        return
    try:
        key = os.path.realpath(full_path)
    except OSError:
        key = full_path
    if key in seen:
        return
    # v0.57: read GGUF metadata for reasoning detection. Falls back to
    # filename substring match if the GGUF is unreadable / malformed.
    is_reasoning, arch = _detect_is_reasoning(full_path, filename)
    seen[key] = ModelInfo(
        filename=filename,
        path=full_path,
        size_bytes=size,
        size_gb=round(size / (1024**3), 1),
        has_unsafe_path=("," in full_path),
        is_reasoning=is_reasoning,
        architecture=arch,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: Config Management
# ═══════════════════════════════════════════════════════════════════════════════

def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                cfg.update(saved)
    except Exception:
        pass
    # migrate legacy single-path key to the new list-based field
    legacy = cfg.get("llm_models_path")
    if legacy:
        paths = list(cfg.get("llm_models_paths") or [])
        while len(paths) < MAX_MODELS_PATHS:
            paths.append("")
        if not any(p.strip() for p in paths):
            paths[0] = legacy
        cfg["llm_models_paths"] = paths
        cfg.pop("llm_models_path", None)
    # Ensure the list always has exactly MAX_MODELS_PATHS slots
    paths = list(cfg.get("llm_models_paths") or [])
    while len(paths) < MAX_MODELS_PATHS:
        paths.append("")
    cfg["llm_models_paths"] = paths[:MAX_MODELS_PATHS]
    # v0.56 post-release refactor — merge the two-gate design
    # (safe_settings + autoload checkbox) into a single "Autoload"
    # footer toggle. Preserve user state: if either old flag was True,
    # keep autoload enabled. Drop the legacy key so it doesn't linger
    # in the on-disk config and cause confusion later.
    if "safe_settings" in cfg:
        cfg["autoload"] = bool(cfg.get("autoload", False)
                               or cfg.get("safe_settings", False))
        cfg.pop("safe_settings", None)
    return cfg

def save_config(cfg: dict):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: qnut — Eichhörnchen-Schnellcheck für KV-Cache-Configs
# ═══════════════════════════════════════════════════════════════════════════════
#
# Das Gegenteil eines Labors. Ein Eichhörnchen hat keine Waage: es sieht die
# Nuss an und entscheidet in Sekunden ob sie gut, marginal oder faul ist.
# qnut macht für KV-Cache-Configs genau dasselbe — drei harte Probes, pro
# Config < 90 Sekunden, und wenn schon die erste Probe failt, fertig.
#
# Das Ergebnis ist KEINE detaillierte Pass-Rate-Tabelle, sondern drei
# semantische Anker pro Modell:
#
#   quality_max  = die HÖCHSTE (d.h. am wenigsten komprimierte) Config die
#                  alle drei Probes besteht → "bedenkenlos benutzbar"
#   context_max  = die NIEDRIGSTE (d.h. am stärksten komprimierte) Config
#                  die alle drei Probes besteht → "maximaler Context der
#                  noch hält"
#   neutral      = die mittlere Config zwischen beiden Ankern → "Normalfall"
#
# Der Slider in QLauncher v0.49 wird diese drei Anker als Rast-Zonen
# interpretieren. Zwischen quality_max und context_max kann er stufenlos
# laufen, jenseits der Anker ist Sperrzone — pro Modell unterschiedlich
# breit, weil jede Architektur andere Kompressions-Grenzen hat.

# Geordnete KV-Config-Liste von Quality-Extrem (Index 0) zu Context-Extrem.
# Muss mit KV_CACHE_OPTIONS konsistent bleiben. Die Reihenfolge spiegelt die
# nominelle KV-Compression aus KV_COMPRESSION wider.
NUT_KV_ORDER: tuple[str, ...] = (
    "f16 (default)",       # 1.000 — Quality-Extrem
    "q8_0 / q8_0",         # 0.500
    "q8_0-K + turbo4-V",   # 0.375
    "q8_0-K + turbo3-V",   # 0.350 — typischer Neutral-Kandidat
    "q8_0-K + turbo2-V",   # 0.328 — Boundary V (benötigt LA=7)
    "turbo4 / turbo4",     # 0.250
    "turbo3 / turbo3",     # 0.200 — Context-Extrem
)

# LA modes that qnut tests in phase 2. Must match the LA buttons in
# the launcher's settings bar (off=0, 1, 5, 7). LA=0 is the default
# and is also implicitly tested in phase 0 (baseline) and phase 1
# (KV sweep), so phase 2 only iterates the non-zero LAs.
NUT_LA_MODES: tuple[int, ...] = (0, 1, 5, 7)

# Verdict-Füllfarben für die KV-Buttons. Default-Grau bleibt die theme.border
# und wird in _refresh_kv_button_visuals() selbst verwendet, nicht hier.
NUT_COLORS: dict[str, str] = {
    "good":    "#4ade80",   # grün — prall, bedenkenlos
    "suspect": "#fbbf24",   # amber — marginal, mit Vorsicht
    "rotten":  "#ef4444",   # rot — faul, nicht benutzen
}

# Glyph-Mapping für die drei Anker. Wird in die obere rechte Ecke des
# jeweiligen KV-Buttons gezeichnet wenn Nut-Daten für das Modell vorliegen.
NUT_ANCHOR_GLYPHS: dict[str, str] = {
    "quality_max": "Q",
    "neutral":     "N",
    "context_max": "C",
}

# Die drei Probes. Jede Probe ist ein einzelner HTTP-Call an den llama-server.
# Abbruch bei erstem Fail — wenn direct_math schon bricht, brauchen die
# anderen beiden gar nicht mehr zu laufen (die Config ist schon rotten).
#
# Filler-Texte für die kv_recall- und lookup-Probes sind fest und kurz
# gehalten (≈2000 Tokens). Kein Paragraph-Sampling, keine Context-Buckets —
# das Eichhörnchen fragt nicht "wie gut ist die Nuss bei 500 vs 1500
# Tokens", sondern "hält die Nuss den Standard-Stress oder nicht".

_NUT_FILLER_TEXT = (
    # ≈ 2000 Tokens. Ein einziger fester Block aus neutralen Fakten; kein
    # Random-Sampling, damit jeder Check reproduzierbar ist.
    "The study of long-context language models requires careful attention "
    "to several distinct failure modes. One mode is positional degradation, "
    "where recall quality decreases as the distance between a fact and its "
    "recall point grows. Another mode is attention dilution, where the "
    "softmax distribution becomes too flat to pick out any single memorized "
    "value. A third mode is KV cache corruption, where quantized key or "
    "value vectors accumulate rounding errors that only manifest at certain "
    "context depths. Research into these failure modes has accelerated "
    "since the introduction of grouped query attention and multi-head "
    "latent attention, both of which change how the KV cache is organized "
    "on a per-layer basis. Studies consistently show that attention-V "
    "corruption is harder to detect than attention-K corruption because "
    "errors in V accumulate through the output projection, while errors "
    "in K typically manifest as visible attention score distortions. "
    "Empirical work on Qwen 2.5 and Llama 3 models has found that "
    "symmetric quantization of both K and V at aggressive levels sometimes "
    "produces catastrophic perplexity collapse in specific architecture "
    "and quantization combinations that cannot be predicted from theory "
    "alone. Hybrid approaches, in which K is quantized conservatively "
    "while V is quantized more aggressively, often outperform symmetric "
    "approaches at the same average compression ratio. The reason appears "
    "to be that attention scores are computed from K via a dot product "
    "that amplifies noise, while V is aggregated through a weighted sum "
    "that tends to average noise out across many positions. This "
    "asymmetry explains why q8_0 K combined with turbo3 V is a stable "
    "operating point on many models while symmetric turbo3 is not. "
    "Further research is needed to understand how architecture choices "
    "such as attention head dimension, number of KV heads, and RoPE base "
    "frequency interact with quantization noise at different context "
    "lengths. Some early evidence suggests that models with larger head "
    "dimensions are more tolerant of aggressive V quantization because "
    "the per-head output has more degrees of freedom to absorb error. "
    "Other evidence suggests the opposite for K quantization, where "
    "larger head dimensions lead to more sensitive dot-product "
    "computations that are harder to quantize without losing information. "
    "The interplay between these two effects is an active area of "
    "investigation in the open-source community around llama.cpp and "
    "related inference engines, and the results of that investigation "
    "will inform the next generation of quantization-aware training and "
    "post-training calibration techniques currently being developed at "
    "several research laboratories and open-source projects working on "
    "efficient deployment of large language models on consumer hardware."
)


NUT_PROBES: tuple[dict, ...] = (
    # Probe 1 — direct math, no filler. Quickest test, runs first so a
    # broken config gets cut off early in the worker. The expected key
    # is gone in v0.50: we no longer judge whether the model gave the
    # mathematically correct answer, only whether it gave the SAME
    # answer it gave on the f16 baseline. A dumb model is still dumb on
    # f16, but as long as the other configs are dumb in exactly the
    # same way, the KV-cache is transparent and that's what we want to
    # know.
    {
        "id":      "direct_math",
        "timeout": 30,
        "prompt":  ("Compute [[3,1],[4,2]] \u00d7 [[5,7],[6,8]]. "
                    "Return ONLY [[a,b],[c,d]], nothing else."),
    },
    # Probe 2 — same matrices but with ~2000 tokens of filler between
    # the statement and the question. Stresses KV-recall over context
    # depth. Diverging from baseline here means the compressed KV cache
    # is losing or distorting information that f16 retains.
    {
        "id":      "kv_recall",
        "timeout": 60,
        "prompt": (
            "Memorize these matrices. Do NOT compute yet.\n\n"
            "A = [[3,1],[4,2]]\n"
            "B = [[5,7],[6,8]]\n\n"
            + _NUT_FILLER_TEXT + "\n\n"
            "Now compute A \u00d7 B. Return ONLY [[a,b],[c,d]], "
            "nothing else."
        ),
    },
    # Probe 3 — key-value lookup after filler. Stresses attention-V
    # specifically: the model has to address one specific binding out
    # of ten and pull its value through the compressed cache. A model
    # too small to do this on f16 will also fail on the other configs;
    # the differential check still tells us whether the configs degrade
    # the response in DIFFERENT ways than f16 already did.
    {
        "id":      "lookup",
        "timeout": 60,
        "prompt": (
            "Memorize these bindings exactly:\n\n"
            "ALPHA -> 48291\n"
            "BETA -> 10573\n"
            "GAMMA -> 77420\n"
            "DELTA -> 91826\n"
            "EPSILON -> 33014\n"
            "ZETA -> 66589\n"
            "ETA -> 10422\n"
            "THETA -> 90817\n"
            "IOTA -> 55120\n"
            "KAPPA -> 28046\n\n"
            + _NUT_FILLER_TEXT + "\n\n"
            "What is the value of THETA? "
            "Return ONLY the 5-digit number, nothing else."
        ),
    },
)


def _nut_normalize(s: str) -> str:
    """Canonicalize a response for differential string comparison.

    Strips <think>...</think> blocks, common formatting punctuation,
    and whitespace, then lowercases. Two responses that differ only
    in markdown emphasis or trailing whitespace will compare equal.
    """
    # Remove <think> blocks (Qwen3 / DeepSeek-R1 reasoning models)
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
    s = s.replace(" ", "").replace("\n", "").replace("\t", "")
    s = s.replace("`", "").replace("*", "").replace("_", "")
    return s.strip().lower()


def _nut_responses_match(baseline: str, candidate: str) -> bool:
    """Return True iff the candidate response matches the baseline.

    The match is generous on formatting (whitespace, markdown emphasis,
    case, <think> blocks) but strict on content. Two responses must
    contain exactly the same characters after normalization to count
    as a match.

    An empty candidate (transport error, HTTP failure, timeout) never
    matches anything — including an empty baseline — because there is
    no positive evidence that the config produced any output at all.
    """
    if not candidate:
        return False
    if not baseline:
        # The baseline failed to produce any output. We cannot
        # meaningfully compare against it; treat as no match so the
        # config is not credited with success.
        return False
    return _nut_normalize(baseline) == _nut_normalize(candidate)


def _nut_http_chat(port: int, prompt: str, timeout: int = 60) -> str:
    """Send a single OpenAI-compatible chat completion to llama-server.

    Uses urllib.request to avoid adding aiohttp as a Nuitka-build
    dependency. Returns the assistant message content, or an empty
    string on any error (transport, HTTP status, JSON parse, timeout).
    """
    payload = json.dumps({
        "model":       "default",  # llama-server ignores this field
        "messages":    [{"role": "user", "content": prompt}],
        "stream":      False,
        "temperature": 0.0,
        "max_tokens":  512,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return ""
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


def _nut_probe_server(port: int,
                      baseline_responses: Optional[dict] = None,
                      progress_cb=None) -> tuple:
    """Run the three probes against a running llama-server.

    Differential mode (v0.50): when a baseline_responses dict is given,
    each probe's output is compared against the baseline response for
    the same probe_id. The score is the number of matches. When no
    baseline is given, this IS the baseline run and we just collect
    responses without judgement.

    Args:
        port:               local port the llama-server is listening on
        baseline_responses: dict {probe_id: response_string} from a
                            previous baseline run, or None if this IS
                            the baseline run
        progress_cb:        optional callable(probe_id, ok_or_None)
                            for live UI updates

    Returns:
        (score, responses, results)
        score:     int 0..3 — number of probes whose response matched
                   the baseline. None during baseline runs (we don't
                   judge the baseline against itself).
        responses: dict {probe_id: response_string} — raw answers from
                   this run, suitable for use as a baseline by future
                   probe runs against the same model.
        results:   list of (probe_id, ok_or_None) in run order, for
                   the dialog row display.

    Behaviour:
        * Without baseline (baseline_responses=None): runs all three
          probes unconditionally, collects responses, returns score=None.
          This is the f16 reference run.
        * With baseline: compares each response against baseline. On the
          first mismatch the run continues — we no longer hard-abort,
          because we want a complete (KV, LA) tupling. Score is the
          number of matching probes.
    """
    results: list[tuple[str, Optional[bool]]] = []
    responses: dict[str, str] = {}
    is_baseline = (baseline_responses is None)

    for probe in NUT_PROBES:
        probe_id = probe["id"]
        if progress_cb is not None:
            progress_cb(probe_id, None)
        response = _nut_http_chat(port, probe["prompt"],
                                  timeout=probe["timeout"])
        responses[probe_id] = response

        if is_baseline:
            # No judgement during baseline collection — just record.
            ok = None
            results.append((probe_id, None))
        else:
            ok = _nut_responses_match(baseline_responses[probe_id], response)
            results.append((probe_id, ok))

        if progress_cb is not None:
            progress_cb(probe_id, ok)

    if is_baseline:
        return None, responses, results

    score = sum(1 for _, ok in results if ok)
    return score, responses, results


# Build a fast lookup of "compression strength" for the KV configs we
# care about, so the anchor logic can sort by stronger-compression-wins.
# Lower KV_COMPRESSION value = stronger compression = closer to context
# extreme. We freeze it as a tuple of (kv_name, compression) sorted from
# weakest compression (f16) to strongest (turbo3/turbo3) so the order is
# deterministic and explicit, not implied by NUT_KV_ORDER's accidental
# sort order.
_NUT_KV_BY_COMPRESSION: tuple[tuple[str, float], ...] = tuple(
    sorted(
        ((name, KV_COMPRESSION.get(name, 1.0)) for name in NUT_KV_ORDER),
        key=lambda kv: -kv[1],   # weakest first (f16=1.0), strongest last
    )
)


def _nut_compression_rank(kv_name: str) -> int:
    """Position of kv_name on the compression axis (0 = weakest, 5 = strongest)."""
    for i, (name, _) in enumerate(_NUT_KV_BY_COMPRESSION):
        if name == kv_name:
            return i
    return -1


def _nut_compute_anchors(scores: dict, baseline_kv: str = "f16 (default)",
                         baseline_la: int = 0) -> dict:
    """Derive the three semantic anchors from a (KV, LA) -> score mapping.

    Args:
        scores: {(kv_name, la_mode): match_score (0..3)}
        baseline_kv / baseline_la: identifies which tuple is the f16
                                   reference. By definition this tuple
                                   has the highest possible quality
                                   (it IS the baseline against which
                                   everything else is measured).

    Returns:
        dict with three anchor entries, each shaped {"kv": str, "la": int}:
            quality:  {"kv": <baseline_kv>, "la": <baseline_la>}
            neutral:  strongest-compression tuple with score == 3
            context:  strongest-compression tuple with score >= 2
        Plus a "degraded" boolean flag if no tuple achieved score == 3.
        Empty dict if not even score >= 2 was achievable anywhere.
    """
    # Quality is always the baseline by definition.
    quality = {"kv": baseline_kv, "la": baseline_la}

    # Find the strongest-compression tuple with score == 3 (neutral).
    perfect = [(kv, la) for (kv, la), score in scores.items() if score >= 3]
    if perfect:
        # Sort by compression rank (descending = strongest first).
        # On ties (same KV, different LA), prefer LA=0 as the safer default.
        perfect.sort(key=lambda t: (-_nut_compression_rank(t[0]), t[1]))
        neutral_kv, neutral_la = perfect[0]
        neutral = {"kv": neutral_kv, "la": neutral_la}
        degraded = False
    else:
        # No tuple achieved a perfect match. Fall back to the highest
        # score we did see, with a degraded flag so the dialog and
        # tooltip can warn the user.
        if not scores:
            return {}
        best_score = max(scores.values())
        if best_score < 2:
            # Even the best config diverges from baseline on 2+ probes —
            # the model is unstable on this hardware with any KV config.
            return {}
        candidates = [(kv, la) for (kv, la), s in scores.items()
                      if s == best_score]
        candidates.sort(key=lambda t: (-_nut_compression_rank(t[0]), t[1]))
        neutral_kv, neutral_la = candidates[0]
        neutral = {"kv": neutral_kv, "la": neutral_la}
        degraded = True

    # Find the strongest-compression tuple with score >= 2 (context).
    tolerable = [(kv, la) for (kv, la), score in scores.items() if score >= 2]
    if not tolerable:
        # Should not happen if we got past the degraded check above,
        # but be defensive.
        return {}
    tolerable.sort(key=lambda t: (-_nut_compression_rank(t[0]), t[1]))
    context_kv, context_la = tolerable[0]
    context = {"kv": context_kv, "la": context_la}

    return {
        "quality":  quality,
        "neutral":  neutral,
        "context":  context,
        "degraded": degraded,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: Single-instance Lock & IPC helpers (v0.56)
# ═══════════════════════════════════════════════════════════════════════════════
#
# These functions form the CLI side of the IPC protocol. The *server* side
# (socket accept loop, command dispatch) lives inside TurboQuantQLauncher
# as _start_ipc_server / _handle_ipc_client, because it needs `self` to
# schedule main-thread work via `self.after`.
#
# Protocol (line-based, plain text over a local TCP connection):
#   Client -> "SHUTDOWN\n"   Server -> "OK shutting down\n"
#   Client -> "AUTOSTART\n"  Server -> "OK starting\n"  (or ERR ...)
#   Client -> "STATUS\n"     Server -> "<JSON status>\n"
#   Client -> anything else  Server -> "ERR unknown command\n"
#
# Not a general-purpose RPC. Only three commands, each a single word, each
# with a single line reply. The server closes the connection after one
# exchange. Intentionally minimal to keep the attack surface tiny — the
# socket is bound to 127.0.0.1 only and only opens when the user has
# explicitly enabled Autoload.

def _port_is_free(port: int, host: str = "0.0.0.0") -> bool:
    """Return True if a fresh TCP bind on (host, port) would succeed.

    v0.57 — used to warn the user before starting llama-server on a port
    that is already held by another process. Probes without SO_REUSEADDR
    so we see exactly the same failure mode the llama-server bind would
    hit. Probes on 0.0.0.0 to match the `--host 0.0.0.0` that
    _start_server passes to llama-server — binding on 127.0.0.1 would
    miss collisions with daemons bound to 0.0.0.0 or specific interfaces.

    This is a probe, not a reservation: the returned socket is closed
    immediately, so there is a tiny race window between this check and
    the real llama-server bind. That is fine for a user-facing warning
    — the goal is catching the 99% case of "something is already
    listening on that port", not preventing a TOCTOU race.
    """
    if not (1 <= port <= 65535):
        return False
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        if s is not None:
            try:
                s.close()
            except OSError:
                pass


def _pid_is_alive(pid: int) -> bool:
    """Return True if a process with the given PID currently exists.

    Cross-platform: uses OpenProcess/GetExitCodeProcess on Windows and
    `os.kill(pid, 0)` on POSIX. Safe on both — neither call kills the
    target process. Returns False on any failure (including permission
    denied — which practically still means "not a process we can talk to").
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            h = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not h:
                return False
            try:
                code = ctypes.c_ulong(0)
                ok = ctypes.windll.kernel32.GetExitCodeProcess(
                    h, ctypes.byref(code))
                return bool(ok) and code.value == STILL_ACTIVE
            finally:
                ctypes.windll.kernel32.CloseHandle(h)
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

def read_lock_file() -> Optional[dict]:
    """Return the lock file contents if a live instance owns it, else None.

    Also removes stale lock files (PID no longer alive) as a side effect,
    so the next launcher instance starts clean. Corrupt JSON is treated
    as stale.
    """
    try:
        if not LOCK_FILE.exists():
            return None
        with open(LOCK_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        pid = int(data.get("pid", 0))
        port = int(data.get("port", 0))
        if pid <= 0 or port <= 0:
            raise ValueError("incomplete lock file")
        if not _pid_is_alive(pid):
            raise ValueError("stale PID")
        return {"pid": pid, "port": port,
                "started": data.get("started", "")}
    except Exception:
        # Stale or corrupt — delete so the next run can write a fresh one.
        try:
            LOCK_FILE.unlink()
        except Exception:
            pass
        return None

def send_ipc_command(port: int, command: str,
                     timeout: float = 3.0) -> Optional[str]:
    """Connect to 127.0.0.1:<port>, send one command, return one reply.

    Returns the stripped reply string on success, None on any failure
    (connection refused, timeout, etc.). The caller decides what to do
    with None — usually "instance unreachable, give up quietly".
    """
    try:
        with socket.create_connection(("127.0.0.1", port),
                                      timeout=timeout) as sock:
            sock.sendall((command.strip() + "\n").encode("utf-8"))
            sock.settimeout(timeout)
            buf = b""
            # Read until we get a newline or the server closes. Cap at
            # 64 KB to avoid pathological responses wedging the CLI.
            while b"\n" not in buf and len(buf) < 65536:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
            return buf.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: Main Application
# ═══════════════════════════════════════════════════════════════════════════════

class TurboQuantQLauncher(tk.Tk):
    def __init__(self, autostart_override: Optional[bool] = None):
        """Construct the launcher.

        autostart_override:
          None  -> honour cfg["autoload"] (default GUI behaviour)
          True  -> force a one-shot autoload this session, regardless of
                   the Autoload toggle state. Still requires
                   install_verified (honest hardware check); otherwise
                   it's silently ignored and a warning is logged.
                   Triggered by `--autostart` on the CLI.
          False -> force-skip autoload even if Autoload is ON. Triggered
                   by `--no-autostart` on the CLI, handy for opening the
                   GUI to change settings without accidentally kicking
                   off a model load.
        """
        super().__init__()
        self.title(f"TurboQuant QLauncher v{APP_VERSION}")
        self._first_run = not CONFIG_FILE.exists()
        self.cfg = load_config()
        # v0.56 — CLI override from `--autostart` / `--no-autostart`. None
        # means "use the config flag", True/False force a one-shot
        # decision for this session only (never written back to cfg).
        self._autostart_override = autostart_override
        # v0.56 — IPC server state. Populated by _start_ipc_server when
        # (and only when) Autoload is enabled. Stays None otherwise.
        self._ipc_socket: Optional[socket.socket] = None
        self._ipc_port: int = 0
        self._ipc_accept_thread: Optional[threading.Thread] = None
        # Theme selection: explicit config override wins, else fall back
        # to OS dark-mode detection. The ☀/🌙 checkbox in the header
        # writes cfg["light_mode"] and prompts a restart (we don't do
        # live re-theming because ~200 widgets reference self.theme
        # directly and would all need rebuilding).
        _light_override = self.cfg.get("light_mode")
        if _light_override is None:
            self.is_dark = detect_system_dark_mode()
        else:
            self.is_dark = not bool(_light_override)
        self.theme = DARK_THEME if self.is_dark else LIGHT_THEME

        # v0.58 — Apply persisted font zoom BEFORE any widget is built.
        # apply_font_scale() rewrites the module-level FONT_* globals, and
        # every widget picks up the current tuple at construction time, so
        # setting it here is sufficient for the initial build. Later,
        # Ctrl+Plus/Minus/0 re-invokes apply_font_scale() + triggers a
        # theme-style rebuild so live zoom takes effect immediately.
        # Clamped + sanitised by apply_font_scale itself, so a garbage
        # value in the config can't crash the build.
        try:
            self.cfg["font_scale"] = apply_font_scale(
                self.cfg.get("font_scale", 1.0))
        except Exception:
            # Defensive: if anything goes sideways, fall back to 1.0 so
            # the launcher still comes up with readable default fonts.
            self.cfg["font_scale"] = apply_font_scale(1.0)

        self.gpus: List[GPUInfo] = []
        self.models: List[ModelInfo] = []
        self.server_process: Optional[subprocess.Popen] = None
        self.running_model: Optional[str] = None
        self._log_reader_thread: Optional[threading.Thread] = None
        self._model_cards: Dict[str, dict] = {}
        self._cpu_ram_gb: float = 0.0
        self._bench_kv_set: set = set()      # KV configs selected for Bench All
        self._bench_la_set: set = set()      # LA modes selected for Bench All
        self._bench_stop_event = threading.Event()
        self._bench_thread: Optional[threading.Thread] = None

        # qnut state (v0.48). _current_nut_verdicts/_anchors hold the data
        # for whichever model is currently displayed in the model list; both
        # are empty until either a model with cached anchors is selected or
        # a Nut-Check is run.
        self._current_nut_verdicts: dict = {}
        self._current_nut_anchors: dict = {}
        self._nut_thread: Optional[threading.Thread] = None
        self._nut_cancel_event = threading.Event()
        self._nut_dialog: Optional[tk.Toplevel] = None

        # qnut v0.50 — Q/N/C mode-button state
        # _current_nut_profile holds the {quality, neutral, context, ...}
        # dict for the active (model, server_slot) combination, or None
        # if no profile exists yet. _active_mode_var tracks which of
        # the three mode buttons (if any) is currently engaged. Manual
        # KV/LA clicks reset _active_mode_var to None.
        self._current_nut_profile: Optional[dict] = None
        self._active_mode: Optional[str] = None
        self._mode_buttons: dict = {}
        self._mode_score_labels: dict = {}
        self._nut_rows_frame: Optional[tk.Frame] = None
        self._nut_phase2_placeholder: Optional[tk.Frame] = None
        # qnut v0.50 — concurrency guard. Set True at the very start of
        # _nut_worker, reset False in its finally block (so it always
        # gets cleared even on cancel/error). All entry points that
        # could otherwise launch a competing server (Probe button, Run
        # button, model double-click, Q/N/C mode buttons) check this
        # flag first and refuse to act while a probe is in progress.
        self._probe_in_progress: bool = False
        # v0.56 — the _probe_thinking_warning_acknowledged flag that used
        # to live here became dead code once the blocking "Reasoning
        # Model — Thinking disabled" messagebox was removed from
        # _start_server. Probe's own one-shot dialog is independent and
        # still fires before each Probe run; it no longer needs to
        # suppress anything at tuple-start time.

        self._configure_theme()
        self._build_header()
        self._build_settings_bar()
        # Footer MUST be packed BEFORE the expanding PanedWindow.
        # tkinter's pack manager gives "expand=True" widgets all remaining
        # space at the moment they are packed. If the footer comes after
        # the paned window, shrinking the window clips the footer first
        # because the paned window has already claimed the entire area.
        # By packing the footer with side="bottom" up front, tkinter
        # reserves its fixed height, and the paned window only expands
        # into what's left. Result: footer stays visible at any size.
        self._build_footer()
        self._paned = tk.PanedWindow(self, orient=tk.VERTICAL,
                                      bg=self.theme.border,
                                      sashwidth=5, sashrelief="flat",
                                      handlesize=0, showhandle=False)
        self._paned.pack(fill="both", expand=True, padx=16, pady=(4, 4))
        self._build_model_list()
        self._build_log_area()
        self._pending_bench = None  # (model, gpu_label, results, kv_name) — set after bench

        # v0.55: Adaptive default window size + minimal min_size.
        #
        # Problem: 1000x800 default + 860x600 min_size + Tk scaling 1.5x
        # on HiDPI = effective 1500x1200 / 1290x900. On a 1366x768 laptop
        # or any 1080p screen with taskbar, the window doesn't fit.
        #
        # Fix: cap the default to what actually fits on this screen
        # (minus a small margin for taskbar/titlebar), and lower the
        # min_size so the user can mouse-drag the window to any size
        # they want without bumping into a hard floor.
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        # Tk scaling factor — 1.0 on FHD, 1.5 / 2.0 on QHD / 4K. Default
        # window dimensions are pre-scaling pixel values, so we have to
        # divide the available screen area by it before clamping.
        try:
            tk_scale = float(self.tk.call("tk", "scaling")) / (96.0 / 72.0)
        except Exception:
            tk_scale = 1.0
        # 60 px margin reserves room for taskbar (Windows ~40-50 px) +
        # window decorations. Better to undershoot than to overshoot.
        max_w = int((screen_w - 60) / tk_scale)
        max_h = int((screen_h - 100) / tk_scale)

        w = min(self.cfg.get("window_w", 1000), max_w)
        h = min(self.cfg.get("window_h", 800), max_h)
        self.geometry(f"{w}x{h}")
        # Aggressive min_size: 600x400 lets the user shrink the window
        # to almost any size. Content remains accessible because the
        # settings bar is now horizontally scrollable (see
        # _build_settings_bar) and the model list / log already scroll.
        self.minsize(600, 400)
        if self.cfg.get("window_x") is not None:
            self.geometry(f"+{self.cfg['window_x']}+{self.cfg['window_y']}")
        else:
            self._center_window(self, w, h)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # v0.58 — Font-zoom keyboard shortcuts. bind_all so the shortcuts
        # fire regardless of which widget currently has focus (otherwise
        # Ctrl+Plus inside the log Text widget would be eaten by Tk's
        # default text binding). We listen on both the main-row "+" / "-"
        # keys AND the numpad variants (KP_Add / KP_Subtract), plus the
        # "=" key because on US/DE layouts Ctrl+= is what you actually
        # press when you mean Ctrl+Plus without the Shift modifier —
        # matches browser convention. Ctrl+0 resets to 100%.
        #
        # Bound here once in __init__ rather than inside the rebuild path
        # so they survive theme swaps and font-zoom rebuilds — bind_all
        # attaches to the root, which is never destroyed.
        self.bind_all("<Control-plus>",
                      lambda e: self._on_font_zoom(+FONT_SCALE_STEP))
        self.bind_all("<Control-equal>",
                      lambda e: self._on_font_zoom(+FONT_SCALE_STEP))
        self.bind_all("<Control-KP_Add>",
                      lambda e: self._on_font_zoom(+FONT_SCALE_STEP))
        self.bind_all("<Control-minus>",
                      lambda e: self._on_font_zoom(-FONT_SCALE_STEP))
        self.bind_all("<Control-KP_Subtract>",
                      lambda e: self._on_font_zoom(-FONT_SCALE_STEP))
        self.bind_all("<Control-Key-0>",
                      lambda e: self._on_font_zoom(0.0))

        if self._first_run:
            self.after(100, self._first_run_setup)
        else:
            self.after(100, self._do_initial_scan)
        if self.cfg.get("sash_pos") is not None:
            self.after(150, self._restore_sash)

        # v0.56 — Bring up the IPC control socket if (and only if) the user
        # has already enabled Autoload in a previous session. For a fresh
        # install / unverified / opted-out setup this is a no-op: no
        # socket is bound, no lock file is written, nothing externally
        # visible happens. The socket is (re)started / torn down on
        # demand when the user flips the Autoload toggle at runtime.
        if self.cfg.get("autoload", False):
            self._start_ipc_server()

        # Note: autoload triggering is done at the tail of
        # _do_initial_scan, once self.models is guaranteed to be populated.
        # See _trigger_autoload_if_eligible for the gating logic.

    # ─── Theme ────────────────────────────────────────────────────────────

    def _configure_theme(self):
        t = self.theme
        self.configure(bg=t.bg)

        # HiDPI Tk scaling. Once DPI awareness is declared in main(),
        # Windows no longer upscales us, which means widgets designed for
        # 96 DPI appear tiny on a 4K monitor. Tk's own scaling factor
        # compensates: it multiplies widget sizes without losing sharpness.
        # Mirrors the Linux build's logic. Runs only on first theme-
        # configure (not on live-swap rebuild) so repeated toggles don't
        # compound.
        if not getattr(self, "_tk_scaling_applied", False):
            try:
                sh = self.winfo_screenheight()
                if sh >= 2000:      # 4K class
                    self.tk.call("tk", "scaling", 2.0)
                elif sh >= 1600:    # 2.5K / QHD class
                    self.tk.call("tk", "scaling", 1.5)
                # Otherwise leave default (FHD / 1080p).
                self._tk_scaling_applied = True
            except Exception:
                pass

        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=t.bg, foreground=t.fg, fieldbackground=t.entry_bg,
                    borderwidth=0, font=FONT_BODY)
        s.configure("TFrame", background=t.bg)
        s.configure("TLabel", background=t.bg, foreground=t.fg, font=FONT_BODY)
        s.configure("H.TFrame", background=t.bg_header)
        s.configure("H.TLabel", background=t.bg_header, foreground=t.fg, font=FONT_BODY)
        s.configure("TCombobox", fieldbackground=t.bg_secondary, foreground=t.fg,
                    font=FONT_BODY, padding=(4, 2))
        s.map("TCombobox", fieldbackground=[("readonly", t.bg_secondary)],
              foreground=[("readonly", t.fg)])
        s.configure("TCheckbutton", background=t.bg, foreground=t.fg, font=FONT_BODY)
        s.map("TCheckbutton", background=[("active", t.bg)])
        s.configure("TSeparator", background=t.border)
        self.option_add("*TCombobox*Listbox.background", t.bg_secondary)
        self.option_add("*TCombobox*Listbox.foreground", t.fg)
        self.option_add("*TCombobox*Listbox.font", FONT_BODY)

    # ─── Header ───────────────────────────────────────────────────────────

    def _build_header(self):
        t = self.theme
        hdr = ttk.Frame(self, style="H.TFrame")
        hdr.pack(fill="x")
        row1 = ttk.Frame(hdr, style="H.TFrame")
        row1.pack(fill="x", padx=16, pady=(8, 2))
        ttk.Label(row1, text="TurboQuant", font=FONT_TITLE,
                  style="H.TLabel", foreground=ACCENT_TURBO).pack(side="left")
        ttk.Label(row1, text="  QLauncher", font=FONT_TITLE,
                  style="H.TLabel").pack(side="left")
        ttk.Label(row1, text=f"v{APP_VERSION}", font=FONT_SMALL,
                  foreground=t.fg_dim, style="H.TLabel").pack(side="left", padx=(8, 0))

        # Theme toggle (right-aligned). Click the label to live-swap
        # between dark and light. The label always shows the *current*
        # mode — "🌙 Dark" while dark, "☀ Light" while light — so the
        # user sees immediate visual feedback after clicking. Backed by
        # self._light_var so the rest of _on_toggle_theme can read it
        # without caring which widget triggered the change.
        self._light_var = tk.BooleanVar(value=not self.is_dark)
        theme_lbl = tk.Label(
            row1,
            text=("☀ Light" if not self.is_dark else "🌙 Dark"),
            font=FONT_SMALL,
            bg=t.bg_header, fg=t.fg_secondary,
            cursor="hand2",
            padx=6, pady=2,
        )
        theme_lbl.pack(side="right")
        self._theme_lbl = theme_lbl

        def _on_theme_label_click(_evt=None):
            self._light_var.set(not self._light_var.get())
            self._on_toggle_theme()

        theme_lbl.bind("<Button-1>", _on_theme_label_click)
        # Subtle hover feedback: brighten the foreground on enter,
        # restore on leave. No background change so the header stays
        # visually quiet.
        theme_lbl.bind("<Enter>",
                       lambda _e: theme_lbl.config(fg=t.fg))
        theme_lbl.bind("<Leave>",
                       lambda _e: theme_lbl.config(fg=t.fg_secondary))
        ToolTip(theme_lbl,
                "Click to switch between dark and light theme.",
                t)

        # Use " " (space) instead of empty string as placeholder text.
        # tkinter gives a Label with text="" zero height — when the real
        # text arrives asynchronously (_do_initial_scan, GPU detection),
        # the label snaps to ~16px line height, pushing the separator,
        # settings bar and paned window down. A single-space placeholder
        # reserves the line height from the start, eliminating the jump.
        self._gpu_label = ttk.Label(hdr, text=" ", font=FONT_SMALL,
                                     foreground=t.fg_secondary, style="H.TLabel")
        self._gpu_label.pack(fill="x", padx=16, pady=(0, 1))
        self._cuda_label = ttk.Label(hdr, text=" ", font=FONT_SMALL,
                                      foreground=t.fg_dim, style="H.TLabel")
        self._cuda_label.pack(fill="x", padx=16, pady=(0, 6))
        ttk.Separator(self).pack(fill="x")

    def _on_toggle_theme(self):
        """Live-swap the active theme without restarting the launcher.

        Tkinter has no built-in theme inheritance — every widget freezes
        its colours at construction time, so a real swap means tearing
        down and rebuilding the widget tree. We do exactly that:

          1. Refuse the swap if a Probe / Bench All run is active
             (those operations would race against the rebuild).
          2. Persist all current UI state to cfg, plus the things that
             only live in widgets (log content, sash position, model
             selection).
          3. Update self.theme / self.is_dark.
          4. Destroy every child of the Tk root.
          5. Re-run the build sequence from __init__ in the same order.
          6. Repopulate everything from cached state on self (gpus,
             models, log lines, qnut profile, running indicator) so the
             user sees the new theme without losing any context.
        """
        want_light = self._light_var.get()
        target_dark = not want_light

        # No-op if the desired state already matches.
        if target_dark == self.is_dark:
            return

        # Refuse mid-operation. Probe and Bench All write to UI widgets
        # from background threads via self.after; tearing down those
        # widgets while they're still being touched would crash.
        if self._probe_in_progress:
            messagebox.showwarning(
                "Theme switch unavailable",
                "A Probe run is currently in progress. Please wait for "
                "it to finish before switching theme.",
            )
            self._light_var.set(not want_light)  # revert checkbox
            return
        if self._bench_thread is not None and self._bench_thread.is_alive():
            messagebox.showwarning(
                "Theme switch unavailable",
                "A Bench All run is currently in progress. Please wait "
                "for it to finish before switching theme.",
            )
            self._light_var.set(not want_light)  # revert checkbox
            return

        # ── Step 1: persist everything we will need to restore ────────
        self.cfg["light_mode"] = want_light
        try:
            self._save_current_config()
        except Exception:
            pass

        # Capture the log Text contents (raw, with embedded timestamps).
        # We lose the per-line tag colours but the timestamps and text
        # itself are preserved verbatim.
        saved_log_dump = ""
        try:
            saved_log_dump = self._log_text.get("1.0", "end-1c")
        except Exception:
            pass

        # Capture the vertical sash position so the model/log split
        # stays where the user put it.
        saved_sash = None
        try:
            _, saved_sash = self._paned.sash_coord(0)
        except Exception:
            pass

        # Capture the active model selection so the same model stays
        # highlighted after the rebuild (otherwise _rebuild_model_cards
        # resets _sel_model to -1).
        saved_sel_model = self._sel_model
        saved_sel_row = self._sel_row

        # ── Step 2: swap the theme ────────────────────────────────────
        self.is_dark = target_dark
        self.theme = DARK_THEME if self.is_dark else LIGHT_THEME

        # ── Step 3: nuke the widget tree ──────────────────────────────
        for child in list(self.winfo_children()):
            try:
                child.destroy()
            except Exception:
                pass

        # ── Step 4: re-run the build sequence (mirror __init__) ───────
        self._configure_theme()
        self._build_header()
        self._build_settings_bar()
        self._build_footer()
        self._paned = tk.PanedWindow(self, orient=tk.VERTICAL,
                                      bg=self.theme.border,
                                      sashwidth=5, sashrelief="flat",
                                      handlesize=0, showhandle=False)
        self._paned.pack(fill="both", expand=True, padx=16, pady=(4, 4))
        self._build_model_list()
        self._build_log_area()

        # ── Step 5: repopulate from cached state ──────────────────────

        # Header GPU label — uses cached self.gpus from the original
        # _do_initial_scan. No re-detection needed.
        try:
            if self.gpus:
                parts = []
                for g in self.gpus:
                    vram = f" ({g.vram_mb // 1024} GB)" if g.vram_mb else ""
                    parts.append(f"GPU {g.index}: {g.name}{vram}")
                if self._cpu_ram_gb > 0:
                    parts.append(f"CPU RAM: {self._cpu_ram_gb:.0f} GB")
                self._gpu_label.config(text="  •  ".join(parts))
            elif self._cpu_ram_gb > 0:
                self._gpu_label.config(
                    text=f"No GPUs detected — CPU mode only — "
                         f"CPU RAM: {self._cpu_ram_gb:.0f} GB")
        except Exception:
            pass

        # Header CUDA label — quick re-detect (no GPU rescan).
        try:
            cuda_ver = detect_cuda_version()
            server_path = self.cfg.get("llama_server_path", "")
            server_dir = os.path.dirname(server_path)
            dll_results = check_required_dlls(server_dir) if server_dir else []
            dll_found = sum(1 for d in dll_results if d["found"])
            dll_total = len(dll_results)
            cuda_parts = []
            if cuda_ver:
                cuda_parts.append(f"CUDA {cuda_ver} (driver)")
            sp_lower = server_path.lower()
            if "cuda128" in sp_lower:
                cuda_parts.append("Build: CUDA 12.8")
            elif "cuda132" in sp_lower:
                cuda_parts.append("Build: CUDA 13.2")
            elif server_dir:
                cuda_parts.append(f"Build: {os.path.basename(server_dir)}")
            if dll_total > 0:
                if dll_found == dll_total:
                    cuda_parts.append(f"DLLs: {dll_found}/{dll_total} ✓")
                    dll_color = self.theme.fg_secondary
                else:
                    cuda_parts.append(f"DLLs: {dll_found}/{dll_total} ✗")
                    dll_color = self.theme.yellow
            else:
                dll_color = self.theme.fg_dim
                cuda_parts.append("DLLs: server path not set")
            self._cuda_label.config(
                text="  •  ".join(cuda_parts), foreground=dll_color)
        except Exception:
            pass

        # GPU buttons (uses cached self.gpus).
        try:
            self._build_gpu_buttons()
        except Exception:
            pass

        # Model cards (uses cached self.models).
        try:
            if self.models:
                self._models_header.config(
                    text=f"Models ({len(self.models)} GGUF, "
                         f"{sum(m.size_gb for m in self.models):.0f} GB)")
            self._rebuild_model_cards()
        except Exception:
            pass

        # Quick-switch slot buttons.
        try:
            self._refresh_slot_buttons()
        except Exception:
            pass

        # Log content — re-insert the saved dump as a single block.
        # We use the "info" tag for everything because per-line tag
        # preservation would require dump/parse cycles that aren't
        # worth the visual fidelity gain.
        if saved_log_dump:
            try:
                self._log_text.config(state="normal")
                self._log_text.delete("1.0", "end")
                self._log_text.insert("end", saved_log_dump + "\n", "info")
                self._log_text.see("end")
                self._log_text.config(state="disabled")
            except Exception:
                pass

        # Restore sash position (after a short delay so the paned
        # window has time to compute its full size).
        if saved_sash is not None:
            self.cfg["sash_pos"] = saved_sash
            self.after(50, self._restore_sash)

        # Restore model selection. _select_cell handles re-selecting
        # the row, refreshing the KV/LA visuals, and triggering the
        # qnut profile load for the active (model, server_slot).
        if (saved_sel_model >= 0
                and saved_sel_model < len(self.models)):
            try:
                self.after(60, lambda: self._select_cell(
                    saved_sel_model, saved_sel_row))
            except Exception:
                pass

        # Refresh the running indicator if a server is still running
        # in the background — _update_running_indicator walks the
        # rebuilt model cards and re-applies the highlight.
        if self.running_model:
            try:
                self.after(70, self._update_running_indicator)
            except Exception:
                pass

        self._log(
            f"Theme switched to {'Light' if want_light else 'Dark'} mode.",
            "info",
        )

    # ─── Font zoom (v0.58) ────────────────────────────────────────────────

    def _on_font_zoom(self, delta: float):
        """Change global font scale and rebuild the UI live.

        delta semantics:
          +FONT_SCALE_STEP → zoom in one notch
          -FONT_SCALE_STEP → zoom out one notch
          0.0              → reset to 1.0 (not "no change" — used by Ctrl+0)

        Mechanics mirror _on_toggle_theme: persist state, rewrite the
        module-level FONT_* tuples via apply_font_scale, nuke the widget
        tree, rebuild, repopulate. The font tuples are re-read by every
        widget at construction time, so the new sizes propagate on the
        rebuild. No widget iteration or .config(font=...) gymnastics.

        Refuses to run mid-operation (Probe, Bench All) for the same
        reason theme-swap does: background threads write to widgets we
        would be tearing down.
        """
        # Refuse during long-running background ops.
        if self._probe_in_progress:
            self._log("Font zoom unavailable during Probe run.", "warn")
            return
        if self._bench_thread is not None and self._bench_thread.is_alive():
            self._log("Font zoom unavailable during Bench All run.", "warn")
            return

        # Compute new scale. delta=0.0 is the "reset to 1.0" convention.
        if delta == 0.0:
            new_scale = 1.0
        else:
            current = float(self.cfg.get("font_scale", 1.0))
            new_scale = round(current + delta, 2)

        # No-op guard: if we're already at the bounds and pushing further
        # in the same direction, bail out before the expensive rebuild.
        clamped = max(FONT_SCALE_MIN, min(FONT_SCALE_MAX, new_scale))
        current = float(self.cfg.get("font_scale", 1.0))
        if abs(clamped - current) < 0.001:
            return

        # ── Save state (mirror _on_toggle_theme) ──────────────────────
        saved_log_dump = ""
        try:
            saved_log_dump = self._log_text.get("1.0", "end-1c")
        except Exception:
            pass
        saved_sash = None
        try:
            _, saved_sash = self._paned.sash_coord(0)
        except Exception:
            pass
        saved_sel_model = self._sel_model
        saved_sel_row = self._sel_row

        # ── Apply new scale + persist ─────────────────────────────────
        applied = apply_font_scale(clamped)
        self.cfg["font_scale"] = applied
        try:
            self._save_current_config()
        except Exception:
            pass

        # ── Rebuild widget tree ───────────────────────────────────────
        for child in list(self.winfo_children()):
            try:
                child.destroy()
            except Exception:
                pass

        self._configure_theme()
        self._build_header()
        self._build_settings_bar()
        self._build_footer()
        self._paned = tk.PanedWindow(self, orient=tk.VERTICAL,
                                      bg=self.theme.border,
                                      sashwidth=5, sashrelief="flat",
                                      handlesize=0, showhandle=False)
        self._paned.pack(fill="both", expand=True, padx=16, pady=(4, 4))
        self._build_model_list()
        self._build_log_area()

        # ── Repopulate cached state ───────────────────────────────────
        try:
            if self.gpus:
                parts = []
                for g in self.gpus:
                    vram = f" ({g.vram_mb // 1024} GB)" if g.vram_mb else ""
                    parts.append(f"GPU {g.index}: {g.name}{vram}")
                if self._cpu_ram_gb > 0:
                    parts.append(f"CPU RAM: {self._cpu_ram_gb:.0f} GB")
                self._gpu_label.config(text="  •  ".join(parts))
            elif self._cpu_ram_gb > 0:
                self._gpu_label.config(
                    text=f"No GPUs detected — CPU mode only — "
                         f"CPU RAM: {self._cpu_ram_gb:.0f} GB")
        except Exception:
            pass

        try:
            cuda_ver = detect_cuda_version()
            server_path = self.cfg.get("llama_server_path", "")
            server_dir = os.path.dirname(server_path)
            dll_results = check_required_dlls(server_dir) if server_dir else []
            dll_found = sum(1 for d in dll_results if d["found"])
            dll_total = len(dll_results)
            cuda_parts = []
            if cuda_ver:
                cuda_parts.append(f"CUDA {cuda_ver} (driver)")
            sp_lower = server_path.lower()
            if "cuda128" in sp_lower:
                cuda_parts.append("Build: CUDA 12.8")
            elif "cuda132" in sp_lower:
                cuda_parts.append("Build: CUDA 13.2")
            elif server_dir:
                cuda_parts.append(f"Build: {os.path.basename(server_dir)}")
            if dll_total > 0:
                if dll_found == dll_total:
                    cuda_parts.append(f"DLLs: {dll_found}/{dll_total} ✓")
                    dll_color = self.theme.fg_secondary
                else:
                    cuda_parts.append(f"DLLs: {dll_found}/{dll_total} ✗")
                    dll_color = self.theme.yellow
            else:
                dll_color = self.theme.fg_dim
                cuda_parts.append("DLLs: server path not set")
            self._cuda_label.config(
                text="  •  ".join(cuda_parts), foreground=dll_color)
        except Exception:
            pass

        try:
            self._build_gpu_buttons()
        except Exception:
            pass

        try:
            if self.models:
                self._models_header.config(
                    text=f"Models ({len(self.models)} GGUF, "
                         f"{sum(m.size_gb for m in self.models):.0f} GB)")
            self._rebuild_model_cards()
        except Exception:
            pass

        try:
            self._refresh_slot_buttons()
        except Exception:
            pass

        if saved_log_dump:
            try:
                self._log_text.config(state="normal")
                self._log_text.delete("1.0", "end")
                self._log_text.insert("end", saved_log_dump + "\n", "info")
                self._log_text.see("end")
                self._log_text.config(state="disabled")
            except Exception:
                pass

        if saved_sash is not None:
            self.cfg["sash_pos"] = saved_sash
            self.after(50, self._restore_sash)

        if (saved_sel_model >= 0
                and saved_sel_model < len(self.models)):
            try:
                self.after(60, lambda: self._select_cell(
                    saved_sel_model, saved_sel_row))
            except Exception:
                pass

        if self.running_model:
            try:
                self.after(70, self._update_running_indicator)
            except Exception:
                pass

        self._log(
            f"Font zoom: {int(applied * 100)}% "
            f"(Ctrl+Plus / Ctrl+Minus / Ctrl+0 to adjust).",
            "info",
        )

    # ─── Settings Bar ─────────────────────────────────────────────────────

    def _make_hbar_autohide(self, hbar):
        """Wrap a scrollbar's set() call so it hides when content fits.

        Returns a callable suitable for the xscrollcommand option of a
        Canvas. When the visible range covers the full content (lo==0,
        hi==1), the scrollbar pack-forgets itself; otherwise it
        re-packs at the bottom of its parent. This avoids reserving
        space for an inactive scrollbar.
        """
        def _set(lo, hi):
            try:
                lo_f, hi_f = float(lo), float(hi)
                if lo_f <= 0.0 and hi_f >= 1.0:
                    hbar.pack_forget()
                else:
                    if not hbar.winfo_ismapped():
                        hbar.pack(side="bottom", fill="x")
                hbar.set(lo, hi)
            except (ValueError, tk.TclError):
                pass
        return _set

    def _build_settings_bar(self):
        t = self.theme
        self._sel_model = -1
        self._sel_row = 0
        self._running_gpu_key = None
        # qnut v0.50 — track the KV/LA settings the currently-running
        # server was started with. _kv_var / _la_var are the "next start"
        # values (they change as soon as the user clicks a KV/LA button),
        # so they cannot be used to detect whether a Q/N/C click would
        # restart the server with identical settings (a no-op case).
        # These two attributes are set in _start_server and cleared in
        # _stop_server / _on_server_exited.
        self._running_kv: Optional[str] = None
        self._running_la: Optional[int] = None

        # v0.55: settings bar lives inside a horizontally scrollable canvas.
        # On narrow windows the KV / LA / Probe / Port / Ctx / Timeout /
        # Run / Stop row used to clip — controls on the right disappeared
        # with no way to reach them. Now a horizontal scrollbar appears
        # on demand (only when content exceeds canvas width).
        bar_outer = tk.Frame(self, bg=t.bg)
        bar_outer.pack(fill="x", padx=16, pady=(8, 0))

        bar_canvas = tk.Canvas(bar_outer, bg=t.bg, highlightthickness=0,
                               height=32, bd=0)
        bar_hbar = ttk.Scrollbar(bar_outer, orient="horizontal",
                                  command=bar_canvas.xview)
        bar_canvas.configure(xscrollcommand=self._make_hbar_autohide(bar_hbar))
        bar_canvas.pack(side="top", fill="x", expand=True)
        # Scrollbar is packed but auto-hides via _make_hbar_autohide when
        # content fits. We pack it in a separate slot so the canvas height
        # doesn't jump when the bar appears/disappears.
        bar_hbar.pack(side="bottom", fill="x")

        row = tk.Frame(bar_canvas, bg=t.bg)
        bar_canvas.create_window((0, 0), window=row, anchor="nw")

        # Update scrollregion when content size changes (e.g. theme rebuild,
        # font reflow). Also resize the canvas height to match the row's
        # natural height so we don't cut off button visuals.
        def _on_row_configure(event):
            bar_canvas.configure(scrollregion=bar_canvas.bbox("all"))
            bar_canvas.configure(height=event.height)
        row.bind("<Configure>", _on_row_configure)

        # Mouse-wheel: shift+wheel scrolls horizontally when cursor is
        # over the bar. Plain wheel is left alone for the model list.
        def _on_shift_wheel(event):
            if sys.platform == "darwin":
                bar_canvas.xview_scroll(-event.delta, "units")
            else:
                bar_canvas.xview_scroll(int(-event.delta / 120) * 3, "units")
        bar_canvas.bind("<Shift-MouseWheel>", _on_shift_wheel)
        row.bind("<Shift-MouseWheel>", _on_shift_wheel)

        kv_label = tk.Label(row, text="KV:", font=FONT_BODY_B, bg=t.bg, fg=t.fg)
        kv_label.pack(side="left")
        ToolTip(kv_label, "KV-Cache quantization for context compression", t)

        self._kv_var = tk.StringVar(value=self.cfg.get("kv_cache", "q8_0-K + turbo4-V"))
        self._kv_buttons = {}
        kv_frame = tk.Frame(row, bg=t.bg)
        kv_frame.pack(side="left", padx=(4, 0))

        kv_config = {
            "f16 (default)":     ("f16",      "16-bit float — full precision, no compression"),
            "q8_0-K + turbo4-V": ("q8₀+t4",   "Keys at q8_0, Values at turbo4 — best for Q4_K_M models"),
            "turbo3 / turbo3":   ("t3/t3",    "Symmetric turbo3 — 5x compression, max context"),
            "turbo4 / turbo4":   ("t4/t4",    "Symmetric turbo4 — 4x compression"),
            "q8_0-K + turbo3-V": ("q8₀+t3",   "Keys at q8_0, Values at turbo3 — high compression"),
            "q8_0-K + turbo2-V": ("q8₀+t2",   "Keys at q8_0, Values at turbo2 — Boundary V "
                                              "(requires LA=7 for first/last 2 layers at q8_0-V, "
                                              "otherwise catastrophic PPL). TheTom 08.04.2026."),
            "q8_0 / q8_0":       ("q8₀/q8₀",  "Symmetric 8-bit quantization, 2x compression"),
        }
        for full_name, (short, tip) in kv_config.items():
            is_active = (full_name == self._kv_var.get())
            color = ACCENT_TURBO if is_active else t.border
            btn = HoverButton(kv_frame, t, text=short, color=color,
                              width=68, height=24,
                              command=lambda n=full_name: self._select_kv(n))
            btn.pack(side="left", padx=1)
            ToolTip(btn, tip, t)
            self._kv_buttons[full_name] = btn

        # qnut v0.50: three mode buttons Q (Quality), N (Neutral),
        # C (Context). Sit between the KV row and the 🐿️qnut trigger
        # button. Disabled (greyed) when no qnut profile exists for
        # the current (model, server_slot) combination. When the user
        # clicks one, the corresponding (kv, la) tuple from the profile
        # is applied to the KV/LA buttons via _select_kv / _select_la
        # so the user sees what the mode means.
        #
        # v0.50.2: each button gets a tiny score label above it (e.g.
        # "3/3", "2/3") that shows the probe score of the anchor tuple
        # that mode points to. The label is blank when no profile
        # exists. This lets the user see — before clicking — exactly
        # how faithful each mode is to the f16 baseline.
        mode_frame = tk.Frame(row, bg=t.bg)
        mode_frame.pack(side="left", padx=(8, 0))
        self._mode_score_labels = {}
        for mode_key, mode_label, mode_tip in (
            ("quality", "Q",
             "Quality mode: highest fidelity, lowest compression "
             "(typically f16 + LA off). Available after running qnut "
             "for this model + server."),
            ("neutral", "N",
             "Neutral mode: best compression that still matches the "
             "f16 baseline exactly. The everyday driver."),
            ("context", "C",
             "Context mode: highest compression that still produces "
             "usable output. Use this for long contexts where you "
             "need every megabyte of KV cache space."),
        ):
            # Each mode gets its own vertical mini-frame: score label
            # on top, button below. Frames are packed side-by-side so
            # Q/N/C stay visually aligned as a group.
            mini = tk.Frame(mode_frame, bg=t.bg)
            mini.pack(side="left", padx=1)

            score_lbl = tk.Label(mini, text=" ", font=FONT_SMALL,
                                 bg=t.bg, fg=t.fg_dim, width=4,
                                 anchor="center")
            score_lbl.pack(side="top")
            self._mode_score_labels[mode_key] = score_lbl

            btn = HoverButton(mini, t, text=mode_label,
                              color=t.border, width=28, height=24,
                              command=lambda k=mode_key: self._apply_nut_mode(k))
            btn.pack(side="top")
            ToolTip(btn, mode_tip, t)
            self._mode_buttons[mode_key] = btn

        # qnut trigger button (v0.50). Sits to the right of the KV button
        # row, after the Q/N/C mode buttons. Internally still called
        # "qnut" everywhere — the user-facing label is "▶ Probe" because
        # qnut is a passive observation tool (no calibration, no
        # modification of the model), and a play-symbol-prefixed verb
        # signals "click to start an action".
        self._nut_button = HoverButton(
            row, t, text="\u25b6 Probe", color=t.border,
            width=80, height=24,
            command=self._open_nut_check_dialog,
        )
        self._nut_button.pack(side="left", padx=(8, 0))
        ToolTip(
            self._nut_button,
            "Probe (qnut): runs all KV+LA combinations against an f16\n"
            "baseline and finds the three Q/N/C anchors for the current\n"
            "model + server combination. Read-only \u2014 does not modify\n"
            "the model. Duration: ~10-25 minutes one-time per model+server,\n"
            "results are cached and reused for all future Q/N/C clicks.",
            t,
        )

        port_label = tk.Label(row, text="  Port:", font=FONT_BODY_B, bg=t.bg, fg=t.fg)
        port_label.pack(side="left", padx=(8, 0))
        ToolTip(port_label, "OpenAI-compatible API port (default: 8080)", t)
        self._port_var = tk.StringVar(value=str(self.cfg.get("port", 8080)))
        port_entry = tk.Entry(row, textvariable=self._port_var, width=5, font=FONT_BODY,
                              bg=t.entry_bg, fg=t.fg, relief="flat", bd=2,
                              insertbackground=t.fg)
        port_entry.pack(side="left", padx=(4, 0))

        # Context size field — passed as -c <ctx> when starting llama-server.
        # Leave empty to use llama-server / model default (256K for Gemma 4 etc.).
        ctx_label = tk.Label(row, text="  Ctx:", font=FONT_BODY_B, bg=t.bg, fg=t.fg)
        ctx_label.pack(side="left", padx=(8, 0))
        ToolTip(ctx_label, "Context size (-c). Leave empty to use llama-server / model default.\n"
                           "Set e.g. 8192 for bounded accuracy tests on models with huge native\n"
                           "context (Gemma 4 native = 256K → enormous KV VRAM reservation).", t)
        self._ctx_var = tk.StringVar(value=str(self.cfg.get("ctx_size", "")))
        ctx_entry = tk.Entry(row, textvariable=self._ctx_var, width=7, font=FONT_BODY,
                             bg=t.entry_bg, fg=t.fg, relief="flat", bd=2,
                             insertbackground=t.fg)
        ctx_entry.pack(side="left", padx=(4, 0))

        # Per-run benchmark timeout (seconds). Replaces the previous
        # hard-coded 300s in _exec_bench. 90s is the empirical sweet spot for
        # 9B–27B models at depths up to 32K — long enough for healthy runs
        # (max observed ~51s on Gemma 4 26B-A4B at d=32768) but tight enough
        # to abort the broken turbo3/turbo3 path on the gemma4 fork (~80–100s)
        # and any kernel timeouts. Increase manually to ~180–600s when
        # benchmarking 70B+ models, then revert.
        timeout_label = tk.Label(row, text="  Timeout:", font=FONT_BODY_B, bg=t.bg, fg=t.fg)
        timeout_label.pack(side="left", padx=(8, 0))
        ToolTip(timeout_label,
                "Benchmark timeout per run, in seconds. Default 90s.\n"
                "Range: 30–1800. Healthy 9B–27B runs finish in <60s; deep\n"
                "contexts on 70B+ models may need 180–600s. Used only by\n"
                "the benchmark path — server start is not affected.", t)
        self._bench_timeout_var = tk.StringVar(
            value=str(self.cfg.get("bench_timeout", 90)))
        timeout_entry = tk.Entry(row, textvariable=self._bench_timeout_var,
                                 width=5, font=FONT_BODY,
                                 bg=t.entry_bg, fg=t.fg, relief="flat", bd=2,
                                 insertbackground=t.fg)
        timeout_entry.pack(side="left", padx=(4, 0))

        self._no_think_var = tk.BooleanVar(value=self.cfg.get("no_thinking", False))
        cb_think = tk.Checkbutton(row, text="No Thinking", variable=self._no_think_var,
                                   font=FONT_SMALL, bg=t.bg, fg=t.fg, selectcolor=t.bg_secondary,
                                   activebackground=t.bg, activeforeground=t.fg)
        cb_think.pack(side="left", padx=(8, 0))
        ToolTip(cb_think, "Disable reasoning/thinking mode (--reasoning off)", t)

        # v0.56 — inline warning for reasoning models + No Thinking. Used
        # to be a blocking messagebox that popped up every time the user
        # started a reasoning model with No Thinking on; now it's a
        # single-line label next to the checkbox. Starts empty (no
        # warning) and is refreshed by _refresh_thinking_warning()
        # whenever (a) the user toggles No Thinking, or (b) a different
        # model row is selected.
        self._thinking_warn_label = tk.Label(
            row, text="", font=FONT_SMALL,
            bg=t.bg, fg=t.yellow)
        self._thinking_warn_label.pack(side="left", padx=(4, 0))
        self._thinking_warn_tooltip = ToolTip(
            self._thinking_warn_label, "", t)
        # Trigger the refresh on checkbox toggle. Model-selection changes
        # call _refresh_thinking_warning() directly from _select_cell
        # (there's no StringVar tracking the active model to trace on).
        self._no_think_var.trace_add(
            "write", lambda *_: self._refresh_thinking_warning())

        # Autoload used to live here as a separate checkbox gated behind
        # a "Safe Settings" footer button (historical v0.56 design).
        # Merged into the single "Autoload" footer toggle — see
        # _build_footer / _refresh_autoload_btn. One gate, one click,
        # one meaning.

        self._bench_var = tk.BooleanVar(value=self.cfg.get("benchmark", False))
        cb_bench = tk.Checkbutton(row, text="Benchmark", variable=self._bench_var,
                                   font=FONT_SMALL, bg=t.bg, fg=t.fg, selectcolor=t.bg_secondary,
                                   activebackground=t.bg, activeforeground=t.fg)
        cb_bench.pack(side="left", padx=(8, 0))
        ToolTip(cb_bench, "When checked, double-click runs llama-bench\ninstead of starting the server", t)

        self._bench_all_var = tk.BooleanVar(value=False)
        cb_bench_all = tk.Checkbutton(row, text="Bench All", variable=self._bench_all_var,
                                       font=FONT_SMALL, bg=t.bg, fg=t.fg, selectcolor=t.bg_secondary,
                                       activebackground=t.bg, activeforeground=t.fg)
        cb_bench_all.pack(side="left", padx=(2, 0))
        ToolTip(cb_bench_all, "Activate: selects ALL KV + LA for benchmarking.\n"
                              "Click individual KV/LA buttons to deselect.\n"
                              "Then press Run to start.", t)
        self._bench_all_var.trace_add("write", self._on_bench_all_toggled)

        la_label = tk.Label(row, text="  LA:", font=FONT_BODY_B, bg=t.bg, fg=t.fg)
        la_label.pack(side="left", padx=(8, 0))
        ToolTip(la_label, "Layer-Adaptive mode (TURBO_LAYER_ADAPTIVE)", t)

        self._la_var = tk.IntVar(value=self.cfg.get("layer_adaptive", 0))
        self._la_buttons = {}
        la_frame = tk.Frame(row, bg=t.bg)
        la_frame.pack(side="left", padx=(4, 0))

        la_config = {
            0: ("off", "Uniform compression — all layers equal"),
            1: ("1",   "First 4 + last 4 layers at q8_0 — best quality"),
            5: ("5",   "Aggressive — enables 128K context on 24GB"),
            7: ("7",   "Boundary V — first 2 + last 2 at q8_0-V"),
        }
        for mode, (short, tip) in la_config.items():
            is_active = (mode == self._la_var.get())
            color = ACCENT_TURBO if is_active else t.border
            btn = HoverButton(la_frame, t, text=short, color=color,
                              width=34, height=24,
                              command=lambda m=mode: self._select_la(m))
            btn.pack(side="left", padx=1)
            ToolTip(btn, tip, t)
            self._la_buttons[mode] = btn

        self._run_btn = HoverButton(row, t, text="Run", color=t.accent,
                                     width=50, height=24,
                                     command=self._bench_run)
        self._run_btn.pack(side="left", padx=(8, 1))
        ToolTip(self._run_btn,
                "Start the selected model on the selected GPU.\n"
                "If 'Benchmark' or 'Bench All' is checked, runs a benchmark\n"
                "with the selected KV × LA configs instead.", t)

        self._stop_btn = HoverButton(row, t, text="Stop", color=t.accent,
                                      width=50, height=24,
                                      command=self._bench_stop)
        self._stop_btn.pack(side="left", padx=1)
        self._stop_btn.configure_btn(state="disabled")
        ToolTip(self._stop_btn, "Cancel running benchmark", t)

        self._status_label = tk.Label(row, text="● Idle", font=FONT_BODY_B,
                                       bg=t.bg, fg=t.fg_dim, cursor="arrow")
        self._status_label.pack(side="right")
        self._status_tooltip = ToolTip(self._status_label, "", t)
        self._status_label.bind("<Button-1>", self._on_status_click)

    def _on_status_click(self, event=None):
        """Open browser when server is running."""
        if self.server_process:
            port = self._port_var.get() or "8080"
            import webbrowser
            webbrowser.open(f"http://localhost:{port}")

    def _update_status_loading(self, filename: str, port: str):
        """Set status label to Loading state (model not yet ready)."""
        t = self.theme
        short = filename[:40] + "…" if len(filename) > 40 else filename
        self._status_label.config(
            text=f"⏳ Loading — {short} @ :{port}",
            fg=t.yellow, cursor="arrow")
        self._status_tooltip.update_text("")

    def _update_status_running(self, filename: str, port: str):
        """Set status label to Running state with browser tooltip."""
        t = self.theme
        self._status_label.config(
            text=f"● Running — {filename} @ :{port}",
            fg=t.green, cursor="hand2")
        self._status_tooltip.update_text(
            f"Click to open in browser\nhttp://localhost:{port}")

    def _update_status_idle(self, text="● Idle"):
        """Reset status label to idle state."""
        t = self.theme
        self._status_label.config(text=text, fg=t.fg_dim, cursor="arrow")
        self._status_tooltip.update_text("")

    @staticmethod
    def _bar_tooltip_text(model_gb: float, eff_gb: float, total_gb: float) -> str:
        """Tooltip text for the VRAM bar explaining KV-adjusted footprint."""
        kv_overhead = eff_gb - model_gb
        pct = eff_gb / total_gb * 100 if total_gb > 0 else 0
        return (f"Model weights:  {model_gb:.1f} GB\n"
                f"KV-cache est.:  {kv_overhead:.1f} GB\n"
                f"Effective:      {eff_gb:.1f} / {total_gb:.0f} GB  ({pct:.0f}%)")

    def _select_kv(self, name: str):
        if self._bench_all_var.get():
            # Toggle mode: add/remove from bench set
            if name in self._bench_kv_set:
                self._bench_kv_set.discard(name)
            else:
                self._bench_kv_set.add(name)
            self._refresh_kv_button_visuals()
        else:
            # Single-select mode (normal)
            self._kv_var.set(name)
            self._refresh_kv_button_visuals()
            self._update_bars_for_kv(name)
            # qnut v0.50: a manual KV click might break the active mode
            # binding (Q/N/C). Clear it if the new selection no longer
            # matches the active anchor.
            self._clear_active_mode_if_user_drifted()

    def _select_la(self, mode: int):
        t = self.theme
        if self._bench_all_var.get():
            # Toggle mode: add/remove from bench set
            if mode in self._bench_la_set:
                self._bench_la_set.discard(mode)
            else:
                self._bench_la_set.add(mode)
            for m, btn in self._la_buttons.items():
                btn.configure_btn(color=ACCENT_TURBO if m in self._bench_la_set else t.border)
        else:
            # Single-select mode (normal)
            self._la_var.set(mode)
            for m, btn in self._la_buttons.items():
                btn.configure_btn(color=ACCENT_TURBO if m == mode else t.border)
            # qnut v0.50: same active-mode-drift check as _select_kv.
            self._clear_active_mode_if_user_drifted()

    def _on_bench_all_toggled(self, *args):
        """Called when Bench All checkbox is toggled."""
        t = self.theme
        if self._bench_all_var.get():
            # Activate: enable Benchmark, select ALL KV and LA
            self._bench_var.set(True)
            self._bench_kv_set = set(KV_CACHE_OPTIONS.keys())
            self._bench_la_set = set(self._la_buttons.keys())
            self._refresh_kv_button_visuals()
            for btn in self._la_buttons.values():
                btn.configure_btn(color=ACCENT_TURBO)
        else:
            # Deactivate: revert to single-select visuals
            self._bench_kv_set.clear()
            self._bench_la_set.clear()
            self._refresh_kv_button_visuals()
            cur_la = self._la_var.get()
            for m, btn in self._la_buttons.items():
                btn.configure_btn(color=ACCENT_TURBO if m == cur_la else t.border)

    def _refresh_kv_button_visuals(self):
        """Apply the classical v0.47 selected/unselected visuals to KV buttons.

        v0.50: the verdict-colour layer (green/amber/red) and the Q/N/C
        anchor glyphs that v0.48 painted on these buttons are gone. The
        new Q/N/C *mode buttons* live as a separate widget group to the
        right of the KV row, and qnut anchors are visualised THERE, not
        on the KV/LA buttons themselves. The KV buttons go back to being
        plain selectors.

        Two visual modes remain:
          1. Bench-All active → toggle visuals (every selected KV in
             accent colour, others in border colour). Same as v0.47.
          2. Normal mode → single-select visuals. Same as v0.47.
        """
        t = self.theme

        if self._bench_all_var.get():
            for name, btn in self._kv_buttons.items():
                active = (name in self._bench_kv_set)
                btn.configure_btn(
                    color=ACCENT_TURBO if active else t.border,
                    border_color=None,
                    corner_glyph=None,
                )
            return

        cur = self._kv_var.get()
        for name, btn in self._kv_buttons.items():
            is_selected = (name == cur)
            btn.configure_btn(
                color=ACCENT_TURBO if is_selected else t.border,
                border_color=None,
                corner_glyph=None,
            )

    # ─── qnut: Nut-Check workflow ──────────────────────────────────────────
    #
    # The Nut-Check is the user-facing trigger for the qnut module defined
    # at module level (NUT_PROBES, NUT_KV_ORDER, _nut_probe_server, ...).
    # All HTTP traffic happens in a background worker thread; tkinter
    # widget updates are marshalled back to the main thread via after().
    #
    # Per config the worker:
    #   1. asks the main thread to set self._kv_var and call _start_server
    #   2. polls /v1/models in the worker thread until ready (or timeout)
    #   3. runs the three probes (HTTP, worker thread)
    #   4. asks the main thread to call _stop_server(silent=True)
    #   5. polls until the port is free again, then moves on
    #
    # The user can cancel any time via the dialog's Abort button. Cancel
    # is observed at the start of every config loop iteration and after
    # every probe.

    def _main_thread_call(self, fn):
        """Schedule fn() in the tkinter main thread and block until done.

        Returns whatever fn() returned, or re-raises its exception in the
        worker thread. Used by the qnut worker so that calls into
        _start_server / _stop_server / dialog widgets stay on the right
        thread without sprinkling after() callbacks all over the worker.
        """
        done = threading.Event()
        result: list = [None]
        error: list = [None]

        def wrapper():
            try:
                result[0] = fn()
            except Exception as exc:  # pragma: no cover
                error[0] = exc
            finally:
                done.set()

        self.after(0, wrapper)
        done.wait()
        if error[0] is not None:
            raise error[0]
        return result[0]

    def _nut_wait_for_server_ready(self, port: int,
                                   timeout: float = 90.0) -> bool:
        """Block until llama-server answers /v1/models, or timeout.

        Runs in the worker thread. Polls every 0.5s. Honours the cancel
        flag — returns False immediately if the user aborts mid-startup.
        """
        deadline = time.monotonic() + timeout
        url = f"http://127.0.0.1:{port}/v1/models"
        while time.monotonic() < deadline:
            if self._nut_cancel_event.is_set():
                return False
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                pass
            time.sleep(0.5)
        return False

    def _nut_wait_for_server_down(self, port: int,
                                  timeout: float = 20.0) -> bool:
        """Block until /v1/models stops answering, or timeout.

        Used between configs so two llama-server instances never try to
        bind to the same port simultaneously.
        """
        deadline = time.monotonic() + timeout
        url = f"http://127.0.0.1:{port}/v1/models"
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=1) as resp:
                    _ = resp.status  # still up
            except Exception:
                return True
            time.sleep(0.3)
        return False

    def _open_nut_check_dialog(self):
        """Confirmation + live-progress dialog for the Nut-Check workflow."""
        if self._probe_in_progress or (
                self._nut_thread and self._nut_thread.is_alive()):
            messagebox.showinfo(
                "Probe Model",
                "A probe run is already in progress. Wait for it to "
                "finish (or cancel it) before starting another.",
            )
            return

        if self._sel_model < 0 or self._sel_model >= len(self.models):
            messagebox.showinfo(
                "Probe Model",
                "Please select a model from the model list first.",
            )
            return

        if self.server_process:
            messagebox.showwarning(
                "Probe Model",
                "A llama-server is currently running. Stop it first "
                "(Esc or the Stop button), then start the probe run.",
            )
            return

        model = self.models[self._sel_model]

        # Pick the GPU the user has highlighted in the model card. The card
        # remembers it as _running_gpu_key when active; otherwise default
        # to GPU 0.
        gpu_key = self._running_gpu_key or "GPU 0"

        # qnut v0.50 — Reasoning model pre-flight check.
        #
        # The Probe worker will force No-Thinking on every tuple to get
        # deterministic outputs (otherwise the differential comparison
        # sees noise from <think> blocks that vary even at temp=0). We
        # still ask the user ONCE here, before the run starts, with an
        # honest probe-specific explanation of the trade-off, so they
        # know the resulting Q/N/C anchors describe the non-thinking
        # behaviour. (In v0.56 the blocking per-tuple warning in
        # _start_server was dropped, so this one-shot dialog is the
        # entire reasoning-model guard for Probe.)

        if model.is_reasoning:
            proceed = messagebox.askyesno(
                "Probe Model — Reasoning Model Detected",
                f"'{model.filename}' looks like a reasoning-capable "
                f"model (Gemma 4 / Qwen3 / DeepSeek-R1 / QwQ family).\n\n"
                f"Probe needs deterministic outputs to compare KV "
                f"configurations against the f16 baseline. Reasoning "
                f"models emit <think> blocks that vary slightly between "
                f"configurations even at temperature 0, which would "
                f"make every tuple look like it diverges.\n\n"
                f"Probe will therefore force '--reasoning off' for the "
                f"entire run. This means the model will be measured "
                f"WITHOUT its thinking ability, and the resulting "
                f"Q / N / C anchors describe how well KV compression "
                f"preserves the model's NON-thinking output.\n\n"
                f"Note: For everyday use you should leave 'No Thinking' "
                f"OFF — this is purely a measurement constraint.\n\n"
                f"Proceed with the Probe run?",
                icon="warning",
                default="yes",
            )
            if not proceed:
                self._log(
                    "Probe cancelled — user declined to disable thinking "
                    "on a reasoning model.",
                    "warn",
                )
                return

        t = self.theme
        dlg = tk.Toplevel(self)
        dlg.title("Probe Model")
        dlg.transient(self)
        # NOTE: no grab_set() — the dialog is intentionally non-modal so
        # the user can resize/scroll/inspect the main window (especially
        # the Server Log) while the probe run is in progress. Concurrent
        # actions on the main window are blocked via _probe_in_progress
        # guards on the Probe button, the Q/N/C mode buttons, the Run
        # button, and the model double-click handler.
        dlg.configure(bg=t.bg)
        dlg.resizable(False, False)
        self._nut_dialog = dlg

        wrap = tk.Frame(dlg, bg=t.bg, padx=18, pady=14)
        wrap.pack()

        tk.Label(wrap, text="\u25b6  Probe Model", font=FONT_TITLE,
                 bg=t.bg, fg=ACCENT_TURBO).pack(anchor="w")
        tk.Label(wrap, text=f"Model: {model.filename}",
                 font=FONT_BODY, bg=t.bg, fg=t.fg).pack(anchor="w", pady=(6, 0))
        tk.Label(wrap, text=f"GPU: {gpu_key}",
                 font=FONT_SMALL, bg=t.bg, fg=t.fg_dim).pack(anchor="w")

        info_lines = (
            "Phase 0: collect f16 baseline responses (\u224890 s).",
            "Phase 1: compare 5 KV configs vs baseline at LA=0.",
            "Phase 2: LA sweep (1, 5, 7) for KVs that survived phase 1.",
            "Read-only \u2014 the model is never modified, only observed.",
            "Cancel any time; partial results are discarded (no profile saved).",
        )
        for line in info_lines:
            tk.Label(wrap, text=line, font=FONT_SMALL,
                     bg=t.bg, fg=t.fg_dim).pack(anchor="w")

        tk.Frame(wrap, bg=t.border, height=1).pack(fill="x", pady=(10, 8))

        # Status rows are added dynamically by the worker via
        # _nut_prepopulate_phase1_rows() before the run starts, plus
        # _nut_prepopulate_phase2_rows() at the phase-1/phase-2 boundary.
        # v0.50 tests up to 24 tuples (6 KV * 4 LA), so a fixed pre-
        # allocation would either waste space or cap the visible info.
        self._nut_status_labels: dict[str, tk.Label] = {}
        self._nut_phase2_placeholder = None
        self._nut_rows_frame = tk.Frame(wrap, bg=t.bg)
        self._nut_rows_frame.pack(fill="x")

        tk.Frame(wrap, bg=t.border, height=1).pack(fill="x", pady=(10, 8))

        # Overall progress line + button row
        self._nut_overall_label = tk.Label(
            wrap, text="Ready. Click 'Start' to begin the check.",
            font=FONT_SMALL, bg=t.bg, fg=t.fg, anchor="w",
        )
        self._nut_overall_label.pack(fill="x", pady=(0, 8))

        btn_row = tk.Frame(wrap, bg=t.bg)
        btn_row.pack(fill="x")

        start_btn = HoverButton(btn_row, t, text="Start",
                                color=ACCENT_TURBO, width=110, height=28)
        start_btn.pack(side="left", padx=(0, 6))

        cancel_btn = HoverButton(btn_row, t, text="Cancel",
                                 color=t.border, width=110, height=28)
        cancel_btn.pack(side="left")

        close_btn = HoverButton(btn_row, t, text="Close",
                                color=t.border, width=110, height=28)
        # Initially hidden — appears when the run finishes.

        def on_start():
            start_btn.configure_btn(state="disabled")
            self._nut_cancel_event.clear()
            self._nut_thread = threading.Thread(
                target=self._nut_worker,
                args=(model, gpu_key),
                daemon=True,
            )
            self._nut_thread.start()

        def on_cancel():
            if self._nut_thread and self._nut_thread.is_alive():
                self._nut_cancel_event.set()
                self._nut_overall_label.configure(
                    text="Cancel requested \u2014 waiting for current probe\u2026")
                cancel_btn.configure_btn(state="disabled")
            else:
                # Not running yet — close the dialog
                dlg.destroy()
                self._nut_dialog = None

        def on_close():
            dlg.destroy()
            self._nut_dialog = None

        start_btn.configure_btn(command=on_start)
        cancel_btn.configure_btn(command=on_cancel)
        close_btn.configure_btn(command=on_close)
        self._nut_dialog_close_btn = close_btn
        self._nut_dialog_btn_row = btn_row

        # Center on parent
        dlg.update_idletasks()
        try:
            x = self.winfo_rootx() + (self.winfo_width() - dlg.winfo_width()) // 2
            y = self.winfo_rooty() + (self.winfo_height() - dlg.winfo_height()) // 2
            dlg.geometry(f"+{max(0, x)}+{max(0, y)}")
        except Exception:
            pass

        dlg.protocol("WM_DELETE_WINDOW", on_cancel)

    def _nut_update_dialog_row(self, kv_name: str, text: str, color: str):
        """Worker → main thread: update one config's status row."""
        if not self._nut_dialog:
            return
        lbl = self._nut_status_labels.get(kv_name)
        if lbl is not None:
            lbl.configure(text=text, fg=color)

    def _nut_update_dialog_overall(self, text: str):
        """Worker → main thread: update the overall progress line."""
        if not self._nut_dialog:
            return
        if hasattr(self, "_nut_overall_label"):
            self._nut_overall_label.configure(text=text)

    def _nut_finish_dialog(self, anchors: dict, scores: dict,
                           cancelled: bool, baseline_failed: bool = False):
        """Worker → main thread: replace Start/Cancel with Close.

        v0.50 takes the new (kv, la)-tuple anchors plus a baseline_failed
        flag for the special "couldn't even start the f16 server" case.
        """
        if not self._nut_dialog:
            return
        if baseline_failed:
            self._nut_overall_label.configure(
                text="Aborted: f16 baseline server failed to start. "
                     "qnut cannot test this model on the current hardware.")
        elif cancelled:
            self._nut_overall_label.configure(
                text=f"Cancelled. {len(scores)} tuples were measured "
                     f"before abort.")
        elif anchors:
            q = anchors["quality"]
            n = anchors["neutral"]
            c = anchors["context"]
            if anchors.get("degraded"):
                # Make the degraded state visible and self-explanatory
                # so the user understands why all three anchors point
                # at the same tuple. Without this hint a degraded run
                # looks like a bug ("why are Q, N and C identical?").
                header = ("Done \u2014 \u26a0 DEGRADED PROFILE\n"
                          "No compressed config matched the f16 baseline\n"
                          "exactly enough. All anchors fall back to f16.")
            else:
                header = "Done."
            self._nut_overall_label.configure(
                text=(f"{header}\n"
                      f"  Q  {q['kv']}  +  LA {q['la']}\n"
                      f"  N  {n['kv']}  +  LA {n['la']}\n"
                      f"  C  {c['kv']}  +  LA {c['la']}"))
        else:
            self._nut_overall_label.configure(
                text="Done — but no (KV, LA) combination matches the "
                     "f16 baseline closely enough to be safe. The model "
                     "is unstable on this hardware with all tested "
                     "compression settings.")
        # Swap button row: hide Start/Cancel, show Close
        for w in self._nut_dialog_btn_row.winfo_children():
            w.pack_forget()
        self._nut_dialog_close_btn.pack(side="left")

    def _nut_worker(self, model, gpu_key: str):
        """Background thread: run the full qnut v0.50 three-phase workflow.

        Phase 0 — Baseline: f16 + LA=off, collect responses (no judgement).
        Phase 1 — KV sweep: 5 remaining KVs at LA=off, score against baseline.
        Phase 2 — LA sweep: for each KV with score >= 2 from phase 1,
                  test the remaining 3 LA modes (1, 5, 7).

        Then derives the three (KV, LA) anchors quality / neutral / context
        and persists them as a profile under the current (model, server)
        combination in cfg['nut_profiles'].
        """
        port_str = self._main_thread_call(lambda: self._port_var.get())
        try:
            port = int(port_str)
        except (TypeError, ValueError):
            port = 8080

        # Save the user's currently-selected KV, LA, and No-Thinking
        # state so we can restore them when the worker finishes (success,
        # cancel, or exception — see the try/finally below).
        saved_kv = self._main_thread_call(lambda: self._kv_var.get())
        saved_la = self._main_thread_call(lambda: self._la_var.get())
        saved_no_thinking = self._main_thread_call(
            lambda: self._no_think_var.get())

        # qnut v0.50 — concurrency lock. Set True NOW so the Probe / Run /
        # Q-N-C / double-click guards on the main window can refuse to
        # act while this worker is running. The matching reset to False
        # lives in the finally block at the end of this method, so it
        # always fires — even if a probe raises mid-run.
        self._probe_in_progress = True
        self._main_thread_call(lambda: self._refresh_mode_button_state())

        # qnut v0.50 — force "No Thinking" for the duration of the run.
        # Reasoning models (Qwen3, DeepSeek-R1) emit <think> blocks that
        # vary slightly between KV/LA configs even at temperature=0,
        # which makes the differential comparison see divergence where
        # there is none. Locking thinking off for the run gives us
        # deterministic, comparable outputs. The original setting is
        # restored in the finally block.
        self._main_thread_call(lambda: self._no_think_var.set(True))

        # The (kv, la) -> score map we will populate over phases 1 and 2.
        scores: dict[tuple[str, int], int] = {}
        # Per-tuple results list, kept for the dialog row display.
        all_results: dict[tuple[str, int], list] = {}
        # The baseline responses we capture in phase 0 and reuse.
        baseline_responses: dict = {}
        # The baseline tuple itself (we use f16 + LA=off as the reference).
        baseline_kv = "f16 (default)"
        baseline_la = 0
        cancelled = False
        baseline_failed = False
        anchors: dict = {}

        def _make_row_label(kv: str, la: int) -> str:
            """Stable display key for the dialog row dict."""
            return f"{kv}|LA={la}"

        def _ensure_dialog_row(kv: str, la: int):
            """Worker -> main thread: add a row to the live dialog if missing."""
            row_label = _make_row_label(kv, la)
            self._main_thread_call(
                lambda l=row_label: self._nut_ensure_row(l))

        def _set_row(kv: str, la: int, text: str, color: str):
            row_label = _make_row_label(kv, la)
            self._main_thread_call(
                lambda l=row_label, t=text, c=color:
                    self._nut_update_dialog_row(l, t, c))

        def _set_overall(text: str):
            self._main_thread_call(
                lambda t=text: self._nut_update_dialog_overall(t))

        def _run_one_tuple(kv: str, la: int,
                           is_baseline: bool) -> tuple[Optional[int], dict, list]:
            """Start the server with (kv, la), run probes, stop the server.

            Returns (score, responses, results). For baseline runs the
            score is None — phase 0 doesn't judge itself.
            """
            _ensure_dialog_row(kv, la)
            _set_row(kv, la, "starting server\u2026", self.theme.fg)

            # Set vars and start server (main thread)
            def _start_with_cfg(k=kv, l=la):
                self._kv_var.set(k)
                self._la_var.set(l)
                self._start_server(model.filename, gpu_key)
            self._main_thread_call(_start_with_cfg)

            if not self._nut_wait_for_server_ready(port, timeout=90):
                self._main_thread_call(
                    lambda: self._stop_server(silent=True))
                self._nut_wait_for_server_down(port, timeout=15)
                _set_row(kv, la, "\u2717 server failed to start",
                         NUT_COLORS["rotten"])
                return None, {}, []

            # Per-probe progress callback updates this row live.
            def _probe_progress(probe_id, ok):
                if ok is None:
                    msg = f"\u25b8 {probe_id}\u2026"
                    color = self.theme.fg_dim
                elif ok is True:
                    msg = f"\u2713 {probe_id} match"
                    color = NUT_COLORS["good"]
                elif ok is False:
                    msg = f"\u2717 {probe_id} diverges"
                    color = NUT_COLORS["rotten"]
                else:
                    msg = f"\u2026 {probe_id}"
                    color = self.theme.fg_dim
                _set_row(kv, la, msg, color)

            score, responses, results = _nut_probe_server(
                port,
                baseline_responses=None if is_baseline else baseline_responses,
                progress_cb=_probe_progress,
            )

            # Final row text from the actual symbols
            if is_baseline:
                # No score for the baseline — show that responses were collected
                final = "\u2713 baseline collected"
                color = NUT_COLORS["good"]
            else:
                symbols = []
                for _pid, ok in results:
                    symbols.append("\u2713" if ok else "\u2717")
                while len(symbols) < 3:
                    symbols.append("\u00b7")
                final = "".join(symbols) + f"  score {score}/3"
                if score == 3:
                    color = NUT_COLORS["good"]
                elif score == 2:
                    color = NUT_COLORS["suspect"]
                else:
                    color = NUT_COLORS["rotten"]
            _set_row(kv, la, final, color)

            self._main_thread_call(
                lambda: self._stop_server(silent=True))
            self._nut_wait_for_server_down(port, timeout=15)

            return score, responses, results

        try:
            # ─────────────────────────────────────────────────────────────
            # Pre-populate the dialog with all 6 phase-0/1 rows AND a
            # placeholder for phase 2. This gives the user the full
            # picture before anything starts running, so they can see
            # the size of the work ahead instead of watching rows
            # appear one by one.
            # ─────────────────────────────────────────────────────────────
            self._main_thread_call(
                lambda: self._nut_prepopulate_phase1_rows(
                    baseline_kv, baseline_la))

            # ─────────────────────────────────────────────────────────────
            # Phase 0 — Baseline (f16 + LA=off)
            # ─────────────────────────────────────────────────────────────
            _set_overall("Phase 0/2: collecting f16 baseline\u2026")
            if self._nut_cancel_event.is_set():
                cancelled = True
            else:
                score, responses, results = _run_one_tuple(
                    baseline_kv, baseline_la, is_baseline=True)
                if not responses:
                    # Baseline server failed to start at all — abort.
                    baseline_failed = True
                else:
                    baseline_responses = responses
                    # The baseline tuple gets a perfect score by definition
                    # (it matches itself).
                    scores[(baseline_kv, baseline_la)] = 3
                    all_results[(baseline_kv, baseline_la)] = results

            # ─────────────────────────────────────────────────────────────
            # Phase 1 — KV sweep at LA=off (skip baseline KV, already done)
            # ─────────────────────────────────────────────────────────────
            if not (cancelled or baseline_failed):
                phase1_kvs = [kv for kv in NUT_KV_ORDER if kv != baseline_kv]
                for idx, kv in enumerate(phase1_kvs, 1):
                    if self._nut_cancel_event.is_set():
                        cancelled = True
                        break
                    _set_overall(
                        f"Phase 1/2: KV sweep [{idx}/{len(phase1_kvs)}]  {kv}")
                    score, _resps, results = _run_one_tuple(
                        kv, 0, is_baseline=False)
                    if score is not None:
                        scores[(kv, 0)] = score
                        all_results[(kv, 0)] = results

            # ─────────────────────────────────────────────────────────────
            # Phase 2 — LA sweep for KVs with score >= 2 (the candidates)
            # ─────────────────────────────────────────────────────────────
            if not (cancelled or baseline_failed):
                la_extras = [la for la in NUT_LA_MODES if la != 0]
                candidates = [
                    kv for kv in NUT_KV_ORDER
                    if scores.get((kv, 0), 0) >= 2 and kv != baseline_kv
                ]
                # Always include the baseline KV in the LA sweep too — we
                # want to know whether f16 with a non-zero LA still
                # matches f16 with LA=off.
                candidates = [baseline_kv] + candidates

                # Pre-populate the phase-2 rows now that we know which
                # tuples will actually run. This replaces the placeholder
                # row with the real list.
                phase2_tuples = [(kv, la) for kv in candidates
                                 for la in la_extras]
                self._main_thread_call(
                    lambda t=list(phase2_tuples):
                        self._nut_prepopulate_phase2_rows(t))

                total_phase2 = len(phase2_tuples)
                done = 0
                for kv in candidates:
                    for la in la_extras:
                        if self._nut_cancel_event.is_set():
                            cancelled = True
                            break
                        done += 1
                        _set_overall(
                            f"Phase 2/2: LA sweep [{done}/{total_phase2}]  "
                            f"{kv}, LA={la}")
                        score, _resps, results = _run_one_tuple(
                            kv, la, is_baseline=False)
                        if score is not None:
                            scores[(kv, la)] = score
                            all_results[(kv, la)] = results
                    if cancelled:
                        break

            # Compute the three anchors from whatever scores we collected
            anchors = _nut_compute_anchors(scores,
                                           baseline_kv=baseline_kv,
                                           baseline_la=baseline_la)

            # Persist the profile under (filename, server_slot)
            if anchors and not baseline_failed:
                self._main_thread_call(
                    lambda: self._nut_save_profile(model.filename, scores,
                                                    baseline_responses,
                                                    anchors))

            # Wrap up the dialog
            self._main_thread_call(
                lambda: self._nut_finish_dialog(anchors, scores, cancelled,
                                                 baseline_failed))

            # Final log line
            if baseline_failed:
                msg = "qnut aborted: f16 baseline server could not be started"
                level = "error"
            elif cancelled:
                msg = (f"qnut cancelled after {len(scores)} tuples measured "
                       f"({sum(1 for s in scores.values() if s == 3)} perfect)")
                level = "warn"
            elif anchors:
                q = anchors["quality"]
                n = anchors["neutral"]
                c = anchors["context"]
                deg = "  [degraded]" if anchors.get("degraded") else ""
                msg = (f"qnut done: {len(scores)} tuples scored. "
                       f"Q={q['kv']}+LA{q['la']}, "
                       f"N={n['kv']}+LA{n['la']}, "
                       f"C={c['kv']}+LA{c['la']}{deg}")
                level = "info"
            else:
                msg = ("qnut done but no usable anchors — model unstable on "
                       "this hardware with all tested KV/LA combinations")
                level = "warn"
            self._main_thread_call(lambda: self._log(msg, level))

        finally:
            # ─────────────────────────────────────────────────────────────
            # Cleanup that MUST happen no matter how the worker exits
            # (success, cancel, exception). Restoring the No-Thinking
            # state and the KV/LA selection here protects the user from
            # finding their settings unexpectedly changed if a probe
            # crashes mid-run.
            # ─────────────────────────────────────────────────────────────
            self._main_thread_call(lambda: self._kv_var.set(saved_kv))
            self._main_thread_call(lambda: self._la_var.set(saved_la))
            self._main_thread_call(
                lambda v=saved_no_thinking: self._no_think_var.set(v))

            # Reload the profile for the current (model, server) so the
            # Q/N/C mode buttons reflect whatever was just saved.
            self._main_thread_call(
                lambda: self._load_nut_profile_for_current_combo())

            # Release the concurrency lock and re-enable the mode buttons
            self._probe_in_progress = False
            self._main_thread_call(lambda: self._refresh_mode_button_state())

    def _nut_save_profile(self, filename: str,
                          scores: dict,
                          baseline_responses: dict,
                          anchors: dict):
        """Persist a qnut profile under (filename, server_slot).

        v0.50 schema:
            cfg["nut_profiles"][filename][server_slot] = {
                checked_at, build_path, build_hash,
                scores: {"<kv>|LA<la>": int_score, ...},
                baseline_responses: {probe_id: response_string, ...},
                quality: {"kv": str, "la": int},
                neutral: {"kv": str, "la": int},
                context: {"kv": str, "la": int},
                degraded: bool,
            }
        """
        if not anchors:
            return
        slot_label = slot_label_from_path(
            self.cfg.get("llama_server_path") or "")
        if not slot_label:
            slot_label = "(unknown_server)"

        store = self.cfg.setdefault("nut_profiles", {})
        if not isinstance(store, dict):
            store = {}
            self.cfg["nut_profiles"] = store

        per_model = store.setdefault(filename, {})
        if not isinstance(per_model, dict):
            per_model = {}
            store[filename] = per_model

        # Stringify the score dict for JSON storage (tuple keys aren't JSONable)
        score_strings = {f"{kv}|LA{la}": s
                         for (kv, la), s in scores.items()}

        per_model[slot_label] = {
            "checked_at":         datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "build_path":         self.cfg.get("llama_server_path", ""),
            "scores":             score_strings,
            "baseline_responses": baseline_responses,
            "quality":            anchors.get("quality"),
            "neutral":            anchors.get("neutral"),
            "context":            anchors.get("context"),
            "degraded":           bool(anchors.get("degraded")),
        }
        save_config(self.cfg)

    def _load_nut_profile_for_current_combo(self):
        """Load the qnut profile for the current (model, server_slot).

        Called whenever the user changes the model selection OR switches
        the active server slot. Updates self._current_nut_profile and
        refreshes the Q/N/C mode buttons accordingly.

        If no profile exists for the current combination, the mode
        buttons are disabled. The KV/LA buttons themselves stay in their
        classical v0.47 appearance — qnut v0.50 no longer paints them
        with verdict colours.
        """
        # Identify the current combination
        if self._sel_model < 0 or self._sel_model >= len(self.models):
            filename = None
        else:
            filename = self.models[self._sel_model].filename

        slot_label = slot_label_from_path(
            self.cfg.get("llama_server_path") or "")

        store = self.cfg.get("nut_profiles") or {}
        profile = None
        if filename and slot_label and isinstance(store, dict):
            per_model = store.get(filename)
            if isinstance(per_model, dict):
                entry = per_model.get(slot_label)
                if isinstance(entry, dict):
                    profile = entry

        self._current_nut_profile = profile

        # Refresh the mode buttons (Q/N/C) — enabled iff profile exists
        if hasattr(self, "_mode_buttons") and self._mode_buttons:
            self._refresh_mode_button_state()

        # KV/LA buttons stay classical in v0.50; clear any leftover
        # verdict-colour state from earlier nut visualisations.
        self._current_nut_verdicts = {}
        self._current_nut_anchors = {}
        if hasattr(self, "_kv_buttons") and self._kv_buttons:
            self._refresh_kv_button_visuals()

    # Backward-compatibility shim — _select_cell still calls this name.
    def _load_nut_anchors_for_current_model(self, filename: Optional[str]):
        """Compatibility wrapper around the v0.50 combo loader."""
        self._load_nut_profile_for_current_combo()

    def _apply_nut_mode(self, mode_key: str):
        """Click handler for the Q / N / C mode buttons.

        Reads the (kv, la) tuple for the requested mode from the active
        profile, sets the KV/LA buttons visually, and starts the server
        with those settings.

        v0.50.1 behaviour:
          * No server running   → set KV/LA, then start server.
          * Server running with EXACTLY this anchor's settings → no-op,
            just log "already active". Avoids wasting ~10 s on a
            tear-down/restart that produces an identical server.
          * Server running with different settings → confirm dialog
            "Restart with these settings? [Yes/No]" (default Yes).
            On Yes, stop the server, wait for the port to free, start
            the new one. On No, leave KV/LA visually applied but do
            not touch the server.
        """
        if self._probe_in_progress:
            # Probe is currently overwriting _kv_var/_la_var per tuple;
            # a Q/N/C click would race against the worker.
            self._log("qnut: cannot apply mode while a probe run is in "
                      "progress.", "warn")
            return
        if not self._current_nut_profile:
            self._log("qnut: no profile for current model + server. "
                      "Run qnut first.", "warn")
            return
        anchor = self._current_nut_profile.get(mode_key)
        if not isinstance(anchor, dict):
            self._log(f"qnut: profile is missing the {mode_key} anchor.",
                      "warn")
            return
        kv = anchor.get("kv")
        la = anchor.get("la")
        if kv is None or la is None:
            self._log(f"qnut: {mode_key} anchor is incomplete "
                      f"(kv={kv}, la={la}).", "warn")
            return
        if kv not in KV_CACHE_OPTIONS:
            self._log(f"qnut: anchor KV {kv!r} is not in the current "
                      "KV_CACHE_OPTIONS — profile may be from an older "
                      "launcher version. Re-run qnut to refresh.", "warn")
            return

        # Identify which model + GPU we should launch on. Without a
        # selected model row we cannot start anything, so the mode-click
        # degrades to "set the buttons but skip the launch".
        if self._sel_model < 0 or self._sel_model >= len(self.models):
            self._log("qnut: anchor applied to KV/LA but no model is "
                      "selected — cannot start a server.", "warn")
            self._active_mode = mode_key
            self._select_kv(kv)
            self._select_la(la)
            self._refresh_mode_button_state()
            return

        target_filename = self.models[self._sel_model].filename
        target_gpu_key = self._running_gpu_key or "GPU 0"

        # No-op detection BEFORE we touch the KV/LA buttons. If the
        # currently-running server already has exactly this anchor's
        # settings, we don't restart — restarting would just waste
        # ~10 seconds tearing down and re-spawning an identical server.
        if (self.server_process
                and self.running_model == target_filename
                and self._running_gpu_key == target_gpu_key
                and self._running_kv == kv
                and self._running_la == la):
            # Just refresh the visual state and log
            self._active_mode = mode_key
            self._select_kv(kv)
            self._select_la(la)
            self._refresh_mode_button_state()
            self._log(f"qnut: mode {mode_key.upper()} already active "
                      f"({kv} + LA={la}), server already running.",
                      "info")
            return

        # If a different server is running, ask for confirmation before
        # restarting it with the new settings. Default Yes — most users
        # who click Q/N/C want to actually launch with those settings.
        if self.server_process:
            mode_label = {"quality": "Quality (Q)",
                          "neutral": "Neutral (N)",
                          "context": "Context (C)"}.get(mode_key, mode_key)
            proceed = messagebox.askyesno(
                "Restart Server",
                f"A llama-server is currently running with different "
                f"settings.\n\n"
                f"Restart it with the {mode_label} anchor settings?\n\n"
                f"  Model:  {target_filename}\n"
                f"  KV:     {kv}\n"
                f"  LA:     {la}\n",
                icon="question",
                default="yes",
            )
            if not proceed:
                self._log(f"qnut: {mode_key.upper()} mode click cancelled "
                          "by user — server left running.", "info")
                return

            # Apply the visuals NOW so the user sees the change while the
            # restart sequence runs in the background.
            self._active_mode = mode_key
            self._select_kv(kv)
            self._select_la(la)
            self._refresh_mode_button_state()

            self._log(f"qnut: restarting server with {mode_key.upper()} "
                      f"settings ({kv} + LA={la})\u2026", "info")

            # Stop the current server and schedule the start once the
            # port is free. We use _nut_schedule_restart to keep the
            # main thread responsive instead of blocking on a sleep loop.
            self._stop_server(silent=True)
            self._nut_pending_restart = (target_filename, target_gpu_key,
                                          mode_key, kv, la)
            self.after(300, self._nut_check_port_and_start)
            return

        # No server running — just apply and start.
        self._active_mode = mode_key
        self._select_kv(kv)
        self._select_la(la)
        self._refresh_mode_button_state()

        deg = "  [profile is degraded]" if self._current_nut_profile.get(
            "degraded") else ""
        self._log(f"qnut: applied {mode_key.upper()} mode \u2192 "
                  f"{kv} + LA={la}{deg} \u2014 starting server\u2026",
                  "info")
        self._start_server(target_filename, target_gpu_key)

    def _nut_check_port_and_start(self):
        """Polled handler that fires the deferred server start once the
        previous server has fully released its port.

        Used by _apply_nut_mode when the user accepts a Q/N/C-driven
        server restart. We can't just call _start_server directly after
        _stop_server because the OS may need a moment to free the port.
        Instead we poll every 300 ms (up to 10 s) until the port is
        free, then launch.
        """
        pending = getattr(self, "_nut_pending_restart", None)
        if pending is None:
            return  # cancelled or already started
        filename, gpu_key, mode_key, kv, la = pending

        # Defensive: if a probe started in the meantime, abandon the restart
        if self._probe_in_progress:
            self._nut_pending_restart = None
            self._log("qnut: deferred restart cancelled (probe started)",
                      "warn")
            return

        # If the old server is still around, wait another tick
        if self.server_process is not None:
            # Bound retries by tracking attempt count
            self._nut_restart_attempts = getattr(
                self, "_nut_restart_attempts", 0) + 1
            if self._nut_restart_attempts > 33:  # ~10 s at 300 ms
                self._log("qnut: timed out waiting for old server to stop",
                          "error")
                self._nut_pending_restart = None
                self._nut_restart_attempts = 0
                return
            self.after(300, self._nut_check_port_and_start)
            return

        # Old server is gone — fire the new one
        self._nut_pending_restart = None
        self._nut_restart_attempts = 0
        self._start_server(filename, gpu_key)

    def _refresh_mode_button_state(self):
        """Recompute the enabled/active visuals of the three mode buttons.

        Called when:
          - the user switches model or server slot (profile may change)
          - the user clicks a mode button (active mode changes)
          - the user manually clicks a KV or LA button (active mode
            gets cleared because the user has wandered off the profile)

        v0.50.2: also updates the tiny score labels above each button
        (e.g. "3/3", "2/3") so the user can see at a glance how faithful
        each mode is to the f16 baseline BEFORE clicking. Blank when no
        profile exists.
        """
        if not hasattr(self, "_mode_buttons") or not self._mode_buttons:
            return
        t = self.theme
        has_profile = bool(self._current_nut_profile)

        # Resolve the scores dict from the profile (string-keyed because
        # JSON doesn't support tuple keys). Key format: "<kv>|LA<la>".
        scores = {}
        if has_profile:
            scores = self._current_nut_profile.get("scores") or {}
            if not isinstance(scores, dict):
                scores = {}

        for mode_key, btn in self._mode_buttons.items():
            score_lbl = self._mode_score_labels.get(mode_key)

            if not has_profile:
                btn.configure_btn(color=t.border, state="disabled")
                if score_lbl is not None:
                    score_lbl.configure(text=" ", fg=t.fg_dim)
                continue

            anchor = self._current_nut_profile.get(mode_key)
            if not isinstance(anchor, dict) or not anchor.get("kv"):
                btn.configure_btn(color=t.border, state="disabled")
                if score_lbl is not None:
                    score_lbl.configure(text=" ", fg=t.fg_dim)
                continue

            is_active = (self._active_mode == mode_key)
            btn.configure_btn(
                color=ACCENT_TURBO if is_active else t.border,
                state="normal",
            )

            # Look up the score for this anchor's (kv, la) tuple. The
            # quality anchor (f16, LA=0) is always 3/3 by definition
            # even if it's not explicitly in the scores dict.
            if score_lbl is not None:
                kv = anchor["kv"]
                la = int(anchor["la"])
                score_key = f"{kv}|LA{la}"
                score = scores.get(score_key)
                if score is None and mode_key == "quality":
                    # f16 baseline is always 3/3 by definition
                    score = 3
                if score is None:
                    score_lbl.configure(text="?/3", fg=t.fg_dim)
                else:
                    # Color-code: 3/3 green, 2/3 amber, lower red
                    if score >= 3:
                        color = NUT_COLORS.get("good", t.green)
                    elif score == 2:
                        color = NUT_COLORS.get("suspect", "#d4a017")
                    else:
                        color = NUT_COLORS.get("rotten", t.red)
                    score_lbl.configure(text=f"{score}/3", fg=color)

    def _clear_active_mode_if_user_drifted(self):
        """Reset _active_mode when the user manually changes KV or LA.

        If the user picks a KV or LA value that does not match the
        currently engaged mode's anchor, the engagement is broken: the
        user has drifted into power-user territory. We clear the mode
        and refresh the buttons so none of Q/N/C is highlighted.
        """
        if self._active_mode is None or not self._current_nut_profile:
            return
        anchor = self._current_nut_profile.get(self._active_mode)
        if not isinstance(anchor, dict):
            self._active_mode = None
            self._refresh_mode_button_state()
            return
        if (anchor.get("kv") != self._kv_var.get()
                or anchor.get("la") != self._la_var.get()):
            self._active_mode = None
            self._refresh_mode_button_state()

    def _nut_prepopulate_phase1_rows(self, baseline_kv: str, baseline_la: int):
        """Build the initial set of dialog rows BEFORE phase 0 starts.

        Lists all 6 KV configs at LA=0 (the phase-0 baseline plus the
        five phase-1 sweep tuples) as pending rows, plus a single
        placeholder row for phase 2 that says "pending until phase 1
        results". This gives the user the full upfront picture of what
        the run will test, so they understand the size of the work
        before any tuple actually starts.

        The phase-2 placeholder is replaced later (when phase 1 has
        finished and we know which KVs scored >=2) by a call to
        _nut_prepopulate_phase2_rows.
        """
        if not self._nut_dialog or self._nut_rows_frame is None:
            return
        t = self.theme

        def _add_row(label_text: str, status_text: str,
                     status_color: str, label_extra: str = ""):
            row = tk.Frame(self._nut_rows_frame, bg=t.bg)
            row.pack(fill="x", pady=1)
            full_label = label_text + (f"  {label_extra}" if label_extra else "")
            tk.Label(row, text=full_label, font=FONT_SMALL,
                     bg=t.bg, fg=t.fg, width=32, anchor="w").pack(side="left")
            status = tk.Label(row, text=status_text, font=FONT_SMALL,
                              bg=t.bg, fg=status_color, anchor="w")
            status.pack(side="left", fill="x", expand=True)
            self._nut_status_labels[label_text] = status

        # Phase 0 + Phase 1: all 6 KV configs at LA=0
        for kv in NUT_KV_ORDER:
            row_label = f"{kv}|LA={baseline_la}"
            extra = "(baseline)" if kv == baseline_kv else ""
            _add_row(row_label, "\u00b7 pending", t.fg_dim, extra)

        # Phase 2 placeholder. Stored under a fixed sentinel key so the
        # phase-2 prepopulator can find and remove it later.
        placeholder_row = tk.Frame(self._nut_rows_frame, bg=t.bg)
        placeholder_row.pack(fill="x", pady=(6, 1))
        tk.Label(placeholder_row,
                 text="Phase 2: LA sweep",
                 font=FONT_SMALL, bg=t.bg, fg=t.fg, width=32,
                 anchor="w").pack(side="left")
        placeholder_status = tk.Label(
            placeholder_row,
            text="\u00b7 candidates determined after phase 1",
            font=FONT_SMALL, bg=t.bg, fg=t.fg_dim, anchor="w")
        placeholder_status.pack(side="left", fill="x", expand=True)
        self._nut_phase2_placeholder = placeholder_row

    def _nut_prepopulate_phase2_rows(self,
                                     tuples: list):
        """Replace the phase-2 placeholder with real tuple rows.

        Called by the worker after phase 1 finishes, with the list of
        (kv, la) tuples that will actually be tested. Removes the
        placeholder row added in _nut_prepopulate_phase1_rows and adds
        one pending row per tuple.
        """
        if not self._nut_dialog or self._nut_rows_frame is None:
            return
        t = self.theme

        # Remove the placeholder if present
        ph = getattr(self, "_nut_phase2_placeholder", None)
        if ph is not None:
            try:
                ph.destroy()
            except tk.TclError:
                pass
            self._nut_phase2_placeholder = None

        # If phase 2 has nothing to do, leave a single explanatory row
        if not tuples:
            row = tk.Frame(self._nut_rows_frame, bg=t.bg)
            row.pack(fill="x", pady=(6, 1))
            tk.Label(row, text="Phase 2: (skipped)", font=FONT_SMALL,
                     bg=t.bg, fg=t.fg, width=32,
                     anchor="w").pack(side="left")
            tk.Label(row,
                     text="no KV reached score \u22652 in phase 1",
                     font=FONT_SMALL, bg=t.bg, fg=t.fg_dim,
                     anchor="w").pack(side="left", fill="x", expand=True)
            return

        # Add one pending row per (kv, la) tuple
        for (kv, la) in tuples:
            row_label = f"{kv}|LA={la}"
            if row_label in self._nut_status_labels:
                # Already exists from phase 1 (the f16+LA=0 baseline
                # case is the only overlap, but be defensive).
                continue
            row = tk.Frame(self._nut_rows_frame, bg=t.bg)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=row_label, font=FONT_SMALL,
                     bg=t.bg, fg=t.fg, width=32,
                     anchor="w").pack(side="left")
            status = tk.Label(row, text="\u00b7 pending", font=FONT_SMALL,
                              bg=t.bg, fg=t.fg_dim, anchor="w")
            status.pack(side="left", fill="x", expand=True)
            self._nut_status_labels[row_label] = status

    def _nut_ensure_row(self, row_label: str):
        """Worker -> main thread: ensure a status row exists in the dialog.

        v0.50: this is now a fallback for any tuple the worker touches
        that wasn't already pre-populated by _nut_prepopulate_phase1_rows
        or _nut_prepopulate_phase2_rows. In normal runs every row exists
        before the worker reaches it, so this is essentially a no-op.
        Kept as a safety net for unexpected code paths and for backward
        compatibility with the v0.48 dynamic-growth model.
        """
        if not self._nut_dialog:
            return
        if row_label in self._nut_status_labels:
            return  # already exists
        if not hasattr(self, "_nut_rows_frame") or self._nut_rows_frame is None:
            return
        t = self.theme
        row = tk.Frame(self._nut_rows_frame, bg=t.bg)
        row.pack(fill="x", pady=1)
        tk.Label(row, text=row_label, font=FONT_SMALL,
                 bg=t.bg, fg=t.fg, width=32, anchor="w").pack(side="left")
        status = tk.Label(row, text="\u2013 pending", font=FONT_SMALL,
                          bg=t.bg, fg=t.fg_dim, anchor="w")
        status.pack(side="left", fill="x", expand=True)
        self._nut_status_labels[row_label] = status

    # ─── /qnut ─────────────────────────────────────────────────────────────

    def _bench_run(self):
        """Run button: start benchmark OR start model server.

        Behavior depends on the Benchmark/Bench All checkboxes:
          - Bench All checked        → run full KV × LA benchmark matrix
          - Benchmark checked        → run single benchmark (baseline + selected KV)
          - Neither checked          → start the selected model on the selected GPU
                                       (same effect as double-clicking the GPU row)
        """
        if self._probe_in_progress:
            messagebox.showinfo(
                "Probe Model",
                "A probe run is in progress. Wait for it to finish "
                "(or cancel it via the Probe dialog) before running "
                "a benchmark or starting a server.",
            )
            return
        if self._bench_thread and self._bench_thread.is_alive():
            self._log("Benchmark already running.", "warn")
            return
        if self._sel_model < 0 or self._sel_model >= len(self.models):
            self._log("No model selected.", "warn")
            return

        fn = self.models[self._sel_model].filename
        card_data = self._model_cards.get(fn)
        if not card_data:
            return
        gpu_row = card_data["gpu_rows"][self._sel_row]
        gpu_key = gpu_row["key"]
        model = self.models[self._sel_model]
        is_cpu = (gpu_key == "CPU")

        # If no benchmark mode is active, treat Run as "start server"
        # and delegate to the same handler the double-click uses.
        bench_mode = self._bench_all_var.get() or self._bench_var.get()
        if not bench_mode:
            self._on_run_model(fn, gpu_key)
            return

        # Benchmark modes require no running server (shared VRAM / port conflicts)
        if self.server_process:
            self._log("Server is running — stop first (Escape)", "warn")
            return

        # Resolve GPU display name
        if is_cpu:
            gpu_display = f"CPU RAM ({self._cpu_ram_gb:.0f} GB)"
        else:
            gpu_display = gpu_key
            for g in self.gpus:
                if f"GPU {g.index}" == gpu_key:
                    gpu_name = g.name.replace("NVIDIA GeForce ", "")
                    gpu_display = f"{gpu_name} ({g.vram_mb // 1024} GB) — {gpu_key}"
                    break

        if self._bench_all_var.get() and self._bench_kv_set and self._bench_la_set:
            # Bench All mode — KV × LA × Depth matrix
            kv_list = [k for k in KV_CACHE_OPTIONS if k in self._bench_kv_set]
            la_list = sorted(self._bench_la_set)
            depth_list = list(BENCH_DEPTH_LIST)
            n_runs = len(kv_list) * len(la_list) * len(depth_list)

            # Per-run estimate grows with model size and with depth (pre-fill
            # dominates at d=32K for large models). Rough empirical rule:
            #   base = 20s + 2s per GB model size (covers load + warmup + short test)
            #   + per-depth: 0s @ d=0, 5–15s @ d=8K, 30–90s @ d=32K depending on model
            model_gb = model.size_gb or 8.0
            per_run_base = 20 + 2 * model_gb
            depth_cost = 0.0
            for d in depth_list:
                if d == 0:
                    depth_cost += 0
                elif d <= 8192:
                    depth_cost += 5 + 0.6 * model_gb
                else:  # 32K and above
                    depth_cost += 20 + 2.2 * model_gb
            # Average per-run seconds: base + (total depth cost / n_depths)
            avg_per_run = per_run_base + (depth_cost / max(1, len(depth_list)))
            total_sec = int(n_runs * avg_per_run)
            est_min = total_sec // 60
            est_sec = total_sec % 60

            depths_str = ", ".join(str(d) for d in depth_list)
            msg = (f"Run Benchmark ALL?\n\n"
                   f"Model: {model.filename}\n"
                   f"Target: {gpu_display}\n"
                   f"KV configs: {len(kv_list)}   LA modes: {len(la_list)}   "
                   f"Depths: {len(depth_list)} ({depths_str})\n"
                   f"Total runs: {n_runs}\n\n"
                   f"Estimated time: ~{est_min}m {est_sec}s")
            if not messagebox.askyesno("Benchmark All", msg):
                return
            self._bench_stop_event.clear()
            self._status_label.config(text="● Bench All...", fg=self.theme.yellow)
            self._run_btn.configure_btn(state="disabled")
            self._stop_btn.configure_btn(state="normal", color=self.theme.red)
            self._bench_thread = threading.Thread(
                target=self._run_benchmark_all,
                args=(model, gpu_key, kv_list, la_list, is_cpu),
                daemon=True)
            self._bench_thread.start()
        elif self._bench_var.get():
            # Single benchmark mode
            self._run_bench_for_selection()

    def _bench_stop(self):
        """Stop button: cancel running benchmark."""
        if self._bench_thread and self._bench_thread.is_alive():
            self._bench_stop_event.set()
            self._log("Stopping benchmark after current run...", "warn")
            self._stop_btn.configure_btn(state="disabled")

    def _update_bars_for_kv(self, kv_name: str):
        """Redraw all VRAM bars and fit labels for the newly selected KV config."""
        t = self.theme
        for fn, card_data in self._model_cards.items():
            for rd in card_data["gpu_rows"]:
                size_gb  = rd["size_gb"]
                total_gb = rd["total_gb"]
                eff_gb   = kv_effective_gb(size_gb, kv_name)
                kv_fits  = eff_gb <= total_gb * 0.97
                bg       = rd["card_bg"]

                rd["bar"].set_value(eff_gb, total_gb, canvas_bg=bg)
                rd["fit_text_lbl"].config(text=f"{eff_gb:.1f}/{total_gb:.0f} GB")
                rd["fit_icon_lbl"].config(
                    text=" ✓" if kv_fits else " ✗",
                    fg=t.green if kv_fits else t.red)
                rd["bar_tip"].update_text(
                    self._bar_tooltip_text(size_gb, eff_gb, total_gb))

    def _build_gpu_buttons(self):
        """No-op — GPU selection is now per-model via GPU rows in cards."""
        pass

    # ─── Model List ───────────────────────────────────────────────────────

    def _build_model_list(self):
        t = self.theme
        container = tk.Frame(self._paned, bg=t.bg)
        self._paned.add(container, stretch="always", minsize=150)

        hdr_frame = tk.Frame(container, bg=t.bg)
        hdr_frame.pack(fill="x")
        self._models_header = tk.Label(hdr_frame, text="Models",
                                        font=FONT_HEADER, bg=t.bg, fg=t.fg)
        self._models_header.pack(side="left")
        self._models_path_label = tk.Label(hdr_frame, text="", font=FONT_DIM,
                                            bg=t.bg, fg=t.fg_dim)
        self._models_path_label.pack(side="left", padx=(8, 0))

        tk.Frame(container, height=1, bg=t.border).pack(fill="x", pady=(4, 0))

        scroll_frame = tk.Frame(container, bg=t.bg)
        scroll_frame.pack(fill="both", expand=True)

        self._canvas = tk.Canvas(scroll_frame, bg=t.bg, highlightthickness=0, bd=0)
        self._scrollbar = tk.Scrollbar(scroll_frame, orient="vertical",
                                        command=self._canvas.yview)
        self._scrollbar.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)
        self._canvas.configure(yscrollcommand=self._scrollbar.set)

        self._inner_frame = tk.Frame(self._canvas, bg=t.bg)
        self._canvas_window = self._canvas.create_window((0, 0), window=self._inner_frame,
                                                          anchor="nw")
        self._inner_frame.bind("<Configure>",
                               lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>", self._on_canvas_resize)
        # Scope-checked mousewheel. bind_all catches the event globally,
        # but we only scroll the model list when the cursor is actually
        # inside our canvas hierarchy. Without this check, scrolling
        # anywhere in the window (e.g. over the log Text widget) would
        # also move the model list — confusing and wrong.
        def _on_mousewheel(event):
            w = event.widget
            # Walk up from the widget under the cursor. If we reach our
            # canvas, the cursor is over the model list → scroll it.
            # Otherwise let the event fall through to the default handler.
            while w is not None:
                if w is self._canvas:
                    self._canvas.yview_scroll(-1 * (event.delta // 120), "units")
                    return
                try:
                    w = w.master
                except AttributeError:
                    break
        self._canvas.bind_all("<MouseWheel>", _on_mousewheel)

    def _on_canvas_resize(self, event):
        self._canvas.itemconfig(self._canvas_window, width=event.width)

    # ─── Log Area ─────────────────────────────────────────────────────────

    def _build_log_area(self):
        t = self.theme
        log_frame = tk.Frame(self._paned, bg=t.bg)
        self._paned.add(log_frame, stretch="never", minsize=80)

        hdr = tk.Frame(log_frame, bg=t.bg)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Server Log", font=FONT_HEADER, bg=t.bg, fg=t.fg).pack(side="left")
        HoverButton(hdr, t, text="Clear", color=t.border, width=60, height=22,
                    command=self._clear_log).pack(side="right")

        self._log_text = tk.Text(log_frame, font=FONT_SMALL, bg=t.bg_secondary, fg=t.fg,
                                  relief="flat", bd=0, wrap="word", height=6, width=80,
                                  cursor="arrow", selectbackground=t.select_bg)
        self._log_text.pack(fill="both", expand=True, pady=(2, 0))
        self._log_text.tag_configure("time", foreground=t.fg_dim)
        self._log_text.tag_configure("info", foreground=t.fg)
        self._log_text.tag_configure("good", foreground=t.green)
        self._log_text.tag_configure("warn", foreground=t.yellow)
        self._log_text.tag_configure("error", foreground=t.red)
        self._log_text.config(state="disabled")

    # ─── Footer ───────────────────────────────────────────────────────────

    def _measure_label_group_width(self, labels, bold: bool = False,
                                   h_pad: int = 20) -> int:
        """Uniform width for a group of button labels.

        Sizes to the widest label in the group using tkfont.measure() so
        the result honors Tk's current scaling factor. h_pad is the total
        horizontal padding (split left+right inside the button canvas).
        bold controls whether we measure against Consolas bold or regular
        — must match the font actually passed to HoverButton.
        """
        import tkinter.font as tkfont
        f = tkfont.Font(family=_MONO, size=10,
                        weight="bold" if bold else "normal")
        if not labels:
            return 80
        return max(f.measure(l) for l in labels) + h_pad

    def _build_footer(self):
        t = self.theme
        # side="bottom" anchors the footer to the window bottom edge
        # and reserves its height BEFORE the paned window expands into the
        # remaining space. This prevents the footer from being clipped when
        # the user shrinks the window or when the model list grows.
        bar = ttk.Frame(self)
        bar.pack(side="bottom", fill="x", padx=16, pady=(6, 8))
        tk.Frame(self, height=1, bg=t.border).pack(side="bottom", fill="x")

        # v0.54: three independent button groups, each with its own uniform
        # width sized to the widest label in that group. This keeps short
        # labels ("About") from being stretched to match long ones
        # ("Update Binaries"). FONT_SMALL (non-bold) gives a lighter,
        # more filigree appearance than the previous FONT_SMALL_B.
        _BTN_H = 28
        _H_PAD = 20   # ~1 char padding each side at Consolas 10

        # Group A: file-ops (left)
        group_a_w = self._measure_label_group_width(
            ["Rescan Models", "Update Binaries"], bold=False, h_pad=_H_PAD)
        HoverButton(bar, t, text="Rescan Models", color=ACCENT_SOFT,
                    width=group_a_w, height=_BTN_H, font=FONT_SMALL,
                    command=self._rescan_models).pack(side="left", padx=2)
        HoverButton(bar, t, text="Update Binaries", color=ACCENT_SOFT,
                    width=group_a_w, height=_BTN_H, font=FONT_SMALL,
                    command=self._update_binaries).pack(side="left", padx=2)

        # Group B: engine slot buttons (middle, dynamic).
        # Sized and spacing handled in _refresh_slot_buttons. Extra left
        # margin separates the slot group from Update Binaries.
        self._slot_bar = tk.Frame(bar, bg=t.bg)
        self._slot_bar.pack(side="left", padx=(12, 0))
        self._slot_buttons: list = []

        # Group C: settings (right, packed right-to-left so About lands
        # at the far right). Save Results is standalone with its own width.
        group_c_w = self._measure_label_group_width(
            ["About", "Paths"], bold=False, h_pad=_H_PAD)
        HoverButton(bar, t, text="About", color=ACCENT_SOFT,
                    width=group_c_w, height=_BTN_H, font=FONT_SMALL,
                    command=self._show_about).pack(side="right", padx=2)
        HoverButton(bar, t, text="Paths", color=ACCENT_SOFT,
                    width=group_c_w, height=_BTN_H, font=FONT_SMALL,
                    command=self._show_paths_dialog).pack(side="right", padx=2)

        # v0.56 — Autoload toggle. Sits just left of Paths in the footer.
        # Three visual states, all rendered with the same label "Autoload"
        # so the button's width stays constant across transitions — state
        # is encoded in color + corner_glyph only, never in text length:
        #   locked  = install_verified==False  -> grey, disabled, "🔒" glyph
        #   OFF     = verified, autoload==False -> ACCENT_SOFT, no glyph
        #   ON      = autoload==True            -> green, "✓" glyph
        # _refresh_autoload_btn() applies the initial state and is also
        # called whenever install_verified flips (first "listening") or
        # the user clicks the button.
        al_w = self._measure_label_group_width(
            ["Autoload"], bold=False, h_pad=_H_PAD)
        self._autoload_btn = HoverButton(
            bar, t, text="Autoload", color=t.border,
            width=al_w, height=_BTN_H, font=FONT_SMALL,
            command=self._on_autoload_click)
        self._autoload_btn.pack(side="right", padx=2)
        self._autoload_tooltip = ToolTip(self._autoload_btn, "", t)
        self._refresh_autoload_btn()

        save_w = self._measure_label_group_width(
            ["💾  Save Results..."], bold=False, h_pad=_H_PAD)
        self._save_bench_btn = HoverButton(bar, t, text="💾  Save Results...",
                                            color=t.border, width=save_w,
                                            height=_BTN_H, font=FONT_SMALL,
                                            command=self._save_pending_bench)
        self._save_bench_btn.configure_btn(state="disabled")
        self._save_bench_btn.pack(side="right", padx=(2, 12))

    # ─── Engine Quick-Switch Slot Buttons ────────────────────────────

    def _refresh_slot_buttons(self):
        """Rebuild the quick-switch slot buttons from self.cfg['server_slots'].

        Called after config changes (Paths dialog save) and after a slot click
        (to update the active highlight). Only non-empty slots get a button.
        The button whose path matches the currently active llama_server_path
        is highlighted with the accent color; all others use ACCENT_SOFT.
        """
        t = self.theme
        # Tear down existing buttons
        for btn in getattr(self, "_slot_buttons", []):
            try:
                btn.destroy()
            except Exception:
                pass
        self._slot_buttons = []

        slots = self.cfg.get("server_slots") or []
        active_path = (self.cfg.get("llama_server_path") or "").strip()
        active_norm = os.path.normcase(os.path.normpath(active_path)) if active_path else ""

        _BTN_H = 28
        # v0.54: slot buttons share a uniform width computed from the
        # widest slot label in the current config. Wider inter-button
        # spacing (padx=6) makes the engine names breathe a bit.
        slot_labels = []
        for sp in slots:
            if sp and sp.strip():
                lbl = slot_label_from_path(sp)
                if lbl:
                    slot_labels.append(lbl)
        btn_w = self._measure_label_group_width(
            slot_labels, bold=False, h_pad=20) if slot_labels else 90

        for slot_path in slots:
            if not slot_path or not slot_path.strip():
                continue
            label = slot_label_from_path(slot_path)
            if not label:
                continue

            # Highlight active slot
            slot_norm = os.path.normcase(os.path.normpath(slot_path))
            is_active = (slot_norm == active_norm) and bool(active_norm)
            color = t.accent if is_active else ACCENT_SOFT

            # Missing-file indicator: still show the button, but in muted color
            # so the user knows the slot is broken. Click will log an error.
            if not os.path.isfile(slot_path):
                color = t.border  # muted/disabled look

            btn = HoverButton(self._slot_bar, t, text=label, color=color,
                              width=btn_w, height=_BTN_H, font=FONT_SMALL,
                              command=lambda p=slot_path: self._on_slot_click(p))
            btn.pack(side="left", padx=6)
            ToolTip(btn, slot_path, theme=t)
            self._slot_buttons.append(btn)

    def _on_slot_click(self, slot_path: str):
        """Switch the active llama-server.exe to the given slot path.

        Refuses to switch while a server is running (must be stopped first).
        Refreshes header info, DLL check, and model cards via _do_initial_scan.
        """
        if self.server_process:
            self._log("Cannot switch engine: server is running. Stop it first.", "error")
            return
        if not slot_path or not os.path.isfile(slot_path):
            self._log(f"Slot path not found: {slot_path}", "error")
            return
        # Already active? No-op.
        current = (self.cfg.get("llama_server_path") or "").strip()
        if current and os.path.normcase(os.path.normpath(current)) == \
                      os.path.normcase(os.path.normpath(slot_path)):
            self._log(f"Engine already active: {slot_label_from_path(slot_path)}", "info")
            return

        self.cfg["llama_server_path"] = slot_path
        save_config(self.cfg)
        label = slot_label_from_path(slot_path)
        self._log(f"Switched engine → {label}", "good")
        # Full refresh: re-detect CUDA build tag, DLL check, header, model scan,
        # and slot highlighting. GPU list is re-probed too (cheap on modern drivers).
        self._do_initial_scan()
        # qnut v0.50: re-evaluate the Q/N/C mode buttons against the
        # new (model, server_slot) combination. If the new combination
        # has a stored profile, the buttons become active; otherwise
        # they go grey with the "run qnut to enable" hint.
        self._load_nut_profile_for_current_combo()

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION: Model Cards
    # ═══════════════════════════════════════════════════════════════════════

    def _rebuild_model_cards(self):
        t = self.theme
        HOVER_BG = "#252830" if self.is_dark else "#e8f0fe"
        SEL_BORDER = t.accent

        for w in self._inner_frame.winfo_children():
            w.destroy()
        self._model_cards.clear()
        self._sel_model = -1
        self._sel_row = 0

        if not self.models:
            tk.Label(self._inner_frame, text="No GGUF models found.",
                     font=FONT_BODY, bg=t.bg, fg=t.fg_dim).pack(pady=20)
            return

        paths = [p for p in (self.cfg.get("llm_models_paths") or []) if p]
        if len(paths) == 0:
            label_text = "(no directory configured)"
        elif len(paths) == 1:
            label_text = f"({paths[0]})"
        else:
            label_text = f"({' • '.join(paths)})"
        self._models_path_label.config(text=label_text)

        for i, model in enumerate(self.models):
            bg = t.bg_secondary if i % 2 == 0 else t.bg
            card = tk.Frame(self._inner_frame, bg=bg)
            card.pack(fill="x", pady=1)

            hdr = tk.Frame(card, bg=bg)
            hdr.pack(fill="x", padx=10, pady=(6, 2))
            # Warning marker for paths with commas: llama-bench will refuse to
            # load these (its argparse truncates at the first comma). Surfaced
            # here so the user understands why the bench would FAIL before
            # they even click. Tooltip explains the cause.
            if model.has_unsafe_path:
                warn_lbl = tk.Label(hdr, text="⚠", font=FONT_BODY_B,
                                     bg=bg, fg=t.red)
                warn_lbl.pack(side="left", padx=(0, 4))
                ToolTip(warn_lbl,
                        "Path contains a comma — llama-bench will refuse this model.\n"
                        "llama-bench's argument parser truncates paths at the first\n"
                        "comma even when properly quoted (commas are used as list\n"
                        "separators for parameters like -d 0,8192,32768).\n\n"
                        "llama-server may still work with this model, but bench\n"
                        "runs will fail silently. Move the model to a path without\n"
                        "commas to enable benchmarking.\n\n"
                        f"Path: {model.path}", t)
            tk.Label(hdr, text=model.filename, font=FONT_BODY_B,
                     bg=bg, fg=t.fg).pack(side="left")
            tk.Label(hdr, text=f"{model.size_gb} GB", font=FONT_BODY,
                     bg=bg, fg=t.fg_secondary).pack(side="left", padx=(12, 0))

            gpu_rows = []

            for gpu in self.gpus:
                gpu_key = f"GPU {gpu.index}"
                vram_gb = gpu.vram_mb / 1024
                fits = model.size_gb <= vram_gb * 0.95
                row_data = self._create_gpu_row(
                    card, bg, HOVER_BG, model, i, len(gpu_rows), gpu_key,
                    gpu.name.replace("NVIDIA GeForce ", "").replace("AMD Radeon ", ""),
                    {"nvidia": t.green, "amd": t.red, "intel": t.accent}.get(gpu.vendor, t.fg_dim),
                    model.size_gb, vram_gb, fits)
                gpu_rows.append(row_data)

            if self._cpu_ram_gb > 0:
                cpu_fits = model.size_gb <= self._cpu_ram_gb * 0.7
                row_data = self._create_gpu_row(
                    card, bg, HOVER_BG, model, i, len(gpu_rows), "CPU",
                    "CPU RAM", t.green,
                    model.size_gb, self._cpu_ram_gb, cpu_fits)
                gpu_rows.append(row_data)

            tk.Frame(card, bg=bg, height=4).pack(fill="x")

            self._model_cards[model.filename] = {
                "card": card, "bg": bg, "index": i,
                "gpu_rows": gpu_rows,
            }

        if self.models:
            # Defer the initial selection until after the first idle pass
            # so the geometry manager has a chance to lay out _inner_frame
            # and the <Configure> binding has set the canvas scrollregion.
            # Calling _select_cell(0, 0) inline here used to leave a large
            # empty band above the first card on first display because
            # _inner_frame.winfo_height() was still 1 at that moment.
            self.after_idle(lambda: self._select_cell(0, 0))

        self.bind("<Up>", self._on_key_up)
        self.bind("<Down>", self._on_key_down)
        self.bind("<Left>", self._on_key_left)
        self.bind("<Right>", self._on_key_right)
        self.bind("<Return>", self._on_key_enter)
        self.bind("<Escape>", lambda e: self._stop_server() if self.server_process else None)

    def _create_gpu_row(self, card, card_bg, hover_bg, model, model_idx, row_idx,
                         gpu_key, gpu_label, label_color, size_gb, total_gb, fits):
        """Create a single interactive GPU row inside a model card."""
        t = self.theme
        row = tk.Frame(card, bg=card_bg, cursor="hand2",
                       highlightthickness=1, highlightbackground=card_bg)
        row.pack(fill="x", padx=6, pady=0)

        indicator = tk.Label(row, text="  ", font=FONT_SMALL_B, bg=card_bg, fg=t.green, width=2)
        indicator.pack(side="left")

        tk.Label(row, text=gpu_label, font=FONT_SMALL, bg=card_bg, fg=label_color,
                 width=12, anchor="w", cursor="hand2").pack(side="left")

        bar = BrailleBar(row, t, width=200, height=18, chars=16, canvas_bg=card_bg)
        bar.pack(side="left", padx=(4, 0))
        eff_gb = kv_effective_gb(size_gb, self._kv_var.get())
        bar.set_value(eff_gb, total_gb, canvas_bg=card_bg)

        kv_fits = eff_gb <= total_gb * 0.97
        fit_text_lbl = tk.Label(row, text=f"{eff_gb:.1f}/{total_gb:.0f} GB",
                                 font=FONT_SMALL, bg=card_bg, fg=t.fg_secondary, cursor="hand2")
        fit_text_lbl.pack(side="left", padx=(6, 0))
        fit_icon_lbl = tk.Label(row, text=" ✓" if kv_fits else " ✗",
                                 font=FONT_SMALL_B, bg=card_bg,
                                 fg=t.green if kv_fits else t.red, cursor="hand2")
        fit_icon_lbl.pack(side="left")

        bar_tip = ToolTip(bar, self._bar_tooltip_text(size_gb, eff_gb, total_gb), t)

        def _enter(e):
            if not (model.filename == self.running_model and gpu_key == self._running_gpu_key):
                cur_eff = kv_effective_gb(size_gb, self._kv_var.get())
                for child in row.winfo_children():
                    try: child.configure(bg=hover_bg)
                    except: pass
                row.configure(bg=hover_bg)
                bar.set_value(cur_eff, total_gb, canvas_bg=hover_bg)

        def _leave(e):
            for child in row.winfo_children():
                try: child.configure(bg=card_bg)
                except: pass
            row.configure(bg=card_bg)
            cur_eff = kv_effective_gb(size_gb, self._kv_var.get())
            bar.set_value(cur_eff, total_gb, canvas_bg=card_bg)

        def _click(e):
            self._select_cell(model_idx, row_idx)

        def _dblclick(e):
            self._select_cell(model_idx, row_idx)
            if self._bench_var.get():
                self._bench_run()
            elif model.filename == self.running_model and gpu_key == self._running_gpu_key:
                self._stop_server()
            else:
                self._launch_selected()

        def _bind_recursive(w):
            w.bind("<Enter>", _enter)
            w.bind("<Leave>", _leave)
            w.bind("<Button-1>", _click)
            w.bind("<Double-Button-1>", _dblclick)
            for child in w.winfo_children():
                _bind_recursive(child)

        _bind_recursive(row)

        return {
            "frame": row, "key": gpu_key, "indicator": indicator,
            "card_bg": card_bg, "bar": bar, "bar_tip": bar_tip,
            "size_gb": size_gb, "total_gb": total_gb,
            "fit_text_lbl": fit_text_lbl, "fit_icon_lbl": fit_icon_lbl,
        }

    # ─── 2D Cell Selection ────────────────────────────────────────────────

    def _select_cell(self, model_idx: int, row_idx: int, scroll: str = "none"):
        """Select a specific GPU row within a model card.

        scroll:
          "none"  - never move the canvas view. Used for mouse clicks
                    and initial auto-selection — the user scrolls
                    manually with the mouse wheel when they want to see
                    something off-screen.
          "into"  - only scroll if the selection ended up off-screen,
                    and then only the minimum amount to bring it back
                    into view. Used for keyboard Up/Down/Left/Right so
                    the selection stays visible as you walk through
                    rows past the viewport edges.
        """
        t = self.theme
        if model_idx < 0 or model_idx >= len(self.models):
            return

        for fn, card_data in self._model_cards.items():
            for rd in card_data["gpu_rows"]:
                rd["frame"].config(highlightbackground=rd["card_bg"])

        fn = self.models[model_idx].filename
        card_data = self._model_cards.get(fn)
        if not card_data:
            return
        max_row = len(card_data["gpu_rows"]) - 1
        row_idx = max(0, min(row_idx, max_row))

        self._sel_model = model_idx
        self._sel_row = row_idx

        rd = card_data["gpu_rows"][row_idx]
        rd["frame"].config(highlightbackground=t.accent)

        # v0.56 — refresh the inline "No Thinking on reasoning model"
        # warning whenever the selected model changes. No-op when the
        # warning is already empty (non-reasoning model) or No Thinking
        # is off.
        try:
            self._refresh_thinking_warning()
        except Exception:
            # Defensive: early calls during widget build may fire before
            # the warning label exists. The method's own hasattr guard
            # handles that, but wrap anyway to be safe across reload.
            pass

        if scroll == "into":
            # Ensure geometry is up-to-date — winfo_height() returns 1
            # on freshly built widgets until the idle loop runs.
            self._inner_frame.update_idletasks()
            bbox = self._canvas.bbox("all")
            if bbox:
                self._canvas.configure(scrollregion=bbox)

            frame_y = rd["frame"].winfo_y() + card_data["card"].winfo_y()
            frame_h = rd["frame"].winfo_height()
            canvas_h = self._canvas.winfo_height()
            inner_h = max(1, self._inner_frame.winfo_height())

            # Current visible window in scrollregion-pixel coordinates.
            yview_top, yview_bot = self._canvas.yview()
            view_top_px = yview_top * inner_h
            view_bot_px = yview_bot * inner_h

            sel_top = frame_y
            sel_bot = frame_y + frame_h
            margin = 6

            # Minimum-motion scrolling: only move if off-screen, and
            # only far enough to expose the selected row with a small
            # margin. Never center, never third-of-viewport.
            if sel_top < view_top_px:
                new_top = max(0, sel_top - margin)
                self._canvas.yview_moveto(new_top / inner_h)
            elif sel_bot > view_bot_px:
                new_top = min(inner_h - canvas_h, sel_bot + margin - canvas_h)
                self._canvas.yview_moveto(new_top / inner_h)
            # else: already fully visible — leave the view untouched.

        # qnut (v0.48): refresh KV button visuals from this model's
        # cached nut data. If no data exists, the buttons fall back to
        # their classical v0.47 appearance automatically.
        self._load_nut_anchors_for_current_model(fn)

    def _launch_selected(self):
        """Launch the currently selected model on the selected GPU."""
        if self._probe_in_progress:
            messagebox.showinfo(
                "Probe Model",
                "A probe run is in progress. Wait for it to finish "
                "(or cancel it via the Probe dialog) before launching "
                "a model.",
            )
            return
        if self._sel_model < 0 or self._sel_model >= len(self.models):
            return
        fn = self.models[self._sel_model].filename
        card_data = self._model_cards.get(fn)
        if not card_data:
            return
        gpu_row = card_data["gpu_rows"][self._sel_row]
        gpu_key = gpu_row["key"]
        self._on_run_model(fn, gpu_key)

    # ─── Keyboard Navigation ─────────────────────────────────────────────
    # Up/Down navigate linearly through ALL rows of the flat list:
    # RTX5090 → RTX4090 → CPU RAM → (next model) RTX5090 → ...
    # Without this, pressing Down from row 0 of model A would jump to
    # row 0 of model B, skipping the CPU RAM row entirely.
    # Left/Right still move only within the current model (useful when
    # you want to pick a different GPU without leaving the current model).
    # Keyboard nav is the ONLY path that passes scroll="into" — mouse
    # clicks never move the view.

    def _on_key_up(self, event):
        """Move one row up. At the top row of a model, jump to the last
        row of the previous model."""
        m, r = self._sel_model, self._sel_row
        if m < 0:
            return
        if r > 0:
            self._select_cell(m, r - 1, scroll="into")
        elif m > 0:
            prev_fn = self.models[m - 1].filename
            prev_card = self._model_cards.get(prev_fn)
            if prev_card:
                last_row = len(prev_card["gpu_rows"]) - 1
                self._select_cell(m - 1, last_row, scroll="into")

    def _on_key_down(self, event):
        """Move one row down. At the bottom row of a model, jump to
        row 0 of the next model."""
        m, r = self._sel_model, self._sel_row
        if m < 0:
            return
        fn = self.models[m].filename
        card = self._model_cards.get(fn)
        if not card:
            return
        max_row = len(card["gpu_rows"]) - 1
        if r < max_row:
            self._select_cell(m, r + 1, scroll="into")
        elif m < len(self.models) - 1:
            self._select_cell(m + 1, 0, scroll="into")

    def _on_key_left(self, event):
        """Move one row up within the current model (no crossing)."""
        if self._sel_row > 0:
            self._select_cell(self._sel_model, self._sel_row - 1, scroll="into")

    def _on_key_right(self, event):
        """Move one row down within the current model (no crossing)."""
        fn = self.models[self._sel_model].filename if self._sel_model >= 0 else None
        card_data = self._model_cards.get(fn) if fn else None
        max_row = len(card_data["gpu_rows"]) - 1 if card_data else 0
        if self._sel_row < max_row:
            self._select_cell(self._sel_model, self._sel_row + 1, scroll="into")

    def _on_key_enter(self, event):
        if self._sel_model < 0:
            return
        fn = self.models[self._sel_model].filename
        card_data = self._model_cards.get(fn)
        if not card_data:
            return
        gpu_key = card_data["gpu_rows"][self._sel_row]["key"]
        if self._bench_var.get():
            self._bench_run()
        elif fn == self.running_model and gpu_key == self._running_gpu_key:
            self._stop_server()
        else:
            self._launch_selected()

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION: Benchmark
    # ═══════════════════════════════════════════════════════════════════════

    def _run_bench_for_selection(self):
        """Run benchmark on the currently selected model + GPU row."""
        if self.server_process:
            self._log("Server is running — stop first (Escape)", "warn")
            return
        if self._sel_model < 0 or self._sel_model >= len(self.models):
            return
        fn = self.models[self._sel_model].filename
        card_data = self._model_cards.get(fn)
        if not card_data:
            return
        gpu_row = card_data["gpu_rows"][self._sel_row]
        gpu_key = gpu_row["key"]
        model = self.models[self._sel_model]

        is_cpu = (gpu_key == "CPU")
        kv_name = self._kv_var.get()
        if is_cpu:
            kv_name = "f16 (default)"
            target_label = f"CPU RAM ({self._cpu_ram_gb:.0f} GB)"
        else:
            target_label = gpu_key
            for g in self.gpus:
                if f"GPU {g.index}" == gpu_key:
                    gpu_name = g.name.replace("NVIDIA GeForce ", "")
                    target_label = f"{gpu_name} ({g.vram_mb // 1024} GB) — {gpu_key}"
                    break

        msg = (f"Run benchmark?\n\n"
               f"Model: {model.filename}\n"
               f"Target: {target_label}\n"
               f"KV-Cache: {kv_name}\n\n"
               f"Runs f16 baseline + selected KV config.\n"
               f"Estimated time: ~90s")
        if not messagebox.askyesno("Benchmark", msg):
            return

        self._log(f"Benchmark: {model.filename} on {target_label} ({kv_name})", "info")
        self._status_label.config(text="● Benchmarking...", fg=self.theme.yellow)

        thread = threading.Thread(target=self._run_benchmark,
                                   args=(model, gpu_key, kv_name, is_cpu),
                                   daemon=True)
        thread.start()

    def _run_benchmark_all(self, model: ModelInfo, gpu_key: str,
                            kv_list: list, la_list: list, is_cpu: bool):
        """Background thread: run llama-bench for KV × LA × Depth matrix.

        Added Depth dimension. The outer loop is LA (groups runs for
        log readability as before), the middle loop is depth (reproduces
        Madreag's short/medium/long-context methodology), and the inner loop
        is KV. Result keys use ``{kv}|LA=xx|d=N`` so the file writer can
        group by LA and sub-group by depth.
        """
        server_exe = self.cfg.get("llama_server_path", "")
        server_dir = os.path.dirname(server_exe)
        bench_exe = os.path.join(server_dir, "llama-bench.exe")

        if not os.path.isfile(bench_exe):
            self.after(0, self._log, f"llama-bench.exe not found in {server_dir}", "error")
            self.after(0, self._bench_all_finished, None)
            return

        ngl = "0" if is_cpu else "99"
        results = {}
        step = 0
        depth_list = list(BENCH_DEPTH_LIST)
        total = len(kv_list) * len(la_list) * len(depth_list)

        # Resolve GPU display name
        if is_cpu:
            gpu_display = "CPU"
        else:
            gpu_display = gpu_key
            for g in self.gpus:
                if f"GPU {g.index}" == gpu_key:
                    gpu_display = g.name.replace("NVIDIA GeForce ", "")
                    break

        for la_mode in la_list:
            if self._bench_stop_event.is_set():
                self.after(0, self._log, "Benchmark stopped by user.", "warn")
                break

            la_tag = f"LA={la_mode}" if la_mode > 0 else "LA=off"
            self.after(0, self._log, f"\n  ═══ {la_tag} — {gpu_display} ═══", "info")

            env = os.environ.copy()
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            if not is_cpu:
                env["CUDA_VISIBLE_DEVICES"] = gpu_key.split(" ")[1]
            if la_mode > 0:
                env["TURBO_LAYER_ADAPTIVE"] = str(la_mode)
            elif "TURBO_LAYER_ADAPTIVE" in env:
                del env["TURBO_LAYER_ADAPTIVE"]

            for depth in depth_list:
                if self._bench_stop_event.is_set():
                    break

                depth_tag = f"d={depth}" if depth > 0 else "d=0"
                self.after(0, self._log,
                           f"  ─── {depth_tag} ───", "info")

                for kv_name in kv_list:
                    if self._bench_stop_event.is_set():
                        break
                    step += 1
                    kv = KV_CACHE_OPTIONS.get(kv_name, {})
                    ctk = kv.get("ctk")
                    ctv = kv.get("ctv")

                    self.after(0, self._log,
                               f"  ▸ [{step}/{total}] {kv_name} ({la_tag}, {depth_tag})...", "info")
                    self.after(0, lambda s=step, t=total:
                               self._status_label.config(
                                   text=f"● Bench All [{s}/{t}]...",
                                   fg=self.theme.yellow))

                    result = self._exec_bench(bench_exe, model.path, ngl,
                                              ctk, ctv, env, depth=depth)
                    if result:
                        key = f"{kv_name}|{la_tag}|{depth_tag}"
                        results[key] = result
                        pp_key = next((k for k in result if k.startswith("pp")), None)
                        tg_key = next((k for k in result if k.startswith("tg")), None)
                        pp = result.get(pp_key, "?") if pp_key else "?"
                        tg = result.get(tg_key, "?") if tg_key else "?"
                        self.after(0, self._log,
                                   f"    {pp_key or 'pp'}={pp}, "
                                   f"{tg_key or 'tg'}={tg} t/s", "good")
                    else:
                        self.after(0, self._log, f"    FAILED", "error")

        if results:
            gpu_label = "CPU" if is_cpu else gpu_key
            for g in self.gpus:
                if f"GPU {g.index}" == gpu_key:
                    gpu_label = g.name.replace("NVIDIA GeForce ", "")
                    break
            self._pending_bench = (model, gpu_label, results, "ALL")
        else:
            self._pending_bench = None

        self.after(0, self._bench_all_finished, results)

    def _bench_all_finished(self, results):
        """Called on main thread when Bench All completes."""
        t = self.theme
        self._run_btn.configure_btn(state="normal")
        self._stop_btn.configure_btn(state="disabled", color=t.accent)
        self._bench_stop_event.clear()
        self._bench_finished(results)

    def _run_benchmark(self, model: ModelInfo, gpu_key: str, kv_name: str, is_cpu: bool):
        """Background thread: run llama-bench twice (f16 baseline + selected KV)."""
        server_exe = self.cfg.get("llama_server_path", "")
        server_dir = os.path.dirname(server_exe)
        bench_exe = os.path.join(server_dir, "llama-bench.exe")

        if not os.path.isfile(bench_exe):
            self.after(0, self._log, f"llama-bench.exe not found in {server_dir}", "error")
            self.after(0, self._bench_finished, None)
            return

        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if not is_cpu:
            env["CUDA_VISIBLE_DEVICES"] = gpu_key.split(" ")[1]

        ngl = "0" if is_cpu else "99"
        results = {}

        # Resolve GPU display name
        if is_cpu:
            gpu_display = "CPU"
        else:
            gpu_display = gpu_key
            for g in self.gpus:
                if f"GPU {g.index}" == gpu_key:
                    gpu_display = g.name.replace("NVIDIA GeForce ", "")
                    break

        self.after(0, self._log, f"  ▸ f16 Baseline ({gpu_display})...", "info")
        baseline = self._exec_bench(bench_exe, model.path, ngl, None, None, env)
        if baseline:
            results["f16"] = baseline
            pp_key = next((k for k in baseline if k.startswith("pp")), None)
            tg_key = next((k for k in baseline if k.startswith("tg")), None)
            pp = baseline.get(pp_key, "?") if pp_key else "?"
            tg = baseline.get(tg_key, "?") if tg_key else "?"
            self.after(0, self._log, f"    f16: {pp_key or 'pp'}={pp} t/s, {tg_key or 'tg'}={tg} t/s", "good")

        kv = KV_CACHE_OPTIONS.get(kv_name, {})
        ctk = kv.get("ctk")
        ctv = kv.get("ctv")
        if ctk or ctv:
            short = kv_name.replace(" ", "")
            self.after(0, self._log, f"  ▸ {kv_name}...", "info")
            turbo = self._exec_bench(bench_exe, model.path, ngl, ctk, ctv, env)
            if turbo:
                results[kv_name] = turbo
                pp_key = next((k for k in turbo if k.startswith("pp")), None)
                tg_key = next((k for k in turbo if k.startswith("tg")), None)
                pp = turbo.get(pp_key, "?") if pp_key else "?"
                tg = turbo.get(tg_key, "?") if tg_key else "?"
                self.after(0, self._log, f"    {kv_name}: {pp_key or 'pp'}={pp} t/s, {tg_key or 'tg'}={tg} t/s", "good")

        if results:
            gpu_label = "CPU" if is_cpu else gpu_key
            for g in self.gpus:
                if f"GPU {g.index}" == gpu_key:
                    gpu_label = g.name.replace("NVIDIA GeForce ", "")
                    break
            self._pending_bench = (model, gpu_label, results, kv_name)
        else:
            self._pending_bench = None

        self.after(0, self._bench_finished, results)

    def _get_bench_timeout(self) -> int:
        """Return the per-run benchmark timeout in seconds, validated.

        Reads from the GUI Timeout field. Falls back to 90 on any
        invalid input (non-integer, out of range) and logs a warning so the
        user can see why their custom value didn't take effect. Range is
        clamped to [30, 1800] seconds — 30s is the floor below which even
        small models can't load + run a single test, 1800s (30 min) is a
        safety net against typos like "9000".
        """
        DEFAULT = 90
        MIN_S, MAX_S = 30, 1800
        raw = (self._bench_timeout_var.get() or "").strip() if hasattr(self, "_bench_timeout_var") else ""
        if not raw:
            return DEFAULT
        try:
            val = int(raw)
        except ValueError:
            self.after(0, self._log,
                       f"Invalid Timeout value {raw!r} (expected integer 30–1800), using {DEFAULT}s",
                       "warn")
            return DEFAULT
        if val < MIN_S or val > MAX_S:
            self.after(0, self._log,
                       f"Timeout {val}s out of range [{MIN_S}, {MAX_S}], using {DEFAULT}s",
                       "warn")
            return DEFAULT
        return val

    def _exec_bench(self, bench_exe: str, model_path: str, ngl: str,
                     ctk: Optional[str], ctv: Optional[str], env: dict,
                     depth: int = 0) -> Optional[dict]:
        """Execute llama-bench and parse output. Returns {pp512: str, tg128: str} or None.

        ``depth`` is passed as ``-d <n>`` to llama-bench. This pre-fills
        the KV-cache with N tokens before the tg test, simulating long-context
        decode. Replaces the broken ``-c`` path: vanilla llama-bench has
        no ``-c`` / ``--ctx-size`` switch — every fork we tested (mainline,
        gemma4, thetom, madreag) rejects it with ``error: invalid parameter
        for argument: -c``. The Ctx field in the GUI is for server-start only
        and is now ignored in the benchmark path.
        """
        # Refuse paths with commas — llama-bench's argument parser truncates
        # them at the first comma even when properly quoted. Without this
        # check the bench fails silently as "FAILED" with no diagnostic
        # output, because llama-bench prints the actual error to stderr but
        # the empty stdout produces an unparseable result. See investigation
        # 2026-04-07: TheTom v2 + Qwen 27B from "Q:\AI, Deeplearning, ...".
        if "," in model_path:
            self.after(0, self._log,
                "    Bench refused: model path contains a comma.", "error")
            self.after(0, self._log,
                "    llama-bench's argparse truncates paths at the first", "error")
            self.after(0, self._log,
                "    comma even when properly quoted. Move the model to a", "error")
            self.after(0, self._log,
                "    path without commas to enable benchmarking.", "error")
            self.after(0, self._log,
                f"    Path: {model_path}", "error")
            return None

        cmd = [bench_exe, "-m", model_path, "-ngl", ngl]
        if ctk:
            cmd.extend(["-ctk", ctk])
        if ctv:
            cmd.extend(["-ctv", ctv])

        # Auto-enable Flash Attention when either K or V cache is
        # quantized (same rule as server start). Required by llama.cpp.
        if (ctk and ctk != "f16") or (ctv and ctv != "f16"):
            cmd.extend(["-fa", "1"])

        # Pre-fill depth for tg (decode) test. -d 0 is llama-bench
        # default and matches legacy short-context behavior.
        if depth and depth > 0:
            cmd.extend(["-d", str(depth)])

        try:
            # Per-run timeout is now configurable via the Timeout
            # field in the GUI (default 90s). Replaces the hard-coded
            # 300s which was wasteful for healthy 9B–27B runs (max ~51s
            # observed) and let the broken turbo3/turbo3 path on the gemma4
            # fork eat 5 minutes per call before failing.
            bench_timeout = self._get_bench_timeout()
            kw = {"capture_output": True, "timeout": bench_timeout,
                  "env": env, "stdin": subprocess.DEVNULL}
            if sys.platform == "win32":
                kw["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
            r = subprocess.run(cmd, **kw)
            # llama-bench outputs UTF-8; decode explicitly to avoid cp1252 mangling ±
            output = r.stdout.decode("utf-8", errors="replace") + \
                     r.stderr.decode("utf-8", errors="replace")
        except Exception as e:
            self.after(0, self._log, f"    Error: {e}", "error")
            return None

        # Parse llama-bench table output
        # Extract leading number from a cell — handles "12,569.73 ± 45" and any
        # encoding artefacts like "Â±" that appear when ± is misread as cp1252
        import re as _re
        _num_re = _re.compile(r"^[\s]*([\d,]+\.?\d*)")

        def _extract_num(s: str) -> str:
            m = _num_re.match(s.strip())
            return m.group(1) if m else s.strip()

        result = {}
        for line in output.split("\n"):
            line = line.strip()
            if "|" not in line or "model" in line.lower() or "---" in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            for j, p in enumerate(parts):
                if "pp512" in p or "pp256" in p or "pp1024" in p:
                    test_name = p.strip()
                    if j + 1 < len(parts):
                        ts_str = _extract_num(parts[j + 1])
                        try:
                            result[test_name] = f"{float(ts_str.replace(',', '')):,.2f}"
                        except ValueError:
                            result[test_name] = ts_str
                elif "tg128" in p or "tg256" in p:
                    test_name = p.strip()
                    if j + 1 < len(parts):
                        ts_str = _extract_num(parts[j + 1])
                        try:
                            result[test_name] = f"{float(ts_str.replace(',', '')):,.2f}"
                        except ValueError:
                            result[test_name] = ts_str
        return result if result else None

    def _append_benchmark_file(self, model: ModelInfo, gpu_label: str,
                                results: dict, kv_name: str):
        """Append benchmark results to TurboQuant_Benchmark_results.md beside the launcher."""
        bench_path = BENCH_FILE
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        is_new = not os.path.exists(bench_path)

        # ── Detect column names from first result ──
        pp_col, tg_col = "pp (t/s)", "tg (t/s)"
        for data in results.values():
            pk = next((k for k in data if k.startswith("pp")), None)
            tk_ = next((k for k in data if k.startswith("tg")), None)
            if pk:
                pp_col = f"{pk} (t/s)"
            if tk_:
                tg_col = f"{tk_} (t/s)"
            break

        def _get_pp_tg(data):
            pk = next((k for k in data if k.startswith("pp")), None)
            tk_ = next((k for k in data if k.startswith("tg")), None)
            return (data.get(pk, "—") if pk else "—",
                    data.get(tk_, "—") if tk_ else "—")

        def _delta(val_str, base_val):
            try:
                v = float(val_str.replace(",", ""))
                return f"{(v - base_val) / base_val * 100:+.1f}%" if base_val else "—"
            except (ValueError, ZeroDivisionError):
                return "—"

        # ── Bench All mode (keys contain "|LA=") ──
        # Keys now have format "{kv}|LA=xx|d=N" (three-dimensional).
        # Legacy two-part keys "{kv}|LA=xx" are still accepted as d=0.
        is_bench_all = kv_name == "ALL" and any("|" in k for k in results)

        if is_bench_all:
            # Group by LA tag, then sub-group by depth. Find f16 baseline per
            # (LA, depth) combination so Δ% stays meaningful across contexts.
            from collections import OrderedDict
            groups: "OrderedDict[str, OrderedDict[str, list]]" = OrderedDict()
            for key, data in results.items():
                parts = key.split("|")
                kv_part = parts[0]
                la_part = parts[1] if len(parts) > 1 else ""
                depth_part = parts[2] if len(parts) > 2 else "d=0"
                groups.setdefault(la_part, OrderedDict()) \
                      .setdefault(depth_part, []).append((kv_part, data))

            rows = []
            first_la = True
            for la_tag, depth_groups in groups.items():
                if not first_la:
                    rows.append(("", "", "", "", ""))  # blank separator row
                if len(groups) > 1:
                    rows.append((f"**{la_tag}**", "", "", "", ""))
                first_la = False

                first_depth = True
                for depth_tag, entries in depth_groups.items():
                    # Find f16 baseline in this (LA, depth) group
                    f16_pp, f16_tg = 0.0, 0.0
                    for kv_part, data in entries:
                        if "f16" in kv_part:
                            pp, tg = _get_pp_tg(data)
                            try:
                                f16_pp = float(pp.replace(",", ""))
                                f16_tg = float(tg.replace(",", ""))
                            except ValueError:
                                pass
                            break

                    if not first_depth:
                        rows.append(("", "", "", "", ""))  # blank between depths
                    if len(depth_groups) > 1:
                        rows.append((f"*{depth_tag}*", "", "", "", ""))
                    first_depth = False

                    for kv_part, data in entries:
                        pp, tg = _get_pp_tg(data)
                        if "f16" in kv_part:
                            rows.append((kv_part, pp, tg, "—", "—"))
                        else:
                            rows.append((kv_part, pp, tg,
                                         _delta(pp, f16_pp), _delta(tg, f16_tg)))

            hdr = ("KV-Cache", pp_col, tg_col, "Δ Prefill", "Δ Decode")

        # ── Single bench with f16 baseline (exactly 2 results) ──
        elif len(results) == 2 and "f16" in results:
            f16_data = results["f16"]
            f16_pp_str, f16_tg_str = _get_pp_tg(f16_data)
            try:
                f16_pp = float(f16_pp_str.replace(",", ""))
                f16_tg = float(f16_tg_str.replace(",", ""))
            except ValueError:
                f16_pp, f16_tg = 0.0, 0.0

            rows = []
            delta_row = None
            for config_name, data in results.items():
                pp, tg = _get_pp_tg(data)
                rows.append((config_name, pp, tg))

            other_key = [k for k in results if k != "f16"][0]
            other_pp, other_tg = _get_pp_tg(results[other_key])
            d_pp = _delta(other_pp, f16_pp)
            d_tg = _delta(other_tg, f16_tg)
            if d_pp != "—" or d_tg != "—":
                delta_row = ("**Δ**", f"**{d_pp}**", f"**{d_tg}**")
                rows.append(delta_row)

            hdr = ("KV-Cache", pp_col, tg_col)

        # ── Fallback (no f16, multi-result without LA tags) ──
        else:
            rows = []
            for config_name, data in results.items():
                pp, tg = _get_pp_tg(data)
                rows.append((config_name, pp, tg))
            hdr = ("KV-Cache", pp_col, tg_col)

        # ── Render Markdown table ──
        col_w = [len(h) for h in hdr]
        for r in rows:
            for i, cell in enumerate(r):
                col_w[i] = max(col_w[i], len(cell))

        def _row(cells):
            return "| " + " | ".join(c.ljust(col_w[i]) for i, c in enumerate(cells)) + " |"

        sep = "|-" + "-|-".join("-" * w for w in col_w) + "-|"

        try:
            with open(bench_path, "a", encoding="utf-8") as f:
                if is_new:
                    # Full benchmark environment header on first write.
                    # Documents the run conditions so future readers (and the author)
                    # can tell exactly how each number in this file was produced.
                    f.write("# TurboQuant Benchmark Log\n\n")
                    f.write("Persistent benchmark results from TurboQuant QLauncher.\n\n")
                    f.write("## Benchmark Environment\n\n")
                    f.write("- **Tool:** `llama-bench` (invoked by TurboQuant QLauncher)\n")
                    f.write("- **pp:** 512 tokens (prefill test)\n")
                    f.write("- **tg:** 128 tokens (decode test)\n")
                    f.write("- **ngl:** 99 (all layers offloaded to GPU)\n")
                    f.write("- **Environment:** `CUDA_DEVICE_ORDER=PCI_BUS_ID` "
                            "(mandatory for mixed-arch GPU hosts)\n")
                    f.write("- **Depth (`-d`):** pre-fills the KV-cache with N tokens "
                            "before the tg test — Bench All mode iterates "
                            f"{', '.join(str(d) for d in BENCH_DEPTH_LIST)} "
                            "to expose long-context decode behavior\n")
                    f.write("- **Logged by:** TurboQuant QLauncher "
                            f"v{APP_VERSION} — https://github.com/WaveboSF/TurboQuant-QLauncher\n\n")
                    f.write("---\n\n")

                # Detect CUDA build + driver info for benchmark header
                sp = self.cfg.get("llama_server_path", "").lower()
                if "cuda128" in sp:
                    build_tag = "CUDA 12.8 (sm_89;120)"
                elif "cuda132" in sp:
                    build_tag = "CUDA 13.2 (sm_89;120a)"
                else:
                    build_tag = os.path.basename(os.path.dirname(self.cfg.get("llama_server_path", "")))
                cuda_driver = detect_cuda_version()
                driver_tag = f", Driver {cuda_driver}" if cuda_driver else ""

                # Engine folder name disambiguates forks that share the
                # same CUDA build tag (e.g. three parallel cuda132 builds:
                # mainline, TheTom, Gemma4-Fork). Only adds the tag when the
                # folder name differs from the already-rendered build_tag to
                # avoid redundancy for legacy single-build setups.
                engine_name = slot_folder_name(self.cfg.get("llama_server_path", ""))
                engine_tag = f", Engine {engine_name}" if engine_name and engine_name != build_tag else ""

                # ctx_size no longer appears in the heading — the Ctx
                # GUI field is for server-start only and is ignored in the
                # benchmark path. Depth is recorded per-row in the Bench All
                # table via the ``*d=N*`` sub-headers.
                f.write(f"### {model.filename} — {gpu_label}, Build {build_tag}{driver_tag}{engine_tag}, {ts}\n\n")
                f.write(_row(hdr) + "\n")
                f.write(sep + "\n")
                for r in rows:
                    f.write(_row(r) + "\n")
                f.write("\n---\n\n")

            self.after(0, self._log, f"  ✓ Result saved: {bench_path}", "good")
        except Exception as e:
            self.after(0, self._log, f"  Error writing: {e}", "error")

    def _bench_finished(self, results):
        """Called on main thread when benchmark completes."""
        t = self.theme
        if results:
            self._status_label.config(text="● Benchmark done", fg=t.green)
            self._log("Benchmark completed. Click  💾 Save Results...  to write to file.", "good")
            self._save_bench_btn.configure_btn(color=t.green, state="normal")
        else:
            self._status_label.config(text="● Benchmark failed", fg=t.red)
        self.after(8000, lambda: self._update_status_idle() if not self.server_process else None)

    def _save_pending_bench(self):
        """Save the last benchmark result to the markdown file (manual trigger)."""
        if not self._pending_bench:
            return
        model, gpu_label, results, kv_name = self._pending_bench
        self._append_benchmark_file(model, gpu_label, results, kv_name)
        self._pending_bench = None
        self._save_bench_btn.configure_btn(color=self.theme.border, state="disabled")
        if not self.server_process:
            self._update_status_idle()

    def _update_running_indicator(self):
        t = self.theme
        for filename, card_data in self._model_cards.items():
            for rd in card_data["gpu_rows"]:
                if filename == self.running_model and rd["key"] == self._running_gpu_key:
                    rd["indicator"].config(text="● ", fg=t.green)
                else:
                    rd["indicator"].config(text="  ")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION: Server Management
    # ═══════════════════════════════════════════════════════════════════════

    def _on_run_model(self, filename: str, gpu_key: str = "GPU 0"):
        """Start a model on specified GPU. Stops any running server first."""
        if self.running_model == filename and self._running_gpu_key == gpu_key:
            self._stop_server()
            return

        if self.server_process:
            self._stop_server(silent=True)
            self.after(500, lambda: self._start_server(filename, gpu_key))
        else:
            self._start_server(filename, gpu_key)

    def _start_server(self, filename: str, gpu_key: str = "GPU 0"):
        t = self.theme
        card_data = self._model_cards.get(filename)
        if not card_data:
            return

        model = next((m for m in self.models if m.filename == filename), None)
        if not model:
            return

        server_exe = self.cfg.get("llama_server_path", "")
        if not os.path.isfile(server_exe):
            self._log(f"llama-server.exe not found: {server_exe}", "error")
            self._show_paths_dialog()
            return

        cmd = [server_exe, "-m", model.path, "-ngl", "99"]

        # ────────────────────────────────────────────────────────────────
        # v0.54: Vision model support via mmproj auto-detection.
        #
        # Vision-Language models (Qwen3-VL, Llama-3.2-Vision, etc.) ship
        # as TWO GGUF files: the main weights + a companion multi-modal
        # projector file that encodes images into the model's embedding
        # space. llama.cpp requires both to be passed explicitly:
        #    llama-server -m model.gguf --mmproj mmproj-F16.gguf
        #
        # Rather than adding a UI field, we auto-detect mmproj siblings
        # in the same directory as the selected model. Convention used by
        # Unsloth, Bartowski, and the official Qwen repos: the projector
        # filename starts with 'mmproj' (usually 'mmproj-F16.gguf' or
        # 'mmproj-Q8_0.gguf'). Zero config for the user — drop the pair
        # of files into a model directory and both flags are wired up.
        #
        # For text-only models (no mmproj present) this block is a no-op,
        # so existing workflows (Qwen 3, Llama, Mistral, Gemma text) are
        # unaffected. qnut Probe runs on text-only models also untouched.
        # ────────────────────────────────────────────────────────────────
        try:
            model_dir = os.path.dirname(model.path)
            mmproj_files = [
                f for f in os.listdir(model_dir)
                if f.lower().startswith("mmproj") and f.lower().endswith(".gguf")
            ]
            if mmproj_files:
                # Prefer F16 projector over quantized variants when multiple
                # exist side-by-side — F16 is the reference encoder quality
                # and projector size is negligible (~600MB-1GB) compared to
                # weight savings on the main model.
                mmproj_files.sort(
                    key=lambda f: (0 if "f16" in f.lower() else 1, f.lower())
                )
                mmproj_path = os.path.join(model_dir, mmproj_files[0])
                cmd.extend(["--mmproj", mmproj_path])
                self._log(
                    f"  Vision encoder auto-attached: {mmproj_files[0]}",
                    "info",
                )
        except OSError as e:
            # Directory scan failed — log and continue without vision.
            # The server will still start as text-only.
            self._log(
                f"  mmproj scan failed ({e}); starting text-only",
                "warn",
            )

        is_cpu = (gpu_key == "CPU")
        if is_cpu:
            cmd[cmd.index("-ngl") + 1] = "0"
            env_cuda = None
            gpu_display = "CPU"
        else:
            gpu_idx = gpu_key.split(" ")[1]
            env_cuda = gpu_idx
            gpu_display = gpu_key

        kv_name = self._kv_var.get()
        kv = KV_CACHE_OPTIONS.get(kv_name, {})
        if is_cpu and kv.get("ctk") and "turbo" in (kv.get("ctk", "") + kv.get("ctv", "")):
            self._log(f"CPU mode: TurboQuant KV-Cache not available, using f16", "warn")
        else:
            if kv.get("ctk"):
                cmd.extend(["-ctk", kv["ctk"]])
            if kv.get("ctv"):
                cmd.extend(["-ctv", kv["ctv"]])

        # Auto-enable Flash Attention when either K or V cache is
        # quantized (turbo2/3/4, q8_0, etc.). llama.cpp requires -fa for any
        # quantized cache — without it, context init fails with
        # "quantized V cache was requested, but this requires Flash Attention".
        # Observed on Gemma 4 with turbo3/turbo3 specifically. For f16/f16
        # runs we leave FA at its default so baseline behaviour is unchanged.
        ctk = kv.get("ctk", "")
        ctv = kv.get("ctv", "")
        if (ctk and ctk != "f16") or (ctv and ctv != "f16"):
            cmd.extend(["-fa", "on"])

        if self._no_think_var.get():
            # v0.56 — the blocking "Reasoning model — thinking disabled"
            # messagebox was removed. It fired before every single server
            # start of a reasoning model, which got tedious fast when
            # switching models back-to-back. Replaced with:
            #   (a) an inline warning text in the settings bar, rendered
            #       dynamically by _refresh_thinking_warning() whenever
            #       the selected model + the No Thinking toggle combine
            #       into a "dangerous" state. The user sees it without
            #       being interrupted.
            #   (b) a single log line at the moment of starting, so the
            #       Server Log records that the combination was used.
            # v0.57 — detection now uses the is_reasoning flag computed
            # from GGUF metadata at scan time (falls back to filename).
            if model.is_reasoning:
                self._log(
                    f"Note: '{filename}' is a reasoning-capable model "
                    f"and 'No Thinking' is ON. Accuracy on math / logic "
                    f"tasks may drop substantially (e.g. Gemma 4 26B: "
                    f"~97% → ~64% on our suite). Uncheck 'No Thinking' "
                    f"if this was unintentional.",
                    "warn")
            cmd.extend(["--reasoning", "off"])

        # ────────────────────────────────────────────────────────────────
        # v0.57: Port-conflict pre-flight check.
        #
        # Before spawning llama-server, probe-bind on the requested port.
        # If something is already listening, llama-server will fail with
        # a bind error a few seconds into startup — by which point the
        # user has been staring at "Loading..." and wondering what went
        # wrong. Catching it here lets us surface a concrete, actionable
        # message (with likely causes) and let the user decide whether
        # to try anyway (maybe the port is in TIME_WAIT) or change it.
        # ────────────────────────────────────────────────────────────────
        port_raw = (self._port_var.get() or "8080").strip()
        try:
            port_int = int(port_raw)
            if not (1 <= port_int <= 65535):
                raise ValueError
        except ValueError:
            self._log(
                f"Invalid port value {port_raw!r} — falling back to 8080.",
                "warn")
            port_raw = "8080"
            port_int = 8080

        if not _port_is_free(port_int):
            # Collision with our own IPC listener (Autoload ON + user
            # typed a port that happened to hit it). Unlikely but ugly
            # when it happens — give a specific error and bail.
            if self._ipc_port and port_int == self._ipc_port:
                messagebox.showerror(
                    "Port Conflict with TurboQuant IPC",
                    f"Port {port_int} is currently used by TurboQuant's "
                    f"own IPC listener (started when Autoload was "
                    f"enabled).\n\n"
                    f"Starting llama-server on this port would fail. "
                    f"Please pick a different port in the Port field "
                    f"and try again.")
                self._log(
                    f"Server start cancelled — port {port_int} collides "
                    f"with TurboQuant's IPC socket.",
                    "error")
                return
            proceed = messagebox.askyesno(
                "Port Already in Use",
                f"Port {port_int} is already in use by another process "
                f"on this machine.\n\n"
                f"Starting llama-server on this port will most likely "
                f"fail with a bind error. Common causes:\n\n"
                f"  • Another llama-server instance is still running\n"
                f"  • A different app (Ollama, vLLM, "
                f"text-generation-webui, …) is using this port\n"
                f"  • A previous server did not shut down cleanly\n\n"
                f"Recommended: close the other process, or change the "
                f"port in the Port field and try again.\n\n"
                f"Start anyway?",
                icon="warning",
                default="no",
            )
            if not proceed:
                self._log(
                    f"Server start cancelled — port {port_int} is in "
                    f"use.",
                    "warn")
                return
            self._log(
                f"Starting on port {port_int} despite apparent "
                f"conflict — user confirmed override.",
                "warn")

        port = port_raw
        cmd.extend(["--host", "0.0.0.0", "--port", port])

        # Append -c <ctx> if user provided a context size.
        # Empty field → llama-server uses its default (usually model native).
        ctx_raw = (self._ctx_var.get() or "").strip()
        if ctx_raw:
            try:
                ctx_val = int(ctx_raw)
                if ctx_val > 0:
                    cmd.extend(["-c", str(ctx_val)])
            except ValueError:
                self._log(f"Ignoring invalid Ctx value: {ctx_raw!r} (expected integer)", "warn")

        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if env_cuda is not None:
            env["CUDA_VISIBLE_DEVICES"] = env_cuda

        la_mode = self._la_var.get()
        if la_mode > 0:
            env["TURBO_LAYER_ADAPTIVE"] = str(la_mode)

        la_info = f", LA={la_mode}" if la_mode > 0 else ""
        self._log(f"Starting {filename} on {gpu_display} ({kv_name}, :{port}{la_info})", "info")
        self._log(f"  Server: {server_exe}", "info")
        self._log(f"$ {' '.join(cmd)}", "info")

        try:
            kw = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
                  "stdin": subprocess.DEVNULL, "env": env, "bufsize": 1,
                  "universal_newlines": True}
            if sys.platform == "win32":
                # CREATE_NO_WINDOW (0x08000000): no console window pops
                # up for the server subprocess. Without this every
                # server start (and there are up to 24 of them per
                # Probe run) flashes a black console window across the
                # screen, which is visually disruptive. The server's
                # actual output is captured via the stdout PIPE above
                # and surfaced in the launcher's Server Log panel, so
                # the console window contains nothing useful anyway.
                #
                # CREATE_NEW_PROCESS_GROUP (0x00000200): server runs in
                # its own process group so the launcher can shut it
                # down cleanly via CTRL_BREAK_EVENT without killing
                # itself. Both flags are bitfields, OR them together.
                kw["creationflags"] = (
                    subprocess.CREATE_NEW_PROCESS_GROUP
                    | 0x08000000  # CREATE_NO_WINDOW
                )
            self.server_process = subprocess.Popen(cmd, **kw)
            self.running_model = filename
            self._running_gpu_key = gpu_key
            self._running_port = port
            # qnut v0.50 — remember which KV/LA settings this server
            # was actually started with, so the Q/N/C click handler can
            # detect a no-op restart (clicking a mode whose anchor
            # equals the running settings should not waste 10 seconds
            # tearing down and restarting an identical server).
            self._running_kv = self._kv_var.get()
            self._running_la = self._la_var.get()

            # v0.56 — record the (model, GPU) pair so autoload +
            # `--autostart` know what to launch next session. We save it
            # at the Popen-succeeded point, BEFORE waiting for "listening",
            # on purpose: if the user kills the launcher mid-load we still
            # want the pair recorded. install_verified is what gates
            # autoload at the actual launcher level, and that flag is only
            # flipped once we've seen "listening" (see _read_server_output).
            self.cfg["last_model"]   = filename
            self.cfg["last_gpu_key"] = gpu_key

            self._update_status_loading(filename, port)
            self._update_running_indicator()
            self._save_current_config()

            self._log_reader_thread = threading.Thread(target=self._read_server_output,
                                                        daemon=True)
            self._log_reader_thread.start()

        except Exception as e:
            self._log(f"Failed to start server: {e}", "error")
            self.server_process = None
            self.running_model = None
            self._running_gpu_key = None
            self._running_kv = None
            self._running_la = None

    def _stop_server(self, silent=False):
        t = self.theme
        if self.server_process:
            name = self.running_model or "server"
            if not silent:
                self._log(f"Stopping {name}...", "warn")
            try:
                if sys.platform == "win32":
                    self.server_process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    self.server_process.terminate()
                try:
                    self.server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.server_process.kill()
                    self.server_process.wait(timeout=3)
            except Exception as e:
                if not silent:
                    self._log(f"Force kill: {e}", "warn")
                try:
                    self.server_process.kill()
                except Exception:
                    pass
            self.server_process = None
            self.running_model = None
            self._running_gpu_key = None
            self._running_kv = None
            self._running_la = None

            if not silent:
                self._log("Server stopped.", "info")
                self._update_status_idle()
                self._update_running_indicator()

    def _read_server_output(self):
        """Background thread reading server stdout."""
        proc = self.server_process
        if not proc or not proc.stdout:
            return
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    tag = "info"
                    ll = line.lower()
                    if "error" in ll or "failed" in ll:
                        tag = "error"
                    elif "listening" in ll or "model loaded" in ll or "ready" in ll:
                        tag = "good"
                        if "listening" in ll or "all slots are idle" in ll:
                            fn  = self.running_model or ""
                            prt = getattr(self, "_running_port", "8080")
                            self.after(0, self._upgrade_status_to_running, fn, prt)
                    elif "warning" in ll:
                        tag = "warn"
                    if "all slots are idle" in ll:
                        tag = "good"
                        fn  = self.running_model or ""
                        prt = getattr(self, "_running_port", "8080")
                        self.after(0, self._upgrade_status_to_running, fn, prt)
                    # v0.56 — First time we see the server announce it's
                    # actually listening on its port, the installation is
                    # proven to work. This one-way flip unlocks the
                    # "Autoload" button for the user. Runs on the main
                    # thread to keep cfg writes single-threaded.
                    if ("listening" in ll
                            and not self.cfg.get("install_verified", False)):
                        self.after(0, self._mark_install_verified)
                    self.after(0, self._log, line, tag)
        except Exception:
            pass
        self.after(0, self._on_server_exited)

    def _mark_install_verified(self):
        """Flip the install-verified flag (main thread only)."""
        if self.cfg.get("install_verified", False):
            return
        self.cfg["install_verified"] = True
        save_config(self.cfg)
        self._log(
            "Installation verified — llama-server is listening. "
            "Autoload is now unlockable (footer).", "good")
        self._refresh_autoload_btn()

    def _upgrade_status_to_running(self, filename: str, port: str):
        """Switch status from Loading → Running (called on main thread)."""
        if self.server_process and self.running_model == filename:
            self._update_status_running(filename, port)

    def _on_server_exited(self):
        t = self.theme
        if self.running_model:
            self._log(f"Server process exited.", "warn")
            self.running_model = None
            self._running_gpu_key = None
            self._running_kv = None
            self._running_la = None
            self.server_process = None
            self._status_label.config(text="● Idle", fg=t.fg_dim)
            self._update_running_indicator()

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION: Log
    # ═══════════════════════════════════════════════════════════════════════

    def _log(self, text: str, tag: str = "info"):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_text.config(state="normal")
        self._log_text.insert("end", f"[{ts}] ", "time")
        self._log_text.insert("end", f"{text}\n", tag)
        self._log_text.see("end")
        self._log_text.config(state="disabled")

    def _clear_log(self):
        self._log_text.config(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.config(state="disabled")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION: Scanning & Initialization
    # ═══════════════════════════════════════════════════════════════════════

    def _do_initial_scan(self):
        t = self.theme
        # Show the loading hint right when the scan starts. The
        # initial label text is a placeholder space (reserves line height);
        # setting it here means the first user-visible text transition is
        # "Detecting GPUs..." → final text, and both have the same height.
        self._gpu_label.config(text="Detecting GPUs...")
        self.update_idletasks()
        self.gpus = detect_all_gpus()
        self._cpu_ram_gb = detect_cpu_ram_gb()

        if self.gpus:
            parts = []
            for g in self.gpus:
                vram = f" ({g.vram_mb // 1024} GB)" if g.vram_mb else ""
                parts.append(f"GPU {g.index}: {g.name}{vram}")
            if self._cpu_ram_gb > 0:
                parts.append(f"CPU RAM: {self._cpu_ram_gb:.0f} GB")
            self._gpu_label.config(text="  •  ".join(parts))
        else:
            ram_info = f" — CPU RAM: {self._cpu_ram_gb:.0f} GB" if self._cpu_ram_gb > 0 else ""
            self._gpu_label.config(text=f"No GPUs detected — CPU mode only{ram_info}")

        self._build_gpu_buttons()

        cuda_ver = detect_cuda_version()
        server_path = self.cfg.get("llama_server_path", "")
        server_dir = os.path.dirname(server_path)
        dll_results = check_required_dlls(server_dir) if server_dir else []
        dll_found = sum(1 for d in dll_results if d["found"])
        dll_total = len(dll_results)
        dll_missing = [d["name"] for d in dll_results if not d["found"]]

        cuda_parts = []
        if cuda_ver:
            cuda_parts.append(f"CUDA {cuda_ver} (driver)")
        # Detect build variant from server path
        sp_lower = server_path.lower()
        if "cuda128" in sp_lower:
            cuda_parts.append("Build: CUDA 12.8")
        elif "cuda132" in sp_lower:
            cuda_parts.append("Build: CUDA 13.2")
        elif server_dir:
            cuda_parts.append(f"Build: {os.path.basename(server_dir)}")
        if dll_total > 0:
            if dll_found == dll_total:
                cuda_parts.append(f"DLLs: {dll_found}/{dll_total} ✓")
                dll_color = t.fg_secondary
            else:
                cuda_parts.append(f"DLLs: {dll_found}/{dll_total} ✗ Missing: {', '.join(dll_missing)}")
                dll_color = t.yellow
        else:
            dll_color = t.fg_dim
            cuda_parts.append("DLLs: server path not set")

        self._cuda_label.config(text="  •  ".join(cuda_parts), foreground=dll_color)

        if dll_missing:
            self._log(f"Missing DLLs in {server_dir}: {', '.join(dll_missing)}", "warn")
            cublas_missing = any("cublas" in d.lower() for d in dll_missing)
            if cublas_missing:
                # Extract CUDA major version from the missing dll name (e.g. "cublas64_12.dll" → "12")
                cuda_major = "12"
                for d in dll_missing:
                    if "cublas64_" in d:
                        cuda_major = d.split("_")[-1].replace(".dll", "")
                        break
                self._log(f"  → CUDA {cuda_major}.x runtime required for GPU acceleration.", "warn")
                self._log(f"  → Download: https://developer.nvidia.com/cuda-toolkit", "warn")
                self._log(f"  → Or copy cublas64_{cuda_major}.dll + cublasLt64_{cuda_major}.dll", "info")
                self._log(f"    from your CUDA installation to: {server_dir}", "info")

        self._rescan_models()
        # Refresh quick-switch slot buttons (updates active highlight
        # when the current llama_server_path changes, e.g. after a slot click
        # or a Paths dialog save).
        self._refresh_slot_buttons()

        # v0.56 — Autoload hook. Placed here (end of _do_initial_scan)
        # rather than in __init__ because we need self.models + the GPU
        # rows to be fully populated before _trigger_autoload_if_eligible
        # can resolve the last (model, GPU) pair. Also fires on
        # _first_run_setup path via the Paths dialog's on_close callback,
        # which still routes through _do_initial_scan.
        #
        # v0.58 — deferred to after_idle so it runs AFTER the default
        # _select_cell(0, 0) queued at the end of card construction (see
        # the `if self.models:` block in _build_model_cards). A plain
        # synchronous call here would pick the correct (model, GPU) cell
        # and then get clobbered moments later when the default-selection
        # idle callback fires, leaving the selection rectangle on row
        # (0, 0) while the running indicator correctly shows the
        # autoloaded model elsewhere. after_idle serialises both calls
        # into the same queue, so the autoload selection overwrites the
        # default one in the right order. Failures are still logged
        # before the user can meaningfully interact — after_idle fires
        # well before the first input event.
        self.after_idle(self._trigger_autoload_if_eligible)

    def _rescan_models(self):
        paths = [p for p in (self.cfg.get("llm_models_paths") or []) if p]
        recursive = bool(self.cfg.get("llm_models_recursive", True))
        self.models = scan_models(paths, recursive=recursive)
        self._models_header.config(
            text=f"Models ({len(self.models)} GGUF, "
                 f"{sum(m.size_gb for m in self.models):.0f} GB)")
        self._rebuild_model_cards()
        if not paths:
            self._log("Scan: no model directories configured.", "warn")
        else:
            mode = "recursive" if recursive else "top-level"
            self._log(f"Scan ({mode}): {len(self.models)} models in "
                      f"{len(paths)} dir(s) — {', '.join(paths)}", "info")
        # Surface comma-in-path warnings once per scan so the user sees
        # the issue at startup, not just when they try to bench. The
        # affected models are also marked with ⚠ in the model list.
        unsafe = [m for m in self.models if m.has_unsafe_path]
        if unsafe:
            self._log(
                f"Warning: {len(unsafe)} model(s) live in paths with commas.",
                "warn")
            self._log(
                "  llama-bench will refuse these — move them to a comma-free",
                "warn")
            self._log(
                "  path to enable benchmarking. llama-server is unaffected.",
                "warn")
            for m in unsafe:
                self._log(f"  ⚠ {m.filename}", "warn")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION: Dialogs
    # ═══════════════════════════════════════════════════════════════════════

    def _first_run_setup(self):
        """First run: show paths dialog, then do initial scan after it closes."""
        self._log("First run — please configure paths.", "warn")
        self._show_paths_dialog(on_close=self._do_initial_scan)

    def _show_paths_dialog(self, on_close=None):
        t = self.theme
        dlg = tk.Toplevel(self)
        dlg.title("Configure Paths")
        dlg.configure(bg=t.bg)
        dlg.transient(self)
        dlg.grab_set()

        tk.Label(dlg, text="Configure Paths", font=FONT_TITLE,
                 bg=t.bg, fg=t.fg).pack(padx=20, pady=(12, 8), anchor="w")
        tk.Frame(dlg, height=1, bg=t.border).pack(fill="x", padx=16, pady=(0, 8))

        tk.Label(dlg, text="LLM Models Directories:",
                 font=FONT_BODY_B, bg=t.bg, fg=t.fg).pack(padx=20, anchor="w")
        tk.Label(dlg,
                 text="All configured directories are scanned together. "
                      "Duplicate files are detected automatically.",
                 font=FONT_BODY, bg=t.bg, fg=t.fg_dim,
                 wraplength=964, justify="left").pack(padx=20, anchor="w", pady=(0, 4))

        # Load current model paths, pad to MAX_MODELS_PATHS
        current_models_paths = list(self.cfg.get("llm_models_paths") or [])
        while len(current_models_paths) < MAX_MODELS_PATHS:
            current_models_paths.append("")
        models_vars: list = []
        for i in range(MAX_MODELS_PATHS):
            row = tk.Frame(dlg, bg=t.bg)
            row.pack(fill="x", padx=20, pady=1)
            tk.Label(row, text=f"#{i + 1}", font=FONT_BODY,
                     bg=t.bg, fg=t.fg_secondary,
                     width=4, anchor="w").pack(side="left", padx=(0, 6))
            mvar = tk.StringVar(value=current_models_paths[i])
            models_vars.append(mvar)
            mentry = ttk.Entry(row, textvariable=mvar, font=FONT_BODY)
            mentry.pack(side="left", fill="x", expand=True)
            HoverButton(row, t, text="...", color=t.border, width=36, height=24,
                        command=lambda v=mvar: self._browse_dir(v)).pack(side="left", padx=(4, 0))

        # Global recursive-scan checkbox
        rec_row = tk.Frame(dlg, bg=t.bg)
        rec_row.pack(fill="x", padx=20, pady=(6, 8))
        recursive_var = tk.BooleanVar(value=bool(self.cfg.get("llm_models_recursive", True)))
        cb_rec = tk.Checkbutton(rec_row,
                                 text="Include subdirectories (recursive scan)",
                                 variable=recursive_var,
                                 font=FONT_BODY, bg=t.bg, fg=t.fg,
                                 selectcolor=t.bg_secondary,
                                 activebackground=t.bg, activeforeground=t.fg)
        cb_rec.pack(side="left")

        # ── Quick-switch slots ──
        tk.Frame(dlg, height=1, bg=t.border).pack(fill="x", padx=16, pady=(4, 8))
        tk.Label(dlg, text="Quick-Switch Slots:",
                 font=FONT_BODY_B, bg=t.bg, fg=t.fg).pack(padx=20, anchor="w")
        tk.Label(dlg,
                 text="Bookmark llama-server.exe builds here — they appear "
                      "as one-click buttons in the footer. Click a button to "
                      "activate that engine.",
                 font=FONT_BODY, bg=t.bg, fg=t.fg_dim,
                 wraplength=964, justify="left").pack(padx=20, anchor="w", pady=(0, 4))

        # Load current slots, pad to MAX_SERVER_SLOTS
        current_slots = list(self.cfg.get("server_slots") or [])
        while len(current_slots) < MAX_SERVER_SLOTS:
            current_slots.append("")
        slot_vars: list = []

        # Keep a reference to the preview labels so they update live as the
        # user types/browses. Each row: [Label preview] [Entry] [...]
        slot_preview_labels: list = []

        def _make_updater(var, label_widget):
            def _update(*_):
                preview = slot_label_from_path(var.get()) or "(empty)"
                label_widget.config(text=preview)
            return _update

        for i in range(MAX_SERVER_SLOTS):
            row = tk.Frame(dlg, bg=t.bg)
            row.pack(fill="x", padx=20, pady=1)
            # Fixed-width preview label on the left (shows derived button name)
            preview_lbl = tk.Label(row, text="(empty)", font=FONT_BODY,
                                    bg=t.bg, fg=t.fg_secondary,
                                    width=18, anchor="w")
            preview_lbl.pack(side="left", padx=(0, 6))
            slot_preview_labels.append(preview_lbl)

            var = tk.StringVar(value=current_slots[i])
            slot_vars.append(var)
            entry = ttk.Entry(row, textvariable=var, font=FONT_BODY)
            entry.pack(side="left", fill="x", expand=True)
            HoverButton(row, t, text="...", color=t.border, width=36, height=24,
                        command=lambda v=var: self._browse_file(v)).pack(side="left", padx=(4, 0))

            # Live preview update as the user types or browses
            var.trace_add("write", _make_updater(var, preview_lbl))
            # Initial preview render
            preview_lbl.config(text=slot_label_from_path(var.get()) or "(empty)")

        tk.Frame(dlg, height=1, bg=t.border).pack(fill="x", padx=16, pady=8)

        btn_frame = tk.Frame(dlg, bg=t.bg)
        btn_frame.pack(fill="x", padx=20, pady=(0, 12))

        def _save():
            # persist list of model paths + recursive flag
            new_paths = [v.get().strip() for v in models_vars]
            while len(new_paths) < MAX_MODELS_PATHS:
                new_paths.append("")
            self.cfg["llm_models_paths"] = new_paths[:MAX_MODELS_PATHS]
            self.cfg["llm_models_recursive"] = bool(recursive_var.get())
            # Drop the legacy single-path key if it's still hanging around
            self.cfg.pop("llm_models_path", None)
            # persist slot paths (strip whitespace, keep empty slots)
            new_slots = [v.get().strip() for v in slot_vars]
            self.cfg["server_slots"] = new_slots

            # v0.55: active llama_server_path is set EXCLUSIVELY by
            # _on_slot_click() in the main UI; this dialog never writes
            # to it. If the user removes the slot whose path is currently
            # active, log a hint — the engine still works, just no slot
            # button will be highlighted until they click one.
            current_active = (self.cfg.get("llama_server_path") or "").strip()
            if current_active:
                slot_norms = {os.path.normcase(os.path.normpath(s))
                              for s in new_slots if s}
                active_norm = os.path.normcase(os.path.normpath(current_active))
                if active_norm not in slot_norms:
                    self._log(
                        "Note: active engine is no longer bookmarked in any "
                        "slot. It still works, but no slot button will be "
                        "highlighted until you click one.", "info")

            save_config(self.cfg)
            self._log("Paths saved.", "good")
            dlg.destroy()
            if on_close:
                on_close()
            else:
                self._do_initial_scan()

        def _cancel():
            dlg.destroy()
            if on_close:
                on_close()

        HoverButton(btn_frame, t, text="Save", color=t.accent,
                    width=100, height=28, command=_save).pack(side="left")
        HoverButton(btn_frame, t, text="Cancel", color=t.border,
                    width=100, height=28, command=_cancel).pack(side="left", padx=8)

        dlg.update_idletasks()
        self._center_window(dlg, 1024, dlg.winfo_reqheight())

    def _browse_dir(self, var: tk.StringVar):
        path = filedialog.askdirectory(initialdir=var.get())
        if path:
            var.set(path)

    def _browse_file(self, var: tk.StringVar):
        path = filedialog.askopenfilename(
            initialdir=os.path.dirname(var.get()),
            filetypes=[("Executable", "*.exe"), ("All files", "*.*")])
        if path:
            var.set(path)

    def _show_about(self):
        t = self.theme
        dlg = tk.Toplevel(self)
        dlg.title("About")
        dlg.configure(bg=t.bg)
        dlg.transient(self)
        dlg.grab_set()

        tk.Label(dlg, text="TurboQuant QLauncher", font=FONT_TITLE,
                 bg=t.bg, fg=ACCENT_TURBO).pack(padx=20, pady=(16, 2), anchor="w")
        tk.Label(dlg, text=f"v{APP_VERSION}  •  MIT License", font=FONT_SMALL,
                 bg=t.bg, fg=t.fg_secondary).pack(padx=20, anchor="w")
        tk.Label(dlg, text="© WaveboSF 2026", font=FONT_SMALL,
                 bg=t.bg, fg=t.fg_dim).pack(padx=20, anchor="w")
        tk.Frame(dlg, height=1, bg=t.border).pack(fill="x", padx=16, pady=8)

        info = [
            ("Backend:", "llama-server (spiritbuun TurboQuant fork)"),
            ("KV-Cache:", "TurboQuant KV compression (turbo3/turbo4)"),
            ("API:", "OpenAI-compatible (/v1/chat/completions)"),
            ("License:", "MIT"),
            ("Python:", platform.python_version()),
            ("Platform:", f"{platform.system()} {platform.machine()}"),
        ]
        for label, value in info:
            row = tk.Frame(dlg, bg=t.bg)
            row.pack(fill="x", padx=20, pady=1)
            tk.Label(row, text=label, font=FONT_BODY_B, bg=t.bg, fg=t.fg,
                     width=12, anchor="w").pack(side="left")
            tk.Label(row, text=value, font=FONT_BODY, bg=t.bg, fg=t.fg_secondary,
                     anchor="w").pack(side="left")

        tk.Frame(dlg, height=1, bg=t.border).pack(fill="x", padx=16, pady=8)

        tk.Label(dlg, text="Developed by", font=FONT_BODY_B,
                 bg=t.bg, fg=t.fg).pack(padx=20, anchor="w")
        dev_entries = [
            ("WaveboSF",           "Concept, Design & Development"),
            ("Claude (Anthropic)", "AI Assistance & Code Generation"),
        ]
        for name, role in dev_entries:
            row = tk.Frame(dlg, bg=t.bg)
            row.pack(fill="x", padx=20, pady=1)
            tk.Label(row, text=name, font=FONT_BODY_B, bg=t.bg, fg=t.fg_secondary,
                     width=20, anchor="w").pack(side="left")
            tk.Label(row, text=role, font=FONT_SMALL, bg=t.bg, fg=t.fg_dim,
                     anchor="w").pack(side="left")

        tk.Frame(dlg, height=1, bg=t.border).pack(fill="x", padx=16, pady=8)

        txt_frame = tk.Frame(dlg, bg=t.bg)
        txt_frame.pack(fill="x", padx=16, pady=(0, 8))
        txt = tk.Text(txt_frame, font=FONT_SMALL, bg=t.bg_secondary, fg=t.fg,
                      relief="flat", bd=0, wrap="word", height=14, width=56,
                      cursor="arrow", selectbackground=t.select_bg)
        txt.pack(fill="x")
        txt.tag_configure("ack_hdr", font=FONT_SMALL_B, foreground=t.fg)
        txt.tag_configure("ack",     font=FONT_SMALL,   foreground=t.fg_secondary)
        txt.tag_configure("dim",     font=FONT_SMALL,   foreground=t.fg_dim)
        txt.tag_configure("link",    font=FONT_SMALL,   foreground=ACCENT_TURBO,
                          underline=True)

        def _insert_link(label, url):
            tag = f"link_{url}"
            txt.tag_configure(tag, font=FONT_SMALL, foreground=ACCENT_TURBO, underline=True)
            txt.tag_bind(tag, "<Button-1>",
                         lambda e, u=url: __import__("webbrowser").open(u))
            txt.tag_bind(tag, "<Enter>", lambda e: txt.configure(cursor="hand2"))
            txt.tag_bind(tag, "<Leave>", lambda e: txt.configure(cursor="arrow"))
            txt.insert("end", label, tag)

        txt.insert("end", "GitHub Projects\n", "ack_hdr")

        txt.insert("end", "  spiritbuun fork  (TurboQuant CUDA — this build)\n  ", "ack")
        _insert_link("→ github.com/spiritbuun/llama-cpp-turboquant-cuda",
                     "https://github.com/spiritbuun/llama-cpp-turboquant-cuda/tree/feature/turboquant-kv-cache")
        txt.insert("end", "\n\n", "ack")

        txt.insert("end", "  turboquant_plus  (Metal + CUDA, block_size=128)\n  ", "ack")
        _insert_link("→ github.com/TheTom/turboquant_plus",
                     "https://github.com/TheTom/turboquant_plus")
        txt.insert("end", "\n\n", "ack")

        txt.insert("end", "  ggml-org / llama.cpp  (upstream)\n  ", "ack")
        _insert_link("→ github.com/ggml-org/llama.cpp",
                     "https://github.com/ggml-org/llama.cpp")
        txt.insert("end", "\n\n", "ack")

        txt.insert("end", "  Community discussion  (benchmarks & findings)\n  ", "ack")
        _insert_link("→ github.com/ggml-org/llama.cpp/discussions/20969",
                     "https://github.com/ggml-org/llama.cpp/discussions/20969")
        txt.insert("end", "\n\n", "ack")

        txt.insert("end", "Acknowledgements\n", "ack_hdr")
        txt.insert("end",
                   "  Special thanks to seanrasch (Ampere benchmarks),\n"
                   "  AmesianX (Blackwell / TurboQuant v1.4.0) and primoco\n"
                   "  (accuracy benchmarks) for their community contributions.\n\n",
                   "ack")

        txt.insert("end", "Disclaimer\n", "ack_hdr")
        txt.insert("end",
                   "  \"TurboQuant\" refers to the paper by Zandieh et al.\n"
                   "  (ICLR 2026, Google Research). This tool is an independent\n"
                   "  community project, not affiliated with Google.", "dim")

        txt.config(state="disabled")

        HoverButton(dlg, t, text="OK", color=t.accent, width=80, height=26,
                    command=dlg.destroy).pack(pady=(12, 12))
        dlg.update_idletasks()
        self._center_window(dlg, 500, dlg.winfo_reqheight())

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION: Update Binaries
    # ═══════════════════════════════════════════════════════════════════════

    def _update_binaries(self):
        src = Path("G:\\_Entwicklung\\llama-cpp-turboquant_spiritbuun_v2\\build-cuda128\\bin\\Release")
        dst = Path(self.cfg.get("llama_server_path", "")).parent

        if not src.exists():
            self._log(f"Build directory not found: {src}", "error")
            return
        if not dst.exists():
            self._log(f"Target directory not found: {dst}", "error")
            return

        if self.server_process:
            self._log("Server must be stopped first!", "error")
            return

        files = ["llama-server.exe", "llama-cli.exe", "llama-bench.exe",
                 "llama.dll", "ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll",
                 "ggml.dll", "mtmd.dll"]
        count = 0
        for f in files:
            s = src / f
            if s.exists():
                try:
                    import shutil
                    shutil.copy2(s, dst / f)
                    count += 1
                except Exception as e:
                    self._log(f"Error copying {f}: {e}", "error")
        self._log(f"Update: {count}/{len(files)} files copied.", "good")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION: Config & Cleanup
    # ═══════════════════════════════════════════════════════════════════════

    def _save_current_config(self):
        self.cfg["kv_cache"] = self._kv_var.get()
        self.cfg["port"] = int(self._port_var.get() or 8080)
        self.cfg["ctx_size"] = (self._ctx_var.get() or "").strip()  # persist Ctx field
        # Persist bench timeout. Stored as int (after validation),
        # falls back to 90 if the user typed something invalid.
        self.cfg["bench_timeout"] = self._get_bench_timeout()
        self.cfg["no_thinking"] = self._no_think_var.get()
        self.cfg["benchmark"] = self._bench_var.get()
        self.cfg["bench_all"] = self._bench_all_var.get()
        self.cfg["layer_adaptive"] = self._la_var.get()
        # v0.56 — cfg["autoload"] is written directly by _on_autoload_click
        # (so it's durable the moment the user flips the toggle, even if
        # the launcher crashes before a normal save). install_verified is
        # written by _mark_install_verified. Neither needs a write here.
        try:
            self.cfg["window_x"] = self.winfo_x()
            self.cfg["window_y"] = self.winfo_y()
            self.cfg["window_w"] = self.winfo_width()
            self.cfg["window_h"] = self.winfo_height()
        except Exception:
            pass
        try:

            _, sash_y = self._paned.sash_coord(0)
            self.cfg["sash_pos"] = sash_y
        except Exception:
            pass
        save_config(self.cfg)

    # ─── Inline "No Thinking on reasoning model" warning ────────────────

    def _refresh_thinking_warning(self):
        """Show/hide the inline warning next to the No Thinking checkbox.

        Called after two kinds of events:
          - the user toggles No Thinking (via trace_add on _no_think_var)
          - the user selects a different model row (via _select_cell)
        The label stays empty unless BOTH conditions are true:
          - No Thinking is checked
          - the currently selected model is a reasoning model
        When both hold, a short yellow "⚠ reasoning model" message
        appears next to the checkbox, and its tooltip carries the long
        explanation that used to live in the removed messagebox.
        """
        # Widgets may not exist yet during early __init__ — guard.
        if not hasattr(self, "_thinking_warn_label"):
            return
        # Resolve the currently selected model, if any.
        filename = None
        model_obj = None
        try:
            if (0 <= self._sel_model < len(self.models)):
                model_obj = self.models[self._sel_model]
                filename = model_obj.filename
        except (AttributeError, IndexError):
            filename = None
            model_obj = None
        show = (self._no_think_var.get()
                and model_obj is not None
                and model_obj.is_reasoning)
        if show:
            self._thinking_warn_label.config(
                text="⚠ reasoning model",
                fg=self.theme.yellow)
            self._thinking_warn_tooltip.update_text(
                f"'{filename}' is a reasoning-capable model "
                f"(Gemma 4 / Qwen3 / DeepSeek-R1 / QwQ family).\n\n"
                f"'No Thinking' is ON, which can drop accuracy "
                f"dramatically on math and logic tasks (Gemma 4 26B: "
                f"~97% → ~64% on our suite).\n\n"
                f"Uncheck 'No Thinking' if this was unintentional.")
        else:
            self._thinking_warn_label.config(text="")
            self._thinking_warn_tooltip.update_text("")

    # ─── Autoload toggle (footer button) ────────────────────────────────

    def _refresh_autoload_btn(self):
        """Re-paint the Autoload button to reflect the current state.

        Called after any event that can change the state: button click,
        first successful server listen (flips install_verified), startup.
        The button's label stays fixed at "Autoload" — state is encoded
        purely in color + corner_glyph + disabled, so button width never
        jitters between transitions.
        """
        if not hasattr(self, "_autoload_btn"):
            return
        t = self.theme
        verified = bool(self.cfg.get("install_verified", False))
        enabled  = bool(self.cfg.get("autoload",         False))
        if not verified:
            # Locked — user hasn't completed a successful run yet.
            self._autoload_btn.configure_btn(
                color=t.border, state="disabled",
                corner_glyph="🔒")
            self._autoload_tooltip.update_text(
                "Autoload — locked.\n\n"
                "Complete one successful model start first. Once the "
                "llama-server reports 'listening' in the Server Log, "
                "this button unlocks and you can opt into autoload + "
                "CLI remote control (--autostart / --shutdown).")
        elif enabled:
            # ON — IPC listener is running, autoload fires on next start.
            self._autoload_btn.configure_btn(
                color=t.green, state="normal",
                corner_glyph="✓")
            port_info = (f"  (IPC on 127.0.0.1:{self._ipc_port})"
                         if self._ipc_port else "")
            self._autoload_tooltip.update_text(
                f"Autoload — ON.{port_info}\n\n"
                "On the next launcher start the last-used model + GPU "
                "will be launched automatically. TurboQuant also "
                "accepts CLI commands from other tools (e.g. MyIDE):\n"
                "  TurboQuant_QLauncher --autostart\n"
                "  TurboQuant_QLauncher --shutdown\n"
                "  TurboQuant_QLauncher --status\n\n"
                "Click to disable.")
        else:
            # OFF — unlocked but opted out.
            self._autoload_btn.configure_btn(
                color=ACCENT_SOFT, state="normal",
                corner_glyph=None)
            self._autoload_tooltip.update_text(
                "Autoload — OFF.\n\n"
                "Click to enable: the launcher will automatically start "
                "the last-used model + GPU on future launcher starts, "
                "and will accept CLI remote control "
                "(--autostart / --shutdown / --status) from external "
                "tools. A local listening socket on 127.0.0.1 opens "
                "and a lock file appears next to the config.")

    def _on_autoload_click(self):
        """User clicked the Autoload toggle. Flip state + (re)start IPC."""
        if not self.cfg.get("install_verified", False):
            # Should be unreachable (button is disabled in locked state)
            # but guard anyway.
            return
        new_state = not bool(self.cfg.get("autoload", False))
        self.cfg["autoload"] = new_state
        save_config(self.cfg)
        if new_state:
            self._start_ipc_server()
            self._log(
                "Autoload enabled — IPC listener active; the last-used "
                "(model, GPU) will launch automatically on next start.",
                "good")
        else:
            self._stop_ipc_server()
            self._log(
                "Autoload disabled — IPC listener stopped; no automatic "
                "launch on next start.", "info")
        self._refresh_autoload_btn()

    def _trigger_autoload_if_eligible(self):
        """Run the last-used (model, GPU) if the gate permits.

        Gating precedence:
          1. CLI override `--no-autostart` wins over everything (force skip).
          2. CLI override `--autostart` wins over cfg (force run).
          3. Otherwise honour cfg["autoload"].
        In all three "run" paths we additionally require:
          - install_verified (otherwise the launcher has no business
            auto-spawning anything)
          - last_model exists in the current scan
          - no server already running (don't clobber an in-flight launch)
        Each bail-out is logged so the user can see *why* autoload didn't
        fire on startup.
        """
        # 1. CLI override
        if self._autostart_override is False:
            self._log("Autoload skipped (--no-autostart override).", "info")
            return
        want_autoload = (self._autostart_override is True
                         or bool(self.cfg.get("autoload", False)))
        if not want_autoload:
            return
        # 2. Safety gates
        if not self.cfg.get("install_verified", False):
            self._log(
                "Autoload requested but install is not verified yet — "
                "skipping. Start a model manually once to unlock.", "warn")
            return
        if self.server_process:
            self._log(
                "Autoload skipped — a server is already running.", "info")
            return
        # 3. Resolve last model
        last_model = (self.cfg.get("last_model", "") or "").strip()
        last_gpu   = (self.cfg.get("last_gpu_key", "") or "GPU 0").strip()
        if not last_model:
            self._log(
                "Autoload: no previous model recorded — skipping. "
                "Start a model manually once to record it.", "warn")
            return
        model = next((m for m in self.models if m.filename == last_model), None)
        if not model:
            self._log(
                f"Autoload: last model '{last_model}' not found in the "
                f"current scan — skipping.", "warn")
            return
        # 4. Select the matching GPU row in the UI (best-effort; falls
        #    back to row 0 if the GPU is no longer present — e.g. the
        #    user swapped hardware between runs).
        model_idx = self.models.index(model)
        row_idx = 0
        card = self._model_cards.get(last_model)
        if card:
            for i, gpu_row in enumerate(card.get("gpu_rows", [])):
                if gpu_row.get("key") == last_gpu:
                    row_idx = i
                    break
            else:
                self._log(
                    f"Autoload: last GPU '{last_gpu}' not present — "
                    f"falling back to first available row.", "warn")
        try:
            self._select_cell(model_idx, row_idx, scroll="into")
        except Exception:
            # Non-fatal; just means the selection highlight didn't update.
            pass
        self._log(
            f"Autoload: starting {last_model} on {last_gpu}.", "good")
        # Use _on_run_model so the same stop-then-start machinery kicks in
        # that a manual Run click would use.
        self._on_run_model(last_model, last_gpu)

    # ─── IPC server (Autoload gated) ─────────────────────────────────────

    def _start_ipc_server(self):
        """Bind a local TCP control socket and write the lock file.

        No-op if already running. Prefers a previously-assigned port (so
        we don't burn through ephemeral ports on each toggle) but is
        happy with any free port the kernel hands us. Binding errors are
        logged and leave the launcher otherwise functional — Safe
        Settings is essentially a no-op in that case, but the GUI keeps
        running.
        """
        if self._ipc_socket is not None:
            return
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", 0))  # kernel picks a free port
            sock.listen(4)
            self._ipc_socket = sock
            self._ipc_port = sock.getsockname()[1]
        except Exception as e:
            self._log(f"IPC: failed to bind control socket: {e}", "error")
            self._ipc_socket = None
            self._ipc_port = 0
            return

        # Write the lock file so external tools (and a second launcher
        # launch) can find us.
        try:
            with open(LOCK_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "pid": os.getpid(),
                    "port": self._ipc_port,
                    "started": datetime.now().isoformat(timespec="seconds"),
                    "version": APP_VERSION,
                }, f)
        except Exception as e:
            self._log(f"IPC: failed to write lock file: {e}", "warn")

        # Accept loop runs in a daemon thread. Per-connection handlers
        # are also daemons so shutdown doesn't have to reap them.
        def _accept_loop():
            while True:
                try:
                    conn, _addr = sock.accept()
                except OSError:
                    # Socket was closed from _stop_ipc_server — exit loop.
                    return
                except Exception:
                    # Transient error — keep serving.
                    continue
                threading.Thread(
                    target=self._handle_ipc_client,
                    args=(conn,),
                    daemon=True).start()
        self._ipc_accept_thread = threading.Thread(
            target=_accept_loop, daemon=True)
        self._ipc_accept_thread.start()

    def _stop_ipc_server(self):
        """Tear down the control socket + lock file. Safe to call twice."""
        if self._ipc_socket is None:
            # Still clean up a stray lock file if one exists from a prior run.
            self._cleanup_lock_file()
            return
        try:
            self._ipc_socket.close()
        except Exception:
            pass
        self._ipc_socket = None
        self._ipc_port = 0
        self._ipc_accept_thread = None
        self._cleanup_lock_file()

    def _cleanup_lock_file(self):
        try:
            if LOCK_FILE.exists():
                LOCK_FILE.unlink()
        except Exception:
            pass

    def _handle_ipc_client(self, conn):
        """Process one IPC command and close the connection.

        Runs on a worker thread. Any work that touches Tk state is
        dispatched to the main thread via self.after(...). The socket is
        closed unconditionally at the end so a misbehaving client can't
        leak file descriptors.
        """
        try:
            with conn:
                conn.settimeout(5)
                buf = b""
                while b"\n" not in buf and len(buf) < 4096:
                    try:
                        chunk = conn.recv(1024)
                    except socket.timeout:
                        break
                    if not chunk:
                        break
                    buf += chunk
                command = buf.decode("utf-8", errors="replace").strip()
                parts = command.split()
                if not parts:
                    try:
                        conn.sendall(b"ERR empty\n")
                    except Exception:
                        pass
                    return
                cmd = parts[0].upper()

                if cmd == "SHUTDOWN":
                    # Reply first, THEN schedule the shutdown, so the
                    # caller sees a clean "OK" before we tear down. The
                    # 100 ms delay gives the reply time to flush.
                    try:
                        conn.sendall(b"OK shutting down\n")
                    except Exception:
                        pass
                    self.after(100, self._on_close)

                elif cmd == "AUTOSTART":
                    # Optional extension: AUTOSTART MODEL GPU
                    # lets MyIDE override the recorded last model.
                    new_model = parts[1] if len(parts) > 1 else None
                    new_gpu   = parts[2] if len(parts) > 2 else None
                    def _fire():
                        if new_model:
                            self.cfg["last_model"] = new_model
                        if new_gpu:
                            self.cfg["last_gpu_key"] = new_gpu
                        if new_model or new_gpu:
                            save_config(self.cfg)
                        # Temporarily force-on the override for this run
                        # regardless of the checkbox state, then fire.
                        prev = self._autostart_override
                        self._autostart_override = True
                        try:
                            self._trigger_autoload_if_eligible()
                        finally:
                            self._autostart_override = prev
                    try:
                        conn.sendall(b"OK starting\n")
                    except Exception:
                        pass
                    self.after(50, _fire)

                elif cmd == "STATUS":
                    status = {
                        "version":       APP_VERSION,
                        "pid":           os.getpid(),
                        "ipc_port":      self._ipc_port,
                        "running_model": self.running_model,
                        "running_gpu":   self._running_gpu_key,
                        "running_port":  getattr(self, "_running_port", None),
                        "last_model":    self.cfg.get("last_model", ""),
                        "last_gpu_key":  self.cfg.get("last_gpu_key", ""),
                        "autoload":      bool(self.cfg.get("autoload", False)),
                        "install_verified":
                            bool(self.cfg.get("install_verified", False)),
                    }
                    try:
                        conn.sendall(
                            (json.dumps(status) + "\n").encode("utf-8"))
                    except Exception:
                        pass

                else:
                    try:
                        conn.sendall(b"ERR unknown command\n")
                    except Exception:
                        pass
        except Exception:
            # Never let an IPC client crash the launcher.
            pass

    def _on_close(self):
        self._save_current_config()
        # v0.56 — Tear down the IPC listener *before* we destroy Tk, so
        # an in-flight AUTOSTART command can't schedule main-thread work
        # on a dying interpreter.
        self._stop_ipc_server()
        if self.server_process:
            self._stop_server(silent=True)
        self.destroy()

    def _restore_sash(self):
        """Restore the vertical splitter position from config."""
        try:
            sash_y = int(self.cfg["sash_pos"])
            self._paned.sash_place(0, 0, sash_y)
        except Exception:
            pass

    @staticmethod
    def _center_window(win, w=None, h=None):
        win.update_idletasks()
        w = w or win.winfo_reqwidth()
        h = h or win.winfo_reqheight()
        sx = win.winfo_screenwidth()
        sy = win.winfo_screenheight()
        x = (sx - w) // 2
        y = (sy - h) // 2
        win.geometry(f"{w}x{h}+{x}+{y}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: Main
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_cli_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the launcher.

    Split into its own function so main() stays readable and so we can
    unit-test argument parsing without spinning up Tk.

    Flags:
      --autostart / -a     Force one-shot autoload this session regardless
                           of the Autoload toggle state. If another
                           instance is already running with Autoload ON,
                           the command is forwarded via IPC and the new
                           process exits. Requires install_verified;
                           otherwise the launcher opens normally with a
                           warning logged.
      --no-autostart       Force-skip autoload even if the toggle is ON.
                           Useful for opening the GUI to tweak settings
                           without kicking off a model load.
      --shutdown / -q      Ask a running instance to quit cleanly (stop
                           llama-server, save config, close window). No
                           dialog, no confirmation. Exits 0 on success,
                           exit 1 if no reachable instance.
      --status             Print a JSON status report from the running
                           instance to stdout and exit. Exit 1 if no
                           reachable instance.
      --version            Print version and exit.
    """
    p = argparse.ArgumentParser(
        prog="TurboQuant_QLauncher",
        description=(
            "TurboQuant QLauncher — llama-server model switcher with "
            "TurboQuant KV-cache support. CLI mode is intended for "
            "external tools (e.g. MyIDE) and requires 'Autoload' to be "
            "enabled in the GUI first."),
        # Keep the epilog readable — argparse wraps it automatically.
        epilog=(
            "Usage examples:\n"
            "  TurboQuant_QLauncher                  open the GUI normally\n"
            "  TurboQuant_QLauncher --autostart      open + start last model\n"
            "  TurboQuant_QLauncher --shutdown       ask running instance to quit\n"
            "  TurboQuant_QLauncher --status         print instance status as JSON"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Mutually-exclusive autostart group so -a and --no-autostart can't
    # both be passed in the same call (would be nonsensical).
    g = p.add_mutually_exclusive_group()
    g.add_argument("-a", "--autostart", action="store_true",
                   help="Force autoload for this session.")
    g.add_argument("--no-autostart", action="store_true",
                   help="Skip autoload for this session.")
    p.add_argument("-q", "--shutdown", action="store_true",
                   help="Tell a running instance to close cleanly and exit.")
    p.add_argument("--status", action="store_true",
                   help="Print JSON status from the running instance, "
                        "then exit.")
    p.add_argument("--version", action="version",
                   version=f"TurboQuant QLauncher v{APP_VERSION}")
    return p.parse_args(argv)


def _handle_cli_only_commands(args: argparse.Namespace) -> Optional[int]:
    """Handle commands that don't need a GUI: --shutdown, --status.

    Returns an exit code (0 or 1) if the command was fully handled here
    and the caller should exit now. Returns None when the launcher must
    proceed to open the GUI (covers plain launch and --autostart).
    """
    if not (args.shutdown or args.status):
        return None

    lock = read_lock_file()
    if lock is None:
        # No running instance. For --shutdown we treat "nothing to do" as
        # success (0): MyIDE can call this unconditionally on its own
        # shutdown path without racing a startup. For --status it's a
        # genuine failure (1): the caller explicitly wants a report.
        if args.shutdown:
            print("TurboQuant is not running.", file=sys.stderr)
            return 0
        else:
            print("TurboQuant is not running.", file=sys.stderr)
            return 1

    port = lock.get("port", 0)

    if args.shutdown:
        reply = send_ipc_command(port, "SHUTDOWN")
        if reply is None:
            print(
                f"Could not reach running instance on 127.0.0.1:{port}. "
                f"Is Autoload still enabled?", file=sys.stderr)
            return 1
        # Best-effort wait for the instance to actually go away. The
        # IPC handler sends "OK" before the 100 ms teardown hook fires,
        # so the socket is still reachable for a moment. We poll the
        # lock file (which is removed in _on_close → _stop_ipc_server).
        deadline = time.time() + 10
        while time.time() < deadline:
            if not LOCK_FILE.exists():
                break
            # Re-read will also purge a stale lock if the PID died.
            if read_lock_file() is None:
                break
            time.sleep(0.1)
        print("TurboQuant closed.")
        return 0

    if args.status:
        reply = send_ipc_command(port, "STATUS")
        if reply is None:
            print(
                f"Could not reach running instance on 127.0.0.1:{port}.",
                file=sys.stderr)
            return 1
        # Reply is a single line of JSON. Print verbatim so callers can
        # pipe it into jq / their own JSON parser.
        print(reply)
        return 0

    return None


def main():
    args = _parse_cli_args()

    # CLI-only paths (shutdown/status) take precedence over GUI startup.
    # They neither require nor benefit from Tk, so we short-circuit here.
    rc = _handle_cli_only_commands(args)
    if rc is not None:
        sys.exit(rc)

    # v0.56 — Single-instance coordination for --autostart. If an
    # instance with Autoload is already running, forward the
    # AUTOSTART command and exit; don't try to spin up a second Tk.
    # Plain launches (no --autostart) just open a second window on
    # purpose — matches legacy behaviour.
    if args.autostart:
        lock = read_lock_file()
        if lock is not None:
            reply = send_ipc_command(lock["port"], "AUTOSTART")
            if reply is not None:
                print("TurboQuant already running — autostart forwarded.")
                sys.exit(0)
            # Fell through — existing instance is unreachable. Proceed
            # to start a new one rather than failing.
            print(
                "Existing instance did not reply; starting a new one.",
                file=sys.stderr)

    # ─── DPI awareness (MUST run before any Tk window is created) ─────────
    # Without this, Windows applies DWM bitmap scaling on HiDPI displays:
    # Tk renders at 96 DPI into a small bitmap, Windows bilinearly upscales
    # → blurry text ("unscharf"). Declaring System-DPI-Aware disables the
    # bitmap upscale so text renders at native resolution. Per-Monitor-V2
    # would be nicer but Tk doesn't handle WM_DPICHANGED, so staying at
    # level 1 is the right trade-off.
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            # Pre-Windows-8.1 or shcore missing — fall back to the legacy
            # user32 API which exists since Vista.
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

    # Hide the console window on Windows when launched via py.exe / python.exe
    if sys.platform == "win32":
        try:
            import ctypes
            hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            if hwnd:
                ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE
        except Exception:
            pass

    # Translate CLI flags to the __init__ override triple-state:
    #   --autostart    -> True
    #   --no-autostart -> False
    #   (neither)      -> None
    if args.autostart:
        autostart_override: Optional[bool] = True
    elif args.no_autostart:
        autostart_override = False
    else:
        autostart_override = None

    app = TurboQuantQLauncher(autostart_override=autostart_override)
    app.mainloop()

if __name__ == "__main__":
    main()
