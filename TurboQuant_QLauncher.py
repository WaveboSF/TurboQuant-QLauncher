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

APP_VERSION = "0.50"

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

KV_CACHE_OPTIONS = {
    "f16 (default)":       {"ctk": None,     "ctv": None},
    "q8_0-K + turbo4-V":   {"ctk": "q8_0",   "ctv": "turbo4"},
    "turbo3 / turbo3":     {"ctk": "turbo3",  "ctv": "turbo3"},
    "turbo4 / turbo4":     {"ctk": "turbo4",  "ctv": "turbo4"},
    "q8_0-K + turbo3-V":   {"ctk": "q8_0",   "ctv": "turbo3"},
    "q8_0 / q8_0":         {"ctk": "q8_0",   "ctv": "q8_0"},
}

# Context depth list for the Bench All matrix.
# Each depth value is passed to llama-bench as `-d <n>`, which pre-fills the
# KV-cache with N tokens before running the tg (decode) test. This reproduces
# Madreag's and TheTom's published benchmark methodology from Discussion #20969
# and directly exposes TurboQuant's long-context decode-speed advantage.
#
# With 6 KV × 4 LA × 3 depths = 72 runs per "Bench All" pass.
# Edit here to add/remove depths. Keep 0 as the first entry (short-context
# baseline) — it stays comparable to earlier benchmark entries in the log.
BENCH_DEPTH_LIST = [0, 8192, 32768]

# KV-Cache compression factor relative to f16 (1.0 = no compression).
# K and V keys are compressed independently; factor = mean(K_ratio, V_ratio).
#   f16  = 1.0,  q8_0 = 0.5,  turbo4 = 0.25,  turbo3 = 0.20
KV_COMPRESSION: Dict[str, float] = {
    "f16 (default)":     1.000,   # no compression
    "q8_0 / q8_0":       0.500,   # both halved
    "q8_0-K + turbo4-V": 0.375,   # mean(0.5, 0.25)
    "q8_0-K + turbo3-V": 0.350,   # mean(0.5, 0.20)
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
FONT_TITLE    = (_MONO, 15, "bold")
FONT_SUBTITLE = (_MONO, 12)
FONT_HEADER   = (_MONO, 11, "bold")
FONT_BODY     = (_MONO, 11)
FONT_BODY_B   = (_MONO, 11, "bold")
FONT_SMALL    = (_MONO, 10)
FONT_SMALL_B  = (_MONO, 10, "bold")
FONT_DIM      = (_MONO, 10)
FONT_BRAILLE  = (_MONO, 12)

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
                    gpus.append(GPUInfo(index=idx, name=name, vendor=vendor, vram_mb=vram))
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

def scan_models(models_dirs, recursive: bool = True) -> List[ModelInfo]:
    """Scan one or more directories for *.gguf files.

    Accepts either a single path (str) or a list of paths. When
    ``recursive`` is True (default), descends into subdirectories via
    ``os.walk``; otherwise only the top level of each directory is scanned.
    Results are deduplicated by canonical (realpath) location, so overlapping
    directories or symlinks won't list the same file twice.
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
            walker = os.walk(models_dir)
        else:
            try:
                entries = os.listdir(models_dir)
            except OSError:
                continue
            walker = [(models_dir, [], entries)]
        for root, _dirs, files in walker:
            for f in files:
                if not f.lower().endswith(".gguf"):
                    continue
                full_path = os.path.join(root, f)
                try:
                    key = os.path.realpath(full_path)
                    if key in seen:
                        continue
                    size = os.path.getsize(full_path)
                    seen[key] = ModelInfo(
                        filename=f,
                        path=full_path,
                        size_bytes=size,
                        size_gb=round(size / (1024**3), 1),
                        has_unsafe_path=("," in full_path),
                    )
                except OSError:
                    pass
    models = list(seen.values())
    models.sort(key=lambda m: m.size_bytes, reverse=True)
    return models

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
# SECTION: Main Application
# ═══════════════════════════════════════════════════════════════════════════════

class TurboQuantQLauncher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"TurboQuant QLauncher v{APP_VERSION}")
        self.is_dark = detect_system_dark_mode()
        self.theme = DARK_THEME if self.is_dark else LIGHT_THEME
        self._first_run = not CONFIG_FILE.exists()
        self.cfg = load_config()
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
        self._nut_rows_frame: Optional[tk.Frame] = None
        self._nut_phase2_placeholder: Optional[tk.Frame] = None
        # qnut v0.50 — concurrency guard. Set True at the very start of
        # _nut_worker, reset False in its finally block (so it always
        # gets cleared even on cancel/error). All entry points that
        # could otherwise launch a competing server (Probe button, Run
        # button, model double-click, Q/N/C mode buttons) check this
        # flag first and refuse to act while a probe is in progress.
        self._probe_in_progress: bool = False
        # qnut v0.50 — reasoning-model warning bypass. The Probe worker
        # forces No-Thinking on every server start (for deterministic
        # outputs) but the existing _start_server warning would pop up
        # before EVERY tuple. This flag suppresses the warning for the
        # duration of a single Probe run; it is set in the worker after
        # the user confirms the warning ONCE, and cleared in the same
        # finally block that releases _probe_in_progress.
        self._probe_thinking_warning_acknowledged: bool = False

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

        w = self.cfg.get("window_w", 1000)
        h = self.cfg.get("window_h", 800)
        self.geometry(f"{w}x{h}")
        self.minsize(860, 600)
        if self.cfg.get("window_x") is not None:
            self.geometry(f"+{self.cfg['window_x']}+{self.cfg['window_y']}")
        else:
            self._center_window(self, w, h)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        if self._first_run:
            self.after(100, self._first_run_setup)
        else:
            self.after(100, self._do_initial_scan)
        if self.cfg.get("sash_pos") is not None:
            self.after(150, self._restore_sash)

    # ─── Theme ────────────────────────────────────────────────────────────

    def _configure_theme(self):
        t = self.theme
        self.configure(bg=t.bg)
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

    # ─── Settings Bar ─────────────────────────────────────────────────────

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

        row = tk.Frame(self, bg=t.bg)
        row.pack(fill="x", padx=16, pady=(8, 4))

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
        mode_frame = tk.Frame(row, bg=t.bg)
        mode_frame.pack(side="left", padx=(8, 0))
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
            btn = HoverButton(mode_frame, t, text=mode_label,
                              color=t.border, width=28, height=24,
                              command=lambda k=mode_key: self._apply_nut_mode(k))
            btn.pack(side="left", padx=1)
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
        # The Probe worker will force No-Thinking on EVERY tuple to get
        # deterministic outputs (otherwise the differential comparison
        # sees noise from <think> blocks that vary even at temp=0). The
        # standard _start_server warning would fire before every single
        # tuple, blocking the worker each time.
        #
        # Solution: ask the user ONCE here, before the run starts, with
        # an honest probe-specific explanation of the trade-off. Their
        # answer is remembered for the duration of this Probe run only,
        # via the _probe_thinking_warning_acknowledged flag which the
        # worker sets after this dialog returns Yes and clears in its
        # finally block.
        #
        # Reset the flag defensively so a previous Probe run's "yes"
        # doesn't carry over.
        self._probe_thinking_warning_acknowledged = False

        if _is_reasoning_model(model.filename):
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
            # User accepted — set the bypass flag so _start_server
            # doesn't re-prompt before every tuple. The worker will
            # clear it in its finally block.
            self._probe_thinking_warning_acknowledged = True

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
            # Clear the reasoning-warning bypass so the next normal
            # _start_server call (outside of a Probe run) sees the
            # warning again. Important: this MUST be cleared even on
            # cancel/exception so a future non-Probe server start with
            # No-Thinking on a reasoning model is not silently allowed.
            self._probe_thinking_warning_acknowledged = False
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
        """
        if not hasattr(self, "_mode_buttons") or not self._mode_buttons:
            return
        t = self.theme
        has_profile = bool(self._current_nut_profile)
        for mode_key, btn in self._mode_buttons.items():
            if not has_profile:
                btn.configure_btn(color=t.border, state="disabled")
                continue
            anchor = self._current_nut_profile.get(mode_key)
            if not isinstance(anchor, dict) or not anchor.get("kv"):
                btn.configure_btn(color=t.border, state="disabled")
                continue
            is_active = (self._active_mode == mode_key)
            btn.configure_btn(
                color=ACCENT_TURBO if is_active else t.border,
                state="normal",
            )

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
        self._canvas.bind_all("<MouseWheel>",
                              lambda e: self._canvas.yview_scroll(-1 * (e.delta // 120), "units"))

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

    def _build_footer(self):
        t = self.theme
        # side="bottom" anchors the footer to the window bottom edge
        # and reserves its height BEFORE the paned window expands into the
        # remaining space. This prevents the footer from being clipped when
        # the user shrinks the window or when the model list grows.
        bar = ttk.Frame(self)
        bar.pack(side="bottom", fill="x", padx=16, pady=(6, 8))
        tk.Frame(self, height=1, bg=t.border).pack(side="bottom", fill="x")

        _BTN_W, _BTN_H = 130, 28
        HoverButton(bar, t, text="Rescan Models", color=ACCENT_SOFT,
                    width=_BTN_W, height=_BTN_H,
                    command=self._rescan_models).pack(side="left", padx=2)
        HoverButton(bar, t, text="Update Binaries", color=ACCENT_SOFT,
                    width=_BTN_W, height=_BTN_H,
                    command=self._update_binaries).pack(side="left", padx=2)

        # Quick-switch slot buttons for bookmarked llama-server.exe paths.
        # Populated dynamically from self.cfg["server_slots"]. Rebuilt whenever
        # the paths dialog is saved or a slot is clicked (for active highlight).
        # Small left margin to separate visually from "Update Binaries".
        self._slot_bar = tk.Frame(bar, bg=t.bg)
        self._slot_bar.pack(side="left", padx=(10, 0))
        self._slot_buttons: list = []  # filled by _refresh_slot_buttons

        HoverButton(bar, t, text="About", color=ACCENT_SOFT,
                    width=80, height=_BTN_H,
                    command=self._show_about).pack(side="right", padx=2)
        HoverButton(bar, t, text="Paths", color=ACCENT_SOFT,
                    width=80, height=_BTN_H,
                    command=self._show_paths_dialog).pack(side="right", padx=2)

        self._save_bench_btn = HoverButton(bar, t, text="💾  Save Results...",
                                            color=t.border, width=168, height=_BTN_H,
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
        for slot_path in slots:
            if not slot_path or not slot_path.strip():
                continue
            label = slot_label_from_path(slot_path)
            if not label:
                continue
            # Width scales with label length. Base 20px padding + ~8px per char.
            # Clamped to [90, 180] so very long names don't blow out the layout.
            btn_w = max(90, min(180, len(label) * 8 + 20))

            # Highlight active slot
            slot_norm = os.path.normcase(os.path.normpath(slot_path))
            is_active = (slot_norm == active_norm) and bool(active_norm)
            color = t.accent if is_active else ACCENT_SOFT

            # Missing-file indicator: still show the button, but in muted color
            # so the user knows the slot is broken. Click will log an error.
            if not os.path.isfile(slot_path):
                color = t.border  # muted/disabled look

            btn = HoverButton(self._slot_bar, t, text=label, color=color,
                              width=btn_w, height=_BTN_H,
                              command=lambda p=slot_path: self._on_slot_click(p))
            btn.pack(side="left", padx=2)
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
                    "CPU RAM", t.fg_dim,
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

    def _select_cell(self, model_idx: int, row_idx: int):
        """Select a specific GPU row within a model card."""
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

        # Force a full geometry update of the entire scrollable area —
        # not just rd["frame"]. This is required for two reasons:
        #   1. _inner_frame.winfo_height() returns 1 if the geometry
        #      manager hasn't laid out the inner frame yet, which is
        #      always the case on the very first auto-selection right
        #      after _populate_models() builds the cards. Without this
        #      update we compute frame_y / canvas_h / inner_h against
        #      stale geometry and yview_moveto leaves a visible empty
        #      band above the first card.
        #   2. The canvas scrollregion is set lazily via the <Configure>
        #      binding on _inner_frame, which fires on the next event-
        #      loop iteration — too late for an inline yview_moveto
        #      that runs in the same call. Setting the scrollregion
        #      explicitly here makes the scroll math consistent.
        self._inner_frame.update_idletasks()
        bbox = self._canvas.bbox("all")
        if bbox:
            self._canvas.configure(scrollregion=bbox)

        frame_y = rd["frame"].winfo_y() + card_data["card"].winfo_y()
        canvas_h = self._canvas.winfo_height()
        inner_h = max(1, self._inner_frame.winfo_height())
        self._canvas.yview_moveto(max(0, (frame_y - canvas_h // 3)) / inner_h)

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

    def _on_key_up(self, event):
        if self._sel_model > 0:
            self._select_cell(self._sel_model - 1, self._sel_row)

    def _on_key_down(self, event):
        if self._sel_model < len(self.models) - 1:
            self._select_cell(self._sel_model + 1, self._sel_row)

    def _on_key_left(self, event):
        if self._sel_row > 0:
            self._select_cell(self._sel_model, self._sel_row - 1)

    def _on_key_right(self, event):
        fn = self.models[self._sel_model].filename if self._sel_model >= 0 else None
        card_data = self._model_cards.get(fn) if fn else None
        max_row = len(card_data["gpu_rows"]) - 1 if card_data else 0
        if self._sel_row < max_row:
            self._select_cell(self._sel_model, self._sel_row + 1)

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
            # Guard against silently disabling thinking on reasoning
            # models. Gemma 4 26B-A4B accuracy drops from ~97% to ~64% on
            # the math suite when thinking is off — warn the user before
            # launching. They can still proceed if it's intentional.
            #
            # qnut v0.50: skip this warning if the Probe worker has
            # already obtained explicit acknowledgement from the user
            # at the start of its run. Without this bypass the dialog
            # would re-appear before every single tuple (potentially
            # 24 times per Probe run) and block the worker on each one.
            if (_is_reasoning_model(filename)
                    and not self._probe_thinking_warning_acknowledged):
                proceed = messagebox.askyesno(
                    "Reasoning Model — Thinking disabled",
                    f"'{filename}' looks like a reasoning-capable model "
                    f"(Gemma 4 / Qwen3 / DeepSeek-R1 / QwQ family).\n\n"
                    f"'No Thinking' is currently ENABLED, which can drop "
                    f"accuracy dramatically on math and logic tasks "
                    f"(Gemma 4 26B: ~97% → ~64% on our suite).\n\n"
                    f"Start the server with thinking disabled anyway?",
                    icon="warning",
                    default="no",
                )
                if not proceed:
                    self._log(
                        f"Server start cancelled by user — "
                        f"'No Thinking' flag guard on reasoning model",
                        "warn",
                    )
                    return
            cmd.extend(["--reasoning", "off"])

        port = self._port_var.get()
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
                    self.after(0, self._log, line, tag)
        except Exception:
            pass
        self.after(0, self._on_server_exited)

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

        tk.Label(dlg, text=f"LLM Models Directories (up to {MAX_MODELS_PATHS}):",
                 font=FONT_BODY_B, bg=t.bg, fg=t.fg).pack(padx=20, anchor="w")
        tk.Label(dlg,
                 text="All configured directories are scanned together. "
                      "Duplicate files are detected automatically.",
                 font=FONT_BODY, bg=t.bg, fg=t.fg_dim,
                 wraplength=680, justify="left").pack(padx=20, anchor="w", pady=(0, 4))

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

        tk.Label(dlg, text="llama-server.exe Path (active):", font=FONT_BODY_B,
                 bg=t.bg, fg=t.fg).pack(padx=20, anchor="w")
        server_frame = tk.Frame(dlg, bg=t.bg)
        server_frame.pack(fill="x", padx=20, pady=(2, 8))
        server_var = tk.StringVar(value=self.cfg.get("llama_server_path", ""))
        server_entry = ttk.Entry(server_frame, textvariable=server_var, font=FONT_BODY)
        server_entry.pack(side="left", fill="x", expand=True)
        HoverButton(server_frame, t, text="...", color=t.border, width=36, height=24,
                    command=lambda: self._browse_file(server_var)).pack(side="left", padx=(4, 0))

        # ── Quick-switch slots ──
        tk.Frame(dlg, height=1, bg=t.border).pack(fill="x", padx=16, pady=(4, 8))
        tk.Label(dlg, text=f"Quick-Switch Slots (up to {MAX_SERVER_SLOTS}):",
                 font=FONT_BODY_B, bg=t.bg, fg=t.fg).pack(padx=20, anchor="w")
        tk.Label(dlg,
                 text="Bookmark additional llama-server.exe builds here — "
                      "they appear as one-click buttons in the footer.",
                 font=FONT_BODY, bg=t.bg, fg=t.fg_dim,
                 wraplength=680, justify="left").pack(padx=20, anchor="w", pady=(0, 4))

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
            self.cfg["llama_server_path"] = server_var.get()
            # persist slot paths (strip whitespace, keep empty slots)
            self.cfg["server_slots"] = [v.get().strip() for v in slot_vars]
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
        self._center_window(dlg, 720, dlg.winfo_reqheight())

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

    def _on_close(self):
        self._save_current_config()
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

def main():
    # Hide the console window on Windows when launched via py.exe / python.exe
    if sys.platform == "win32":
        try:
            import ctypes
            hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            if hwnd:
                ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE
        except Exception:
            pass

    app = TurboQuantQLauncher()
    app.mainloop()

if __name__ == "__main__":
    main()
