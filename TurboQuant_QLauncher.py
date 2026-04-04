#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TurboQuant QLauncher v0.41     (c) WaveboSF 2026
=============================================
Model Switcher & Server Manager for llama-server with TurboQuant KV-Cache.

Standalone GUI with zero external dependencies.
Uses only Python stdlib (tkinter/ttk) — runs anywhere Python runs.

Features:
- Auto-scan GGUF models from configurable directory
- Per-model GPU selector with VRAM fit indicator (Braille bars)
- One-click model switching (auto-stops previous server)
- KV-Cache config: f16, q8_0+turbo4, turbo3+turbo3, turbo4+turbo4
- NVIDIA + AMD GPU detection (nvidia-smi, rocm-smi, WMI, lspci)
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
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: Constants
# ═══════════════════════════════════════════════════════════════════════════════

APP_VERSION = "0.41"

def get_launcher_dir() -> Path:
    """Return the directory where the launcher .py or compiled .exe lives.

    Works correctly for three execution modes:
      1. python TurboQuant_QLauncher.py   → directory of the .py file
      2. Nuitka --standalone              → directory of the .exe
      3. Nuitka --onefile                 → directory of the .exe
                                            (NOT the temp extraction dir)

    Nuitka sets the global ``__compiled__`` and also ``sys.frozen = True``.
    In --onefile mode ``__file__`` points to the temp dir, so we must use
    ``sys.executable`` (which always holds the real exe path).
    """
    if getattr(sys, "frozen", False) or globals().get("__compiled__"):
        # Compiled by Nuitka (standalone or onefile)
        return Path(sys.executable).resolve().parent
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

DEFAULT_CONFIG = {
    "llm_models_path": "",
    "llama_server_path": "",
    "kv_cache": "q8_0-K + turbo4-V",
    "port": 8080,
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
    """Colored button with hover effect, drawn on Canvas for full color control."""

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
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonRelease-1>", self._on_click)
        self._draw()

    def configure_btn(self, text=None, color=None, state=None, command=None):
        if text is not None:
            self._text = text
        if color is not None:
            self._color = color
        if state is not None:
            self._disabled = (state == "disabled")
            self.configure(cursor="" if self._disabled else "hand2")
        if command is not None:
            self._command = command
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
        self._rounded_rect(1, 1, w - 1, h - 1, r, fill=bg, outline="")
        self.create_text(w // 2, h // 2, text=self._text, fill=fg,
                         font=self._font, anchor="center")

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

def scan_models(models_dir: str) -> List[ModelInfo]:
    """Scan directory recursively for *.gguf files."""
    models = []
    if not os.path.isdir(models_dir):
        return models
    for root, dirs, files in os.walk(models_dir):
        for f in files:
            if f.lower().endswith(".gguf"):
                full_path = os.path.join(root, f)
                try:
                    size = os.path.getsize(full_path)
                    models.append(ModelInfo(
                        filename=f,
                        path=full_path,
                        size_bytes=size,
                        size_gb=round(size / (1024**3), 1),
                    ))
                except OSError:
                    pass
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
    return cfg

def save_config(cfg: dict):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

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

        self._configure_theme()
        self._build_header()
        self._build_settings_bar()
        self._paned = tk.PanedWindow(self, orient=tk.VERTICAL,
                                      bg=self.theme.border,
                                      sashwidth=5, sashrelief="flat",
                                      handlesize=0, showhandle=False)
        self._paned.pack(fill="both", expand=True, padx=16, pady=(4, 4))
        self._build_model_list()
        self._build_log_area()
        self._build_footer()
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

        self._gpu_label = ttk.Label(hdr, text="Detecting GPUs...", font=FONT_SMALL,
                                     foreground=t.fg_secondary, style="H.TLabel")
        self._gpu_label.pack(fill="x", padx=16, pady=(0, 1))
        self._cuda_label = ttk.Label(hdr, text="", font=FONT_SMALL,
                                      foreground=t.fg_dim, style="H.TLabel")
        self._cuda_label.pack(fill="x", padx=16, pady=(0, 6))
        ttk.Separator(self).pack(fill="x")

    # ─── Settings Bar ─────────────────────────────────────────────────────

    def _build_settings_bar(self):
        t = self.theme
        self._sel_model = -1
        self._sel_row = 0
        self._running_gpu_key = None

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

        port_label = tk.Label(row, text="  Port:", font=FONT_BODY_B, bg=t.bg, fg=t.fg)
        port_label.pack(side="left", padx=(8, 0))
        ToolTip(port_label, "OpenAI-compatible API port (default: 8080)", t)
        self._port_var = tk.StringVar(value=str(self.cfg.get("port", 8080)))
        port_entry = tk.Entry(row, textvariable=self._port_var, width=5, font=FONT_BODY,
                              bg=t.entry_bg, fg=t.fg, relief="flat", bd=2,
                              insertbackground=t.fg)
        port_entry.pack(side="left", padx=(4, 0))

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
        ToolTip(self._run_btn, "Start benchmark with selected KV × LA configs", t)

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
        t = self.theme
        if self._bench_all_var.get():
            # Toggle mode: add/remove from bench set
            if name in self._bench_kv_set:
                self._bench_kv_set.discard(name)
            else:
                self._bench_kv_set.add(name)
            for n, btn in self._kv_buttons.items():
                btn.configure_btn(color=ACCENT_TURBO if n in self._bench_kv_set else t.border)
        else:
            # Single-select mode (normal)
            self._kv_var.set(name)
            for n, btn in self._kv_buttons.items():
                btn.configure_btn(color=ACCENT_TURBO if n == name else t.border)
            self._update_bars_for_kv(name)

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

    def _on_bench_all_toggled(self, *args):
        """Called when Bench All checkbox is toggled."""
        t = self.theme
        if self._bench_all_var.get():
            # Activate: enable Benchmark, select ALL KV and LA
            self._bench_var.set(True)
            self._bench_kv_set = set(KV_CACHE_OPTIONS.keys())
            self._bench_la_set = set(self._la_buttons.keys())
            for btn in self._kv_buttons.values():
                btn.configure_btn(color=ACCENT_TURBO)
            for btn in self._la_buttons.values():
                btn.configure_btn(color=ACCENT_TURBO)
        else:
            # Deactivate: revert to single-select visuals
            self._bench_kv_set.clear()
            self._bench_la_set.clear()
            cur_kv = self._kv_var.get()
            for n, btn in self._kv_buttons.items():
                btn.configure_btn(color=ACCENT_TURBO if n == cur_kv else t.border)
            cur_la = self._la_var.get()
            for m, btn in self._la_buttons.items():
                btn.configure_btn(color=ACCENT_TURBO if m == cur_la else t.border)

    def _bench_run(self):
        """Run button: start benchmark with selected configs."""
        if self._bench_thread and self._bench_thread.is_alive():
            self._log("Benchmark already running.", "warn")
            return
        if self.server_process:
            self._log("Server is running — stop first (Escape)", "warn")
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
            # Bench All mode: KV × LA matrix
            kv_list = [k for k in KV_CACHE_OPTIONS if k in self._bench_kv_set]
            la_list = sorted(self._bench_la_set)
            n_runs = len(kv_list) * len(la_list)
            est_min = (n_runs * 45) // 60
            est_sec = (n_runs * 45) % 60
            msg = (f"Run Benchmark ALL?\n\n"
                   f"Model: {model.filename}\n"
                   f"Target: {gpu_display}\n"
                   f"KV configs: {len(kv_list)}   LA modes: {len(la_list)}\n"
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
        else:
            self._log("Check 'Benchmark' or 'Bench All' first.", "warn")

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
        tk.Frame(self, height=1, bg=t.border).pack(fill="x")
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=16, pady=(6, 8))

        _BTN_W, _BTN_H = 130, 28
        HoverButton(bar, t, text="Rescan Models", color=ACCENT_SOFT,
                    width=_BTN_W, height=_BTN_H,
                    command=self._rescan_models).pack(side="left", padx=2)
        HoverButton(bar, t, text="Update Binaries", color=ACCENT_SOFT,
                    width=_BTN_W, height=_BTN_H,
                    command=self._update_binaries).pack(side="left", padx=2)

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

        self._models_path_label.config(text=f"({self.cfg['llm_models_path']})")

        for i, model in enumerate(self.models):
            bg = t.bg_secondary if i % 2 == 0 else t.bg
            card = tk.Frame(self._inner_frame, bg=bg)
            card.pack(fill="x", pady=1)

            hdr = tk.Frame(card, bg=bg)
            hdr.pack(fill="x", padx=10, pady=(6, 2))
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
            self._select_cell(0, 0)

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

        rd["frame"].update_idletasks()
        frame_y = rd["frame"].winfo_y() + card_data["card"].winfo_y()
        canvas_h = self._canvas.winfo_height()
        inner_h = max(1, self._inner_frame.winfo_height())
        self._canvas.yview_moveto(max(0, (frame_y - canvas_h // 3)) / inner_h)

    def _launch_selected(self):
        """Launch the currently selected model on the selected GPU."""
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
        """Background thread: run llama-bench for KV × LA matrix."""
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
        total = len(kv_list) * len(la_list)

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

            for kv_name in kv_list:
                if self._bench_stop_event.is_set():
                    break
                step += 1
                kv = KV_CACHE_OPTIONS.get(kv_name, {})
                ctk = kv.get("ctk")
                ctv = kv.get("ctv")

                self.after(0, self._log,
                           f"  ▸ [{step}/{total}] {kv_name} ({la_tag})...", "info")
                self.after(0, lambda s=step, t=total:
                           self._status_label.config(
                               text=f"● Bench All [{s}/{t}]...",
                               fg=self.theme.yellow))

                result = self._exec_bench(bench_exe, model.path, ngl, ctk, ctv, env)
                if result:
                    key = f"{kv_name}|{la_tag}"
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

    def _exec_bench(self, bench_exe: str, model_path: str, ngl: str,
                     ctk: Optional[str], ctv: Optional[str], env: dict) -> Optional[dict]:
        """Execute llama-bench and parse output. Returns {pp512: str, tg128: str} or None."""
        cmd = [bench_exe, "-m", model_path, "-ngl", ngl]
        if ctk:
            cmd.extend(["-ctk", ctk])
        if ctv:
            cmd.extend(["-ctv", ctv])

        try:
            kw = {"capture_output": True, "timeout": 300,
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
        is_bench_all = kv_name == "ALL" and any("|" in k for k in results)

        if is_bench_all:
            # Group by LA tag, find f16 baseline per group
            from collections import OrderedDict
            groups = OrderedDict()
            for key, data in results.items():
                parts = key.split("|", 1)
                kv_part = parts[0]
                la_part = parts[1] if len(parts) > 1 else ""
                groups.setdefault(la_part, []).append((kv_part, data))

            rows = []
            first_group = True
            for la_tag, entries in groups.items():
                # Find f16 baseline in this LA group
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

                if len(groups) > 1:
                    if not first_group:
                        rows.append(("", "", "", "", ""))  # empty separator row
                    rows.append((f"**{la_tag}**", "", "", "", ""))
                    first_group = False

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
                    f.write("# TurboQuant Benchmark Log\n\n")
                    f.write("Persistent benchmark results from TurboQuant QLauncher.\n\n")
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

                f.write(f"### {model.filename} — {gpu_label}, Build {build_tag}{driver_tag}, {ts}\n\n")
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

        if self._no_think_var.get():
            cmd.extend(["--reasoning", "off"])

        port = self._port_var.get()
        cmd.extend(["--host", "0.0.0.0", "--port", port])

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
                kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            self.server_process = subprocess.Popen(cmd, **kw)
            self.running_model = filename
            self._running_gpu_key = gpu_key
            self._running_port = port

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

    def _rescan_models(self):
        self.models = scan_models(self.cfg.get("llm_models_path", ""))
        self._models_header.config(
            text=f"Models ({len(self.models)} GGUF, "
                 f"{sum(m.size_gb for m in self.models):.0f} GB)")
        self._rebuild_model_cards()
        self._log(f"Scan: {len(self.models)} models in {self.cfg['llm_models_path']}", "info")

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

        tk.Label(dlg, text="LLM Models Directory:", font=FONT_BODY_B,
                 bg=t.bg, fg=t.fg).pack(padx=20, anchor="w")
        models_frame = tk.Frame(dlg, bg=t.bg)
        models_frame.pack(fill="x", padx=20, pady=(2, 8))
        models_var = tk.StringVar(value=self.cfg.get("llm_models_path", ""))
        models_entry = ttk.Entry(models_frame, textvariable=models_var, font=FONT_BODY)
        models_entry.pack(side="left", fill="x", expand=True)
        HoverButton(models_frame, t, text="...", color=t.border, width=36, height=24,
                    command=lambda: self._browse_dir(models_var)).pack(side="left", padx=(4, 0))

        tk.Label(dlg, text="llama-server.exe Path:", font=FONT_BODY_B,
                 bg=t.bg, fg=t.fg).pack(padx=20, anchor="w")
        server_frame = tk.Frame(dlg, bg=t.bg)
        server_frame.pack(fill="x", padx=20, pady=(2, 8))
        server_var = tk.StringVar(value=self.cfg.get("llama_server_path", ""))
        server_entry = ttk.Entry(server_frame, textvariable=server_var, font=FONT_BODY)
        server_entry.pack(side="left", fill="x", expand=True)
        HoverButton(server_frame, t, text="...", color=t.border, width=36, height=24,
                    command=lambda: self._browse_file(server_var)).pack(side="left", padx=(4, 0))

        tk.Frame(dlg, height=1, bg=t.border).pack(fill="x", padx=16, pady=8)

        btn_frame = tk.Frame(dlg, bg=t.bg)
        btn_frame.pack(fill="x", padx=20, pady=(0, 12))

        def _save():
            self.cfg["llm_models_path"] = models_var.get()
            self.cfg["llama_server_path"] = server_var.get()
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
        self._center_window(dlg, 560, dlg.winfo_reqheight())

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
