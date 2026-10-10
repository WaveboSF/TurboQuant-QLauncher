#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TurboQuant QLauncher  —  Windows + Linux        (c) WaveboSF 2026
=================================================================
Model Switcher & Server Manager for llama-server with TurboQuant KV-Cache.

Standalone GUI with zero external dependencies. Cross-platform: same file
runs on Windows and Linux (exe_name/lib_name/no_console_kwargs helpers).
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
    python TurboQuant_QLauncher.py          (Windows and Linux — one file)

AP-QL-95 (2026-08-28) removed the second, byte-identical file
`TurboQuant_QLauncher_Linux.py`. It never decided anything: platform behaviour
comes from `IS_WINDOWS = sys.platform == "win32"`, and so do the config and
lock file names. It was a named entry point and a mirroring obligation across
four copies. Both neighbour projects already resolved either name with a
fallback, so nothing broke by dropping it.

(c) 2026 — MIT License
"""

import os
import stat
import sys
import json
import glob
import time
import signal
import socket
import struct
import argparse
import platform
import colorsys
import subprocess
import threading
import queue
import re
import urllib.request
if sys.platform == "win32":
    import msvcrt
else:
    import fcntl
# AP-QL-66: explizit — urllib.request zieht urllib.error heute implizit mit,
# aber _nut_wait_for_server_down faengt HTTPError ab und darf nicht von
# einem Implementierungsdetail von CPython abhaengen.
import urllib.error
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: Platform Helpers (Cross-Platform: Windows + Linux)
# ═══════════════════════════════════════════════════════════════════════════════

IS_WINDOWS = sys.platform == "win32"
IS_LINUX   = sys.platform.startswith("linux")
IS_MACOS   = sys.platform == "darwin"

EXE_SUFFIX = ".exe" if IS_WINDOWS else ""
LIB_SUFFIX = ".dll" if IS_WINDOWS else (".dylib" if IS_MACOS else ".so")

def exe_name(base: str) -> str:
    """'llama-server' -> 'llama-server.exe' on Windows, 'llama-server' on POSIX."""
    return base + EXE_SUFFIX

def no_console_kwargs() -> dict:
    """subprocess kwargs to hide a console window on Windows; no-op elsewhere."""
    if IS_WINDOWS:
        return {"creationflags": 0x08000000}  # CREATE_NO_WINDOW
    return {}

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: Constants
# ═══════════════════════════════════════════════════════════════════════════════

# ── AP-QL-38: EINE Versionszaehlung fuer beide Varianten ────────────────────
# Bis 2026-08-01 fuehrten Windows- und Linux-Fassung getrennte Linien (0.76
# bzw. 0.75). Beide liefen numerisch in denselben Bereich wie die
# historischen "# v0.6x:"-Marken im Quelltext, und dann war nicht mehr
# unterscheidbar, ob "v0.70" den Stand der Datei oder die Windows-Version
# meint, in der eine Aenderung entstand. Ab 0.77 tragen BEIDE Dateien
# denselben Wert und werden gemeinsam hochgezaehlt.
#
# AP-QL-61 (2026-08-03) verschaerfte die Regel von "gleiche Nummer" auf
# "gleicher Inhalt": es gab zwei byte-identische Zwillinge, je Plattform
# einen, und Aenderungen mussten gespiegelt werden.
#
# AP-QL-95 (2026-08-28) hebt das auf: es gibt nur noch DIESE Datei, benutzt
# von Windows UND Linux. Der Dateiname hat nie etwas entschieden — welche
# Plattform-Zweige laufen, entscheidet IS_WINDOWS zur Laufzeit, und daran
# haengen auch die Config- und Lock-Namen. Der Zwilling war ein benannter
# Einstiegspunkt und eine Spiegelpflicht ueber VIER Kopien; aus vier wurden
# zwei (diese und eine Einsatzkopie in einem Nachbarprojekt).
# AP-QL-38 bewacht seitdem, dass der Zwilling NICHT zurueckkommt — taucht er
# wieder auf, ist die Spiegelpflicht still wieder da.
#
# Die vorhandenen "# v0.6x:"-Kommentare bleiben unangetastet — sie sind
# Geschichte und dokumentieren, WANN etwas entstand. Neue Marken nutzen die
# gemeinsame Zaehlung.
#
# build.py liest APP_VERSION weiterhin aus der jeweiligen Datei (einzige
# Versionsquelle je Ziel); Versionsgleichheit UND Byte-Identitaet prueft
# dev/_scratch/AP-QL-38/test_apql38_single_version.py.
APP_VERSION = "1.37"

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
# Data file names: on Linux we use a _Linux suffix so the Windows and Linux
# configs don't step on each other if the launcher folder is shared (e.g. via
# NAS mount or git). Windows build keeps the original unsuffixed filenames.
if IS_WINDOWS:
    CONFIG_FILE  = LAUNCHER_DIR / "TurboQuant_QLauncher.json"
    BENCH_FILE   = LAUNCHER_DIR / "TurboQuant_Benchmark_results.md"
    LOCK_FILE    = LAUNCHER_DIR / "TurboQuant_QLauncher.lock"
    # AP-QL-26 (AP-QL-10d): der API-Schluessel liegt in einer EIGENEN Datei,
    # nicht in der Hauptkonfiguration. Die wird viel freier herumgereicht
    # (Screenshots, Fehlerberichte, Backups), diese hier haelt ein Geheimnis
    # und steht getrennt in .gitignore. Format: {"api_key": "<wert>"}.
    KEY_FILE     = LAUNCHER_DIR / "key.json"
else:
    CONFIG_FILE  = LAUNCHER_DIR / "TurboQuant_QLauncher_Linux.json"
    BENCH_FILE   = LAUNCHER_DIR / "TurboQuant_Benchmark_results_Linux.md"
    LOCK_FILE    = LAUNCHER_DIR / "TurboQuant_QLauncher_Linux.lock"
    KEY_FILE     = LAUNCHER_DIR / "key_Linux.json"

# ── AP-QL-60: Ankerpunkte fuer die Auswahlfenster ───────────────────────────
# Der Launcher liegt IM Workspace, also ist LAUNCHER_DIR.parent dessen Wurzel.
# get_launcher_dir() loest das auch fuer den Nuitka-Onefile-Fall korrekt auf
# (dort zeigt es auf den Ordner der benutzersichtbaren .exe, nicht ins
# Auspack-Verzeichnis) — genau deshalb ist dieser Anker der richtige
# Startpunkt fuer die Dateiauswahl und nicht das Benutzerverzeichnis: eine
# ausgelieferte .exe soll bei ihren eigenen Ordnern anfangen, nicht bei "~".
WORKSPACE_ROOT = LAUNCHER_DIR.parent

# AP-QL-86 (2026-08-21): both platforms' builds now live in ONE folder,
# "TurboQuant/", and the platform is a suffix on the slot folder itself
# (".linux" / ".windows"). It used to be the parent folder — "TurboQuant"
# for Windows, "TurboQuant.linux" for Linux — which marked exactly one
# platform and made the other the unmarked default. That asymmetry is why
# the Windows slot kept carrying its own "_cuda132_sm89_sm120a" suffix
# while the Linux slots stayed short: nothing named the Windows side, so
# the name had to.
#
# Merging is safe because this constant has exactly ONE consumer — the
# file picker's start folder — and slots are never auto-discovered: they
# are explicit config entries, and each platform reads its own config
# (IS_WINDOWS picks the file name). A ".windows" slot therefore never
# becomes a button on Linux.
BUILDS_DIRNAME = "TurboQuant"

# The platform marker every slot folder in BUILDS_DIRNAME carries. It is the
# ONLY thing that tells a runnable build from one for the other platform now
# that both share a folder — so tools and test suites must take it from here
# instead of repeating the literal. AP-QL-83 learned that the hard way with
# the config file name and the llama-server binary name: a suite that spells
# out a platform-specific name goes red on the other platform without
# anything being wrong.
PLATFORM_SLOT_SUFFIX = ".windows" if IS_WINDOWS else ".linux"

# Die GGUF-Bibliothek liegt seit dem 02.08.2026 NEBEN dem Workspace, nicht
# darin — sie wird von mehreren Projekten benutzt. Beide Orte kommen als
# Startpunkt in Frage, der aeussere zuerst, weil er der aktuelle ist.
MODELS_DIRNAME = "llm-models"

# ── AP-QL-26 (Portierung von AP-QL-10a/b): Bind-Adresse ─────────────────────
# Die Schnittstelle, an die llama-server bindet. Loopback haelt das Modell auf
# dieser Maschine; die Wildcard gibt es ans ganze Netz frei. cfg["server_host"]
# haelt den wirksamen Wert und darf auch eine bestimmte Adapter-IP sein
# (z. B. "192.168.1.50") — alles ausser Loopback zaehlt als "LAN freigegeben"
# und treibt die LAN-Checkbox.
#
# ACHTUNG, das war der Zustand vor diesem Paket: die Linux-Variante band
# UNBEDINGT an 0.0.0.0. Der Server war damit ohne Zutun und ohne Schluessel
# aus dem ganzen Netz erreichbar, und es gab keinen Weg, das abzustellen.
SERVER_HOST_LOOPBACK = "127.0.0.1"
SERVER_HOST_ALL = "0.0.0.0"


def server_host_is_lan(cfg: dict) -> bool:
    """Ist der Server ueber Loopback hinaus erreichbar?

    Ein fehlender oder leerer Schluessel bedeutet Loopback (die sichere
    Vorgabe), damit aeltere Konfigurationen privat bleiben, bis der Owner
    die Freigabe ausdruecklich setzt.

    Args:
        cfg: Konfigurations-Dict.

    Returns:
        True, wenn die Bind-Adresse etwas anderes als Loopback ist.
    """
    host = str(cfg.get("server_host") or "").strip()
    return bool(host) and host != SERVER_HOST_LOOPBACK


def api_auth_headers(api_key: str, base: Optional[Dict] = None) -> Dict:
    """Kopfzeilen fuer eigene Anfragen, um Bearer-Auth ergaenzt.

    llama-servers --api-key erwartet die OpenAI-Konvention
    "Authorization: Bearer <key>". Ohne Schluessel bleiben die Kopfzeilen
    unveraendert, der schluessellose Betrieb funktioniert also wie bisher.

    Args:
        api_key: Der konfigurierte Schluessel, ggf. leer.
        base: Vorhandene Kopfzeilen, die ergaenzt werden sollen.

    Returns:
        Neues Dict mit den Kopfzeilen.
    """
    headers = dict(base or {})
    key = str(api_key or "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def mask_command_for_log(cmd: List[str]) -> List[str]:
    """Den Wert hinter --api-key fuers Log durch einen Platzhalter ersetzen.

    Der Launcher schreibt die vollstaendige llama-server-Kommandozeile ins
    Server-Log. Das steht auf dem Schirm und landet in Screenshots — der
    Schluessel darf dort nie auftauchen.

    Args:
        cmd: Die Kommandozeile als Liste.

    Returns:
        Kopie der Liste mit maskiertem Schluessel.
    """
    out = list(cmd)
    for i, part in enumerate(out):
        if part == "--api-key" and i + 1 < len(out):
            out[i + 1] = "***"
    return out


def detect_lan_ip() -> str:
    """Beste Schaetzung der lokalen IPv4-Adresse, sonst "".

    Oeffnet einen UDP-Socket in Richtung einer oeffentlichen Adresse, nur um
    zurueckzulesen, ueber welche lokale Schnittstelle das Betriebssystem
    routen wuerde — es wird kein Paket gesendet und keine Verbindung
    aufgebaut, das funktioniert also auch ohne Internet. Dient allein dazu,
    nach einer LAN-Freigabe eine brauchbare URL auszugeben.

    Returns:
        Die IPv4-Adresse als Zeichenkette, oder "" wenn unbekannt.
    """
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        s.connect(("203.0.113.1", 9))   # TEST-NET-3, garantiert nicht routbar
        ip = s.getsockname()[0]
        return ip if ip and not ip.startswith("127.") else ""
    except OSError:
        return ""
    finally:
        if s is not None:
            try:
                s.close()
            except OSError:
                pass


def load_api_key() -> str:
    """Den API-Schluessel aus KEY_FILE lesen, sonst "".

    Jedes Problem (Datei fehlt, unlesbar, ungueltiges JSON, falscher Typ)
    gilt als "kein Schluessel" — ein schluesselloser Server ist die
    dokumentierte Vorgabe, der Launcher darf daran nie scheitern.

    Returns:
        Der Schluessel, oder "" wenn keiner gesetzt ist.
    """
    try:
        if not KEY_FILE.exists():
            return ""
        with open(KEY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return str(data.get("api_key") or "").strip()
    except Exception:
        pass
    return ""


def save_api_key(key: str) -> None:
    """Den Schluessel in KEY_FILE ablegen, oder die Datei bei leer loeschen.

    Ein geleertes Feld entfernt die Datei, statt einen leeren Geheimnis-
    Behaelter stehen zu lassen. Geloescht wird ausschliesslich diese eine
    launcher-eigene Datei.

    Args:
        key: Der zu speichernde Schluessel, ggf. leer.
    """
    key = str(key or "").strip()
    try:
        if not key:
            if KEY_FILE.exists():
                KEY_FILE.unlink()
            return
        with open(KEY_FILE, "w", encoding="utf-8") as f:
            json.dump({"api_key": key}, f, indent=2, ensure_ascii=False)
        try:
            os.chmod(KEY_FILE, 0o600)   # nur der Eigentuemer darf lesen
        except OSError:
            pass
    except Exception:
        pass

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

KV_CACHE_OPTIONS = {
    "f16 (default)":       {"ctk": None,     "ctv": None},
    # AP-QL-05b: Stufe 1 der asymmetrischen Leiter, die TheTom fd1be210
    # empfiehlt ("sicherer Einstieg"). ctk ist hier "f16" EXPLIZIT und nicht
    # None: None laesst -ctk weg und ergaebe erneut das Default-Preset.
    "f16-K + turbo4-V":    {"ctk": "f16",    "ctv": "turbo4"},
    "q8_0-K + turbo4-V":   {"ctk": "q8_0",   "ctv": "turbo4"},
    "turbo3 / turbo3":     {"ctk": "turbo3",  "ctv": "turbo3"},
    "turbo4 / turbo4":     {"ctk": "turbo4",  "ctv": "turbo4"},
    "q8_0-K + turbo3-V":   {"ctk": "q8_0",   "ctv": "turbo3"},
    "q8_0-K + turbo2-V":   {"ctk": "q8_0",   "ctv": "turbo2"},
    "q8_0 / q8_0":         {"ctk": "q8_0",   "ctv": "q8_0"},
}


# ═══════════════════════════════════════════════════════════════
# AP-QL-16: aus TurboQuant_QLauncher.py (Windows, v0.62) portiert.
# Woertlich uebernommen — NICHT abgetippt und nicht 'angepasst'.
# Der Block ist plattformneutral: er klassifiziert die Serverfamilie
# ueber den Ordnernamen und leitet daraus gueltige KV/LA-Tupel ab.
# Beim naechsten Abgleich wieder 1:1 uebernehmen, damit die beiden
# Dateien nicht erneut auseinanderlaufen.
# ═══════════════════════════════════════════════════════════════

# ── AP-QL-01: Server-family capability gating ──────────────────────────────
# Mainline llama.cpp rejects the turbo* cache types at argument-parse time
# ("Unsupported cache type: turbo4"), so the launcher must never offer or
# pass a turbo KV config to a mainline (or unidentified) build. The family
# is classified from the server exe's parent folder name only — the full
# path is unreliable because the workspace root itself contains "TurboQuant".
# An unknown build is conservatively treated like mainline: no turbo
# capability is ever guessed positively. Exact --version fingerprinting is
# a separate package (AP-QL-06).
SERVER_FAMILY_MAINLINE   = "mainline"
SERVER_FAMILY_TURBOQUANT = "turboquant"
SERVER_FAMILY_UNKNOWN    = "unknown"

# Folder-name substrings that identify a TurboQuant-capable fork build.
_TURBOQUANT_FOLDER_HINTS = ("thetom", "turboquant", "spiritbuun", "madreag")

# The safe KV config every family supports; also the reset target when a
# previously selected config becomes invalid after an engine switch.
SAFE_KV_FALLBACK = "f16 (default)"

# KV presets allowed per family. Mainline/unknown get only the plain
# llama.cpp cache types; TurboQuant forks get the full option list.
_MAINLINE_KV_NAMES = ("f16 (default)", "q8_0 / q8_0")

def detect_server_family(server_exe_path: str) -> str:
    """Classify a llama-server build by its parent folder name.

    Checks only the exe's immediate parent folder (e.g.
    'llama-server_mainline_b10012.windows'), never the full
    path. 'mainline' wins over the turbo hints so a hypothetical
    'turboquant_mainline_test' folder stays conservative.

    Args:
        server_exe_path: Path to llama-server.exe (may be empty).

    Returns:
        One of SERVER_FAMILY_MAINLINE / SERVER_FAMILY_TURBOQUANT /
        SERVER_FAMILY_UNKNOWN.
    """
    folder = slot_folder_name(server_exe_path).lower()
    if not folder:
        return SERVER_FAMILY_UNKNOWN
    if "mainline" in folder:
        return SERVER_FAMILY_MAINLINE
    if any(h in folder for h in _TURBOQUANT_FOLDER_HINTS):
        return SERVER_FAMILY_TURBOQUANT
    return SERVER_FAMILY_UNKNOWN

def allowed_kv_names(family: str) -> tuple:
    """Return the KV preset names valid for the given server family.

    Unknown families are treated like mainline (no turbo types).
    """
    if family == SERVER_FAMILY_TURBOQUANT:
        return tuple(KV_CACHE_OPTIONS.keys())
    return tuple(n for n in KV_CACHE_OPTIONS if n in _MAINLINE_KV_NAMES)

def is_kv_allowed(kv_name: str, family: str) -> bool:
    """Return True if the KV preset may be used with the server family."""
    return kv_name in allowed_kv_names(family)

# ── AP-QL-02: Layer-Adaptive state model and validity matrix ────────────────
# TheTom fd1be210 semantics (src/llama-kv-cache.cpp):
#   env var MISSING → Auto: with turbo2-V (and ≥8 layers) mode 7 activates
#                     automatically; otherwise behaves like off.
#   "0"            → explicitly uniform / LA off.
#   "1"            → first/last 4 layers K+V at q8_0 — only with a Turbo-K.
#   "5"            → first/last 2 V=turbo4, remaining V=turbo2 — only Turbo-V.
#   "7"            → first/last 2 V=q8_0,  remaining V=turbo2 — only Turbo-V.
# "Auto" and "off" are therefore DIFFERENT states: off must set "0"
# explicitly, Auto must remove the variable (even an inherited one).
# LA_MODE_AUTO = -1 keeps IntVar, qnut profiles and integer comparisons
# compatible. Safe fallback for any invalid mode is always 0.
LA_MODE_AUTO = -1

def la_label(la_mode: int) -> str:
    """Return the human-readable label for an LA mode ('auto'/'off'/'5'...)."""
    if la_mode == LA_MODE_AUTO:
        return "auto"
    if la_mode == 0:
        return "off"
    return str(la_mode)

def allowed_la_modes(kv_name: str, family: str) -> tuple:
    """Return the LA modes that have a real, distinct effect for the tuple.

    Mainline/unknown builds ignore TURBO_LAYER_ADAPTIVE → only 0.
    On TurboQuant builds the set derives from the KV preset:
    mode 1 needs a Turbo-K, modes 5/7 need a Turbo-V, and Auto is only
    offered for turbo2-V (the one case where it differs from off).
    Modes 2 and 6 exist upstream but stay hidden (AP-QL-02 scope).
    """
    if family != SERVER_FAMILY_TURBOQUANT:
        return (0,)
    kv = KV_CACHE_OPTIONS.get(kv_name)
    if kv is None:
        return (0,)
    ctk = kv.get("ctk") or ""
    ctv = kv.get("ctv") or ""
    modes = [0]
    if ctv == "turbo2":
        modes.insert(0, LA_MODE_AUTO)
    if "turbo" in ctk:
        modes.append(1)
    if "turbo" in ctv:
        modes.extend((5, 7))
    return tuple(modes)

def is_la_allowed(la_mode: int, kv_name: str, family: str) -> bool:
    """Return True if the LA mode is valid for the KV preset + family."""
    return la_mode in allowed_la_modes(kv_name, family)

def family_la_union(family: str) -> set:
    """Union of LA modes valid for ANY allowed KV preset of the family."""
    modes: set = set()
    for kv_name in allowed_kv_names(family):
        modes.update(allowed_la_modes(kv_name, family))
    return modes

def apply_la_env(env: dict, la_mode: int) -> None:
    """Write the exact TURBO_LAYER_ADAPTIVE semantics into an env dict.

    Auto (-1) → remove the key (even if inherited from the parent env);
    off (0)   → set "0" explicitly (a missing key would mean Auto);
    1/5/7     → set the decimal value as string.
    """
    if la_mode == LA_MODE_AUTO:
        env.pop("TURBO_LAYER_ADAPTIVE", None)
    else:
        env["TURBO_LAYER_ADAPTIVE"] = str(la_mode)

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
# at turbo2-V. Boundary V engages via explicit TURBO_LAYER_ADAPTIVE=7
# (LA=7 in the launcher) OR automatically in the Auto state (variable
# unset, ≥8 layers — fd1be210). Only explicit off (LA=0) forces symmetric
# turbo2-V, which catastrophically degrades PPL, so the Probe scoring
# system will correctly mark LA=0 combinations for this KV as rotten.
KV_COMPRESSION: Dict[str, float] = {
    "f16 (default)":     1.000,   # no compression
    "f16-K + turbo4-V":  0.625,   # mean(1.0, 0.25) — K uncompressed
    "q8_0 / q8_0":       0.500,   # both halved
    "q8_0-K + turbo4-V": 0.375,   # mean(0.5, 0.25)
    "q8_0-K + turbo3-V": 0.350,   # mean(0.5, 0.20)
    "q8_0-K + turbo2-V": 0.328,   # mean(0.5, 0.156) — Boundary V via Auto/LA7
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

# ── AP-QL-88: explicit model roles ────────────────────────────────────────
# Which model is the conversation brain and which one does vision used to be
# implicit: autostart simply replayed cfg["last_model"], i.e. whatever was
# clicked last. That is not a decision, it is a side effect — and it drifted.
# Measured 2026-08-21: the ReachyMini production path (QLauncher --autostart)
# was loading Qwen3.8-27B-UD-Q4_K_XL while dev/tools/start_servers_linux.sh
# loaded Qwen3.8-27B-Q6_K. Two start paths, two different brains, nobody's
# choice. These keys make the choice explicit and reviewable.
#
# A role holds a FILENAME, not a path: the launcher already identifies models
# by filename everywhere (cfg["last_model"], self._model_cards), and a
# filename survives the library moving between machines — an absolute path
# does not. resolve_role_model() turns it back into a ModelInfo against the
# current scan and returns None when the file is gone, so a stale assignment
# degrades to "unassigned" instead of pointing into the void.
ROLE_BRAIN = "role_brain"
ROLE_VISION = "role_vision"
ROLE_KEYS = (ROLE_BRAIN, ROLE_VISION)
ROLE_GLYPHS = {ROLE_BRAIN: "\N{BRAIN}", ROLE_VISION: "\N{EYE}"}
ROLE_TITLES = {ROLE_BRAIN: "Brain", ROLE_VISION: "Vision"}
ROLE_NONE_LABEL = "\u2014 none \u2014"


def find_mmproj(model_path: str) -> Optional[str]:
    """Return the vision projector sitting next to a model, or None.

    Any ``mmproj*.gguf`` in the model's own folder counts; F16 wins when
    several are present, because it is the reference encoder quality and the
    projector is small (~600 MB - 1 GB) next to the weights.

    AP-QL-88 lifted this out of _start_server() so exactly one place knows the
    rule. It is consulted from two directions now — the server command line
    and the Vision role check — and two copies of a filename convention is how
    the pair silently drifts apart.

    Args:
        model_path: Full path to the model GGUF.

    Returns:
        Full path to the projector, or None when the folder holds none or is
        unreadable (a scan failure must never stop a text-only start).
    """
    try:
        model_dir = os.path.dirname(model_path)
        found = [f for f in os.listdir(model_dir)
                 if f.lower().startswith("mmproj") and f.lower().endswith(".gguf")]
    except OSError:
        return None
    if not found:
        return None
    found.sort(key=lambda f: (0 if "f16" in f.lower() else 1, f.lower()))
    return os.path.join(model_dir, found[0])


# AP-QL-102: Aktuelle llama.cpp-Builds warnen fuer Qwen-VL ausdruecklich,
# dass Grounding-Aufgaben mindestens 1024 Bildtokens brauchen. Der bisherige
# Vision-Slot deckelte auf 128; der Brain-Pfad setzte gar keine Untergrenze.
# Ein echter Reachy-Data-URL-Bildlauf mit Qwen3.8-27B, 147456 Kontext,
# q8_0-K/turbo4-V, LA1 und eingebettetem MTP hat 1024 auf 32 GB bestanden.
_QWEN_VL_MIN_IMAGE_TOKENS = 1024


def _vision_image_token_args(model, mmproj_path: Optional[str],
                             configured_max: Optional[int] = None) -> List[str]:
    """Build image-token flags without letting a cap undercut Qwen grounding.

    The minimum is tied to both readable GGUF architecture metadata and an
    attached projector, so text-only Qwen models stay byte-for-byte
    unaffected. ``configured_max`` is used only by the separate Vision slot;
    its historical non-Qwen behavior (including a cap with no projector) is
    preserved. The main Brain path passes ``None`` and receives only Qwen's
    measured minimum.
    """
    args: List[str] = []
    qwen_vl = bool(mmproj_path) and getattr(model, "architecture", "") == "qwen35"
    minimum = _QWEN_VL_MIN_IMAGE_TOKENS if qwen_vl else 0
    if minimum:
        args.extend(["--image-min-tokens", str(minimum)])
    if configured_max is not None:
        maximum = int(configured_max or 0)
        if maximum > 0:
            if minimum:
                maximum = max(maximum, minimum)
            args.extend(["--image-max-tokens", str(maximum)])
    return args


def _is_reasoning_model(filename: str) -> bool:
    """Return True if the GGUF filename looks like a reasoning-capable model."""
    if not filename:
        return False
    low = filename.lower()
    return any(h in low for h in REASONING_MODEL_HINTS)

MAX_SERVER_SLOTS = 6  # up to 6 quick-switch bookmark slots for llama-server.exe paths
MAX_MODELS_PATHS = 3  # up to 3 LLM model directories scanned together

_SLOT_DATE_RE = re.compile(r"(?:^|_)(\d{4}-\d{2}-\d{2})(?=_|$)")


def strip_slot_date(label: str) -> str:
    """Remove one `YYYY-MM-DD` segment from a derived slot label.

    AP-QL-94 (2026-08-28). Slot folder names carry the build's commit date, so
    a directory listing can be read at a glance. Two things forced that: a
    commit hash carries no ordering at all (`8a891f4b` vs `df7f5472` says
    nothing about which is newer), and the folder's own timestamp answers a
    different question — "when was anything inside last touched". On
    2026-08-28 that had the OLDER code showing the YOUNGER date, because it
    happened to be built second, and one slot jumped a full hour merely
    because a note was appended to its manifest.

    The BUTTON must not grow for it. Label width is load-bearing:
    `_refresh_slot_buttons()` sizes every button from the widest label, and
    five long names once produced a 2116 px bar that ran off the window edge,
    leaving the rear buttons unreachable (AP-QL-86). So the folder name serves
    the file manager and this function keeps serving the footer — one string,
    two readers with opposite needs, and the conflict is resolved here instead
    of by shortening the name again.

    Only a well-formed date sitting between separators is removed, so a fork
    or commit that merely contains digits keeps its label intact.
    `slot_folder_name()` still returns the FULL folder name, so benchmark logs
    stay unambiguous and dated.

    Args:
        label: Slot label with prefix and platform suffix already removed.

    Returns:
        The label without its date segment, or unchanged if it carries none.
    """
    return _SLOT_DATE_RE.sub("", label, count=1).strip("_")


def slot_label_from_path(server_exe_path: str) -> str:
    """Derive a short button label from a llama-server(.exe) path.

    Strip the common 'llama-server_' prefix from the parent folder name
    so buttons stay compact. Examples:
      G:\\...\\llama-server_mainline_b10012.windows\\llama-server.exe
        → 'mainline_b10012'   (platform suffix dropped, AP-QL-86)
      /mnt/.../llama-server_mainline_b10548.linux/llama-server
        → 'mainline_b10548'
      G:\\...\\my_custom_build\\llama-server.exe            → 'my_custom_build'

    Linux-specific heuristic: if the immediate parent folder is a
    generic build-output name (bin, build, Release, Debug, out), walk
    one level up so the label actually identifies the fork, e.g.:
      /home/user/llama-cpp-turboquant/build/bin/llama-server
        → 'llama-cpp-turboquant'   (not 'bin')
      /home/user/llama-cpp-madreag/build/bin/llama-server
        → 'llama-cpp-madreag'      (not 'bin')

    Returns empty string for empty/invalid input.
    """
    if not server_exe_path:
        return ""
    try:
        parent = os.path.dirname(server_exe_path)
        folder = os.path.basename(parent)
    except Exception:
        return ""
    if not folder:
        return ""
    # Linux heuristic: walk up past generic build-output folder names.
    _GENERIC = {"bin", "build", "release", "debug", "out", "dist"}
    if folder.lower() in _GENERIC:
        try:
            up = os.path.dirname(parent)
            up_name = os.path.basename(up)
            # Walk up at most twice: bin -> build -> <project>.
            if up_name.lower() in _GENERIC:
                up = os.path.dirname(up)
                up_name = os.path.basename(up)
            if up_name:
                folder = up_name
        except Exception:
            pass
    # AP-QL-86: drop the platform suffix (".linux" / ".windows") that the
    # merged builds folder puts on every slot. For a BUTTON it is pure
    # redundancy — only the running platform's slots reach the config at
    # all — and the label width is load-bearing: five slots with long names
    # once produced a 2116 px bar that ran off the window edge, leaving the
    # rear buttons unreachable. slot_folder_name() still returns the FULL
    # name, so benchmark logs stay unambiguous.
    for _platform_suffix in (PLATFORM_SLOT_SUFFIX, ".linux", ".windows"):
        if folder.lower().endswith(_platform_suffix):
            folder = folder[: -len(_platform_suffix)]
            break
    # Strip the "llama-server_" prefix if present (common build-folder naming pattern)
    low = folder.lower()
    if low.startswith("llama-server_"):
        folder = folder[len("llama-server_"):]
    # AP-QL-94: the date belongs to the folder, not to the button.
    return strip_slot_date(folder)

def dedupe_slot_paths(slots) -> tuple:
    """Clear slots that repeat an earlier slot's binary; keep the first.

    AP-QL-93 (2026-08-28). Two slots pointing at the same llama-server are
    indistinguishable in the footer: slot_label_from_path() derives the
    same label from both, and the active highlight in
    _rebuild_slot_buttons() compares normalized paths — so clicking one
    lights BOTH buttons. That reads as two engines being active at once,
    which the launcher cannot do: there is exactly one
    cfg["llama_server_path"].

    Found when the Paths dialog accepted the ACTIVE engine into a second,
    empty slot. The footer then showed two green buttons carrying the same
    label. The display was truthful; the slot list was not — which is why
    this guards the list rather than the highlight.

    The comparison uses the SAME normalization as that highlight
    (normcase + normpath). Anything looser would leave two slots that
    still light up together, i.e. it would not fix the reported symptom.

    Position is preserved: a later duplicate becomes empty instead of
    shifting the slots behind it, so buttons the user knows keep their
    place in the bar.

    Args:
        slots: Raw slot path list as stored in the config.

    Returns:
        Tuple of (cleaned, dropped). `cleaned` is the list with later
        duplicates replaced by "". `dropped` is a list of
        (slot_number, label, kept_slot_number) tuples for reporting, with
        1-based numbers matching the rows in the Paths dialog.
    """
    cleaned = []
    dropped = []
    first_seen = {}
    for number, raw in enumerate(slots or [], start=1):
        path = (raw or "").strip()
        if not path:
            cleaned.append("")
            continue
        key = os.path.normcase(os.path.normpath(path))
        if key in first_seen:
            dropped.append((number, slot_label_from_path(path) or path,
                            first_seen[key]))
            cleaned.append("")
            continue
        first_seen[key] = number
        cleaned.append(path)
    return cleaned, dropped

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
    # AP-QL-01: safe default — valid on every server family (mainline
    # b10012 aborts on turbo* types). Turbo presets remain one click away
    # on recognized TurboQuant builds.
    #
    # AP-QL-72 (Fund A1): stand hier bis 03.08. auf "q8_0-K + turbo4-V".
    # Das ist die fuer Q4_K_M EMPFOHLENE Config — als Vorgabe fuer eine
    # FRISCHE Installation aber falsch: wer den Launcher zum ersten Mal
    # startet und einen Mainline-Build ausgewaehlt hat, bekam einen
    # Serverstart, den llama.cpp mit unbekanntem Cache-Typ abbricht.
    "kv_cache": SAFE_KV_FALLBACK,
    "port": 8080,
    # AP-QL-26 (AP-QL-10a): Bind-Adresse von llama-server. Vorgabe ist
    # Loopback — privat. Die LAN-Checkbox setzt sie auf SERVER_HOST_ALL.
    # Eine hier von Hand eingetragene Adapter-IP bleibt erhalten und wird
    # der Wildcard vorgezogen (siehe _effective_server_host).
    "server_host": SERVER_HOST_LOOPBACK,
    "ctx_size": "",  # empty = use llama-server default, else passed as -c <ctx>
    "bench_timeout": 90,  # per-run timeout for llama-bench (seconds), see Timeout field
    "no_thinking": False,
    "benchmark": False,
    "bench_all": False,
    "layer_adaptive": 0,
    "window_x": None,
    "window_y": None,
    # AP-QL-29 (v0.72): 1000x800 stammte aus der einspaltigen Modellliste.
    # Mit dem Zwei-Spalten-Raster (AP-QL-24) und der hoeheren Kopfzeile ist
    # das die Groesse, bei der die ganze Bibliothek und ein brauchbarer
    # Ausschnitt des Logs gleichzeitig sichtbar sind. Beim Start weiterhin
    # auf die Bildschirmgroesse gedeckelt, kann also kein kleines Display
    # sprengen.
    "window_w": 1240,
    "window_h": 1150,
    "sash_pos": None,
    # Theme override. None = follow system (auto-detect via
    # detect_system_dark_mode), True = force light, False = force dark.
    # Toggled via the ☀/🌙 checkbox in the header. Takes effect on next
    # launch (widget theming is frozen at __init__ time).
    "light_mode": None,
    # Source dir for "Update Binaries" flow. Empty = feature disabled.
    # Win example: "D:\\src\\llama-cpp-turboquant\\build\\bin\\Release"
    # Linux example: "/home/user/llama-cpp-turboquant/build/bin"
    "update_binaries_source_path": "",
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
    # AP-QL-85 — Multi-Token Prediction. "auto" setzt --spec-type draft-mtp,
    # sobald das gewaehlte Modell NextN-Koepfe traegt (ModelInfo.has_nextn),
    # sonst nichts. "off" schaltet es ganz ab.
    #
    # Bewusst KEIN Knopf in der Oberflaeche: Es gibt nichts abzuwaegen. MTP ist
    # verlustfrei (Spekulation aendert die Ausgabe per Konstruktion nicht) und
    # brachte am 2026-08-21 auf Qwen3.8-27B-Q6_K/RTX 5090 gemessen 103,0 gegen
    # 58,5 tok/s, also 1,76x. Der einzige Grund, es NICHT zu setzen, ist ein
    # Modell ohne NextN — und das entscheidet kein Nutzer, das steht in der
    # Datei. Ein globaler Schalter waere sogar gefaehrlich: auf einem Modell
    # ohne die Koepfe bricht llama-server mit Exit 1 ab.
    #
    # Der Schluessel bleibt als Notausgang, falls MTP einmal stoert — etwa auf
    # einem bildlastigen Pfad, wo der Prompt-Aufpreis (+26,6 ms je Anfrage,
    # gemessen 2026-08-20) den Generierungsgewinn auffrisst.
    "mtp_mode": "auto",
    "last_model": "",
    "last_gpu_key": "",
    # AP-QL-88 — explicit role assignment, see ROLE_BRAIN above. Empty means
    # "not assigned"; autostart then falls back to last_model exactly as
    # before, so an untouched config behaves like 1.16.
    ROLE_BRAIN: "",
    ROLE_VISION: "",
    # AP-QL-90 — der zweite Serverplatz. Eigener Port, eigene Karte, eigener
    # Kontext: er laeuft NEBEN dem Gehirn, nicht statt seiner, und teilt
    # nichts mit dessen Einstellungen. Die KV-/LA-Presets gelten bewusst
    # nicht: sie steuern den Textpfad, und der Bildpfad hat andere Hebel.
    "vision_port": 8081,
    "vision_gpu_key": "",
    "vision_ctx": 8192,
    # 128 bleibt der historische Grob-Lokalisierungs-Default: HD stieg damit
    # am 2026-08-20 von 7 auf 38 FPS, und der damalige 2B-Grobtest traf 30/32
    # statt 28/32 bei 1024. AP-QL-102 aendert diesen Default deshalb
    # NICHT global. Aktuelle Qwen-VL-Builds verlangen fuer Grounding jedoch
    # ausdruecklich mindestens 1024 Bildtokens; qwen35 + echter mmproj wird
    # zur Laufzeit gezielt auf >= 1024 geklemmt. 0 laesst bei Nicht-Qwen den
    # Schalter weg und ueberlaesst dem Server seinen Standard.
    "vision_image_max_tokens": 128,
    "vision_autoload": False,
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
    # AP-QL-19 (portiert nach Linux als AP-QL-22): Zeilenhintergrund des
    # LAUFENDEN Modells. Bewusst eine dezente Toenung und nicht `green` selbst —
    # als Flaeche waere das Autoload-Gruen so kraeftig, dass der Zeilentext
    # darauf umgefaerbt werden muesste. Die Toenung spiegelt das vorhandene
    # `select_bg`-Muster (blaue Toenung) ins Gruene.
    row_running_bg: str
    # AP-QL-35: der Standard-Blauton der Zweitrang-Knoepfe (Engine-Slots,
    # Rescan/Update/Paths/About, Autoload im Aus-Zustand). War bis AP-QL-34
    # die Modulkonstante ACCENT_SOFT und damit in BEIDEN Themes derselbe
    # Wert — im Light-Theme ein Fremdkoerper, weil dort alles andere auf
    # #2563eb sitzt. Jetzt je Theme derselbe Ton wie `accent`, also genau
    # der, der auch die Intel-GPU-Zeile faerbt (Owner-Wunsch aus AP-QL-34).
    accent_soft: str

DARK_THEME = ThemeColors(
    bg="#1a1d23", bg_secondary="#1e2028", bg_header="#12151a",
    fg="#e2e8f0", fg_secondary="#94a3b8", fg_dim="#64748b",
    # AP-QL-29 (v0.68): gruen/rot waren die hellste, saettigste Stufe
    # ihrer Rampe (Tailwind 400) — als Knopffuellung laut, und mit weisser
    # Knopfschrift erreichten sie nur 1,74 / 2,77 Kontrast, also unter den
    # 3,0, die eine grosse fette Beschriftung braucht. Das Abdunkeln loest
    # beides auf einmal: ruhigere Flaeche UND lesbare weisse Schrift
    # (3,37 / 3,76). Sie dienen zugleich als TEXT (die Haken/Kreuze,
    # "● Running", Log-Marken) — das ist die Untergrenze, die sie nicht
    # tiefer gehen laesst: gruen 5,00 und rot 4,49 gegen den Fensterhinter-
    # grund, weiterhin auf/ueber den 4,5 fuer normalen Text.
    accent="#3b82f6", green="#2ea043", yellow="#fbbf24", red="#ef4444",
    border="#2a2d35", button_bg="#2563eb", button_fg="#ffffff",
    entry_bg="#161920", select_bg="#1e3a5f",
    row_running_bg="#1b3326",
    accent_soft="#3b82f6",
)
LIGHT_THEME = ThemeColors(
    bg="#ffffff", bg_secondary="#f1f5f9", bg_header="#e2e8f0",
    fg="#1e293b", fg_secondary="#475569", fg_dim="#64748b",
    accent="#2563eb", green="#15803d", yellow="#a16207", red="#b91c1c",
    border="#cbd5e1", button_bg="#2563eb", button_fg="#ffffff",
    entry_bg="#f1f5f9", select_bg="#dbeafe",
    row_running_bg="#dcfce7",
    accent_soft="#2563eb",
)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: Font Configuration
# ═══════════════════════════════════════════════════════════════════════════════

def _pick_mono_font(probe_root=None) -> str:
    """Pick the best available monospace font for the current platform.

    Windows: Consolas is always present and rendered crisply.
    macOS: Menlo is the system default monospace.
    Linux: probe a preference list in order — modern programming fonts
    first (JetBrains Mono, Fira Code, Cascadia Code), then DejaVu Sans
    Mono as universal fallback.

    Safe at module-import time: if `probe_root` is None and no Tk root
    exists yet, we skip font-list probing and return DejaVu — the
    launcher calls this again later from _configure_theme() once the
    real Tk root exists, which updates the FONT_* globals in place.
    This avoids creating a throwaway Tk() at import (which blows up
    under headless/no-DISPLAY and can conflict with the real root).
    """
    if sys.platform == "win32":
        return "Consolas"
    if sys.platform == "darwin":
        return "Menlo"
    # Linux: only probe if we already have a Tk interpreter to talk to.
    try:
        root = probe_root or tk._default_root
        if root is None:
            return "DejaVu Sans Mono"
        import tkinter.font as _tkfont
        families = set(_tkfont.families(root))
        for cand in ("JetBrains Mono", "Fira Code", "Cascadia Code",
                     "Source Code Pro", "Hack", "Ubuntu Mono",
                     "DejaVu Sans Mono", "Liberation Mono", "Noto Sans Mono"):
            if cand in families:
                return cand
    except Exception:
        pass
    return "DejaVu Sans Mono"

def _pick_braille_font(probe_root=None) -> str:
    """Pick a font with reliable Braille block coverage (U+2800–U+28FF).

    Same safety contract as _pick_mono_font: needs a Tk root or
    returns DejaVu without probing.
    """
    if sys.platform == "win32":
        return "Consolas"
    try:
        root = probe_root or tk._default_root
        if root is None:
            return "DejaVu Sans Mono"
        import tkinter.font as _tkfont
        families = set(_tkfont.families(root))
        for cand in ("DejaVu Sans Mono", "Noto Sans Mono",
                     "Liberation Mono", "Ubuntu Mono"):
            if cand in families:
                return cand
    except Exception:
        pass
    return "DejaVu Sans Mono"

# Initial values — safe defaults, re-resolved once the real Tk root
# exists (see _refresh_fonts() called from _configure_theme).
_MONO = "Consolas" if sys.platform == "win32" else (
        "Menlo" if sys.platform == "darwin" else "DejaVu Sans Mono")
_BRAILLE_FONT = _MONO

# v0.58 — Font-size tuples are derived from these base sizes via
# apply_font_scale(). The module-level FONT_* constants below are what
# every widget references (by re-reading the tuple at build time), so
# re-running apply_font_scale() + a theme rebuild propagates the new
# sizes to every widget on screen. _refresh_fonts() (Linux font
# upgrade) also routes through apply_font_scale so the current user
# zoom is preserved across the post-Tk-root font re-resolution.
_FONT_BASE = {
    "TITLE":    15,
    "SUBTITLE": 12,
    "HEADER":   11,
    "BODY":     11,
    "SMALL":    10,
    "DIM":      10,
    "BRAILLE":  12,
}

# AP-QL-45: Untergrenze 0.75 -> 0.40 (Owner-Wunsch), damit das Fenster
# deutlich kleiner gezogen werden kann. ACHTUNG, das ist keine reine
# Zahlenaenderung: apply_font_scale() deckelt jede Schriftgroesse bei
# 6 px. SMALL (Basis 9) erreicht diesen Boden schon bei 0.67, BODY
# (10) bei 0.60. Unterhalb davon schrumpfen die Schriften NICHT weiter,
# die Knopfbreiten aber schon — sie werden ueber tkfont.measure() der
# tatsaechlich gerenderten Groesse bemessen und bleiben damit korrekt.
# Der Nutzen unterhalb ~0.60 liegt also in den Abstaenden und der
# Zeilenhoehe, nicht mehr in der Schrift selbst.
# AP-QL-56: Obergrenze 2.00 -> 2.50 (Owner-Wunsch). Die Windows-Fassung
# stand immer auf 2.50; beim Aufbau der Linux-Datei war die 2.00 uebernommen
# worden, ohne dass die Abweichung irgendwo festgehalten war. Da die Windows-
# Fassung kuenftig AUS dieser Datei entsteht, waere die 2.00 sonst als
# stillschweigender Rueckschritt dorthin gewandert.
#
# Anders als bei der Untergrenze gibt es nach oben keinen Deckel in
# apply_font_scale() — die Schriften wachsen ueber den ganzen Bereich weiter.
# Was mitwaechst: die Mindestbreite der Modellkarten, die ueber
# tkfont.measure() am laengsten Modellnamen bemessen wird (AP-QL-51). Bei
# 250 % greift dann der Deckel _model_card_max_width bzw. die Spaltenzahl
# faellt auf 1 — beides ist in AP-QL-51/52 abgedeckt und geprueft.
FONT_SCALE_MIN  = 0.40
FONT_SCALE_MAX  = 2.50
FONT_SCALE_STEP = 0.10

# AP-QL-15: Entprellung fuer das automatische Sichern der Fenstergeometrie.
# <Configure> feuert waehrend eines Maus-Resize im Millisekundentakt; erst
# wenn 600 ms lang nichts mehr kommt, gilt das Fenster als "losgelassen" und
# die Geometrie wird EINMAL geschrieben. Ohne das wuerde jedes Ziehen
# hunderte Config-Schreibvorgaenge ausloesen.
GEOM_SAVE_DEBOUNCE_MS = 600

# The active scale is tracked at module level so _refresh_fonts() can
# re-apply it after a font-family upgrade without needing access to the
# launcher instance. Initialised to 1.0; the launcher overrides this
# from cfg during __init__.
_FONT_SCALE_CURRENT = 1.0

FONT_TITLE    = (_MONO, _FONT_BASE["TITLE"], "bold")
FONT_SUBTITLE = (_MONO, _FONT_BASE["SUBTITLE"])
FONT_HEADER   = (_MONO, _FONT_BASE["HEADER"], "bold")
FONT_BODY     = (_MONO, _FONT_BASE["BODY"])
FONT_BODY_B   = (_MONO, _FONT_BASE["BODY"], "bold")
FONT_SMALL    = (_MONO, _FONT_BASE["SMALL"])
FONT_SMALL_B  = (_MONO, _FONT_BASE["SMALL"], "bold")
FONT_DIM      = (_MONO, _FONT_BASE["DIM"])
FONT_BRAILLE  = (_BRAILLE_FONT, _FONT_BASE["BRAILLE"])


def apply_font_scale(scale: float) -> float:
    """Rescale all FONT_* module globals relative to their base sizes.

    Uses the currently-resolved _MONO / _BRAILLE_FONT family names, so
    calling this after _refresh_fonts() has upgraded them to JetBrains
    Mono / Fira Code on Linux will preserve those upgrades while
    applying the user's zoom factor.

    Clamped to [FONT_SCALE_MIN, FONT_SCALE_MAX] so runaway key-repeat
    can't produce illegible or oversized fonts. Integer rounding with
    a floor of 6 guarantees even the smallest fonts remain renderable
    on the lowest zoom level. Returns the actually-applied scale so
    the caller can persist it.
    """
    global FONT_TITLE, FONT_SUBTITLE, FONT_HEADER, FONT_BODY, FONT_BODY_B
    global FONT_SMALL, FONT_SMALL_B, FONT_DIM, FONT_BRAILLE
    global _FONT_SCALE_CURRENT

    scale = max(FONT_SCALE_MIN, min(FONT_SCALE_MAX, float(scale)))
    _FONT_SCALE_CURRENT = scale

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
    FONT_BRAILLE  = (_BRAILLE_FONT, _sz("BRAILLE"))
    return scale


def _refresh_fonts(root) -> None:
    """Re-resolve the FONT_* globals after a real Tk root exists.

    Called once from _configure_theme on first run. On Windows/macOS
    this is effectively a no-op (platform fonts were already correct).
    On Linux it upgrades DejaVu to JetBrains Mono / Fira Code / etc.
    if one of those is installed.

    v0.58 — delegates tuple construction to apply_font_scale() so that
    any user font zoom in effect is preserved after the family upgrade.
    Without this, switching the family here would reset tuple sizes to
    the base values and silently undo Ctrl+Plus steps.
    """
    global _MONO, _BRAILLE_FONT
    new_mono = _pick_mono_font(root)
    new_braille = _pick_braille_font(root)
    if new_mono == _MONO and new_braille == _BRAILLE_FONT:
        return
    _MONO = new_mono
    _BRAILLE_FONT = new_braille
    # Rebuild all FONT_* tuples with the new families at the current
    # user-selected scale. _FONT_SCALE_CURRENT is the source of truth.
    apply_font_scale(_FONT_SCALE_CURRENT)

# AP-QL-34: Owner-Entscheidung 2026-08-01 — der Standard-Blauton ist jetzt
# derselbe wie der von `accent` im Dark-Theme (#3b82f6), der bislang nur
# die Intel-GPU-Zeile ("UHD Graphics") faerbte. Der frueher hier stehende
# hellere Ton #60a5fa erreichte unter weisser Knopfschrift nur 2,54
# Kontrast und lag damit unter den 3,0, die eine grosse fette
# Beschriftung braucht; #3b82f6 kommt auf 3,68. Die Vorliebe des Owners
# und die Lesbarkeit zeigen hier ausnahmsweise in dieselbe Richtung.
#
# BEWUSST weiterhin eine Modulkonstante und kein Theme-Feld: alle
# Aufrufstellen reichen sie als Literal an HoverButton durch. Sie auf
# theme.accent umzustellen (im Light-Theme #2563eb) waere ein eigener
# Umbau — dann muesste jede Aufrufstelle das Theme kennen.
# AP-QL-35: nur noch Rueckfall fuer Aufrufer ohne Theme-Zugriff. Die
# Knoepfe nutzen theme.accent_soft. Wert = Dark-Theme-Ton.
ACCENT_SOFT = "#3b82f6"
ACCENT_TURBO = "#38bdf8"  # Sky blue — TurboQuant accent

# ── AP-QL-41: Quellenfarben der GPU-Zeilen ─────────────────────────────────
# Frueher kodierte die Farbe den HERSTELLER: nvidia gruen, amd rot, intel
# blau. In einer Liste, in der jede Zeile dieselbe Bedeutung hat ("passt das
# Modell hier hinein?"), las sich das falsch — das Blau der Intel-Zeile und
# das Rot einer AMD-Zeile wirkten wie Warnungen, obwohl sie nichts weiter
# als die Marke meinten. Zumal Rot in derselben Zeile schon eine echte
# Bedeutung hat: das ✗ hinter der Groessenangabe.
#
# Owner-Entscheidung 2026-08-01: alle Quellen gruen, je Quelle nur GANZ
# leicht im Farbton verschoben. Damit bleiben sie auf einen Blick eine
# Gruppe, sind aber weiterhin auseinanderzuhalten.
#
# Die Verschiebung ist in Grad Farbton, relativ zum `green` des jeweiligen
# Themes — sie folgt also automatisch dem Hell/Dunkel-Umschalter, statt
# feste Hex-Werte zu behaupten, die nur im Dark-Theme passen.
GPU_SOURCE_HUE_SHIFT = {
    "cpu":    0,    # Bezugspunkt — exakt theme.green
    "nvidia": 8,    # eine Spur ins Blaugruene
    "intel":  16,   # etwas weiter, bleibt aber klar gruen
    "amd":   -10,   # eine Spur ins Gelbgruene
}


def shift_hue(hex_color: str, degrees: float) -> str:
    """Den Farbton einer Farbe um `degrees` verschieben, Rest unveraendert.

    Saettigung und Helligkeit bleiben, damit alle Varianten denselben
    Kontrast gegen den Hintergrund behalten wie die Ausgangsfarbe.

    Args:
        hex_color: Ausgangsfarbe als "#rrggbb".
        degrees: Verschiebung in Grad (-360..360).

    Returns:
        Die verschobene Farbe als "#rrggbb", bei Fehlern die Ausgangsfarbe.
    """
    try:
        r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
        h, sat, val = colorsys.rgb_to_hsv(r, g, b)
        h = (h + degrees / 360.0) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, sat, val)
        return f"#{int(round(r * 255)):02x}{int(round(g * 255)):02x}{int(round(b * 255)):02x}"
    except Exception:
        return hex_color

# AP-QL-24 (Portierung von v0.70): Modellkarten werden in so vielen Spalten
# ausgelegt. 2 haelt eine typische Bibliothek ohne Scrollen auf dem Schirm.
# Ein hoeherer Wert aendert nur das Raster — der Tastaturlauf bleibt
# listen-linear (siehe _on_key_down), ein breiteres Raster wuerde "Runter"
# also weniger raeumlich wirken lassen.
MODEL_CARD_COLS = 2

# AP-QL-51: die Spaltenzahl folgt seither der Fensterbreite UND der
# Schriftgroesse; MODEL_CARD_COLS ist nur noch Vorgabe und Rueckfall, wenn
# noch nichts gemessen werden konnte. Owner-Vorgabe 01.08.2026:
#
#   * Mindestbreite — eine Spalte muss den laengsten Modellnamen samt
#     Groessenangabe tragen, ohne ihn abzuschneiden. Gemessen an der
#     TATSAECHLICH gerenderten Schrift, damit der Zoom mitgeht.
#   * Hoechstbreite — hoechstens das Doppelte der Mindestbreite. Breiter
#     gezogene Karten sind nur noch Leerraum mit einem Namen darin.
#   * Passen mehr Spalten in dieses Fenster, wird eine aufgemacht, statt
#     zwei Karten auseinanderzuziehen.
#
# Der Deckel ist kein Owner-Wunsch, sondern eine Sicherung: jede Karte
# traegt eine Zeile je GPU, ab etwa fuenf Spalten wird die Zuordnung
# Modell -> Zeile schwer lesbar. Wer ihn hebt, hebt ihn bewusst.
MODEL_CARD_COLS_MAX = 4
MODEL_CARD_MAX_FACTOR = 2       # Hoechstbreite = Faktor x Mindestbreite
# Innenabstaende einer Karte: 10 px links + 10 px rechts (hdr.pack), dazu
# 12 px zwischen Name und Groesse und 3 px Spaltenabstand.
MODEL_CARD_PAD_PX = 10 + 10 + 12 + 3

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

# ─── Themed tk.Checkbutton helper ────────────────────────────────────────────
# tk.Checkbutton renders very differently across platforms. On Windows the
# native theme paints a crisp checkmark regardless of our colors. On Linux
# (X11) the indicator box is filled with `selectcolor` when checked and left
# empty when unchecked — no checkmark. If `selectcolor` is close to the
# widget background (as with our old bg_secondary on the dark theme), the
# checked/unchecked states look almost identical. This helper fills the
# indicator in the accent blue when checked, giving a clearly visible signal
# on both OSes, and strips the 3D focus ring for a flat look.
def themed_checkbutton(parent, theme, **kw):
    kw.setdefault("bg", theme.bg)
    kw.setdefault("fg", theme.fg)
    kw.setdefault("activebackground", theme.bg)
    kw.setdefault("activeforeground", theme.fg)
    kw.setdefault("selectcolor", theme.accent)  # filled indicator when checked
    kw.setdefault("highlightthickness", 0)      # no ugly Motif focus ring
    kw.setdefault("bd", 0)
    kw.setdefault("relief", "flat")
    kw.setdefault("font", FONT_SMALL)
    return tk.Checkbutton(parent, **kw)


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
                      border_color=_UNSET, corner_glyph=_UNSET, width=None):
        """Update any subset of the button's visual properties.

        border_color and corner_glyph use a sentinel default so that
        omitting them preserves the current value (unlike passing None,
        which explicitly clears the border or glyph).
        An explicit width resizes both the canvas and its drawn button.
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
        if width is not None:
            self._width = width
            self.configure(width=width)
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
                             font=(_MONO, 7, "bold"))

    def _rounded_rect(self, x1, y1, x2, y2, r, **kw):
        self.create_arc(x1, y1, x1+2*r, y1+2*r, start=90, extent=90, style="pieslice", **kw)
        self.create_arc(x2-2*r, y1, x2, y1+2*r, start=0, extent=90, style="pieslice", **kw)
        self.create_arc(x2-2*r, y2-2*r, x2, y2, start=270, extent=90, style="pieslice", **kw)
        self.create_arc(x1, y2-2*r, x1+2*r, y2, start=180, extent=90, style="pieslice", **kw)
        self.create_rectangle(x1+r, y1, x2-r, y2, **kw)
        self.create_rectangle(x1, y1+r, x2, y2-r, **kw)

    def resync_hover(self) -> None:
        """Hover-Zustand mit der ECHTEN Zeigerposition abgleichen (AP-QL-54).

        Beim Fenster-Resize wandert der Knopf unter dem stehenden Zeiger
        weg (oder darunter hindurch); Tk verpasst dabei gelegentlich das
        <Enter>/<Leave>-Paar, und der Knopf bleibt als heller "Block" im
        Hover-Fill stehen, bis irgendein spaeterer Vorgang neu zeichnet
        (Owner-Beobachtung im Live-Test 01.08.). Hier wird nicht geraten,
        sondern gefragt: liegt der Zeiger WIRKLICH ueber diesem Widget?
        Danach wird bedingungslos neu gezeichnet — das heilt auch reine
        Belichtungsreste, falls der Zustand schon stimmte.
        """
        try:
            px, py = self.winfo_pointerxy()
            over = self.winfo_containing(px, py)
        except tk.TclError:
            return
        self._hover = (over is self)
        self._draw()

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
                    # AP-QL-76 (Fund D): gekuerzt wie im lspci-Zweig. Vorher
                    # stand hier der Rohname ("AMD Radeon(TM) Graphics"), der
                    # in der Beschriftungsspalte abgeschnitten wurde.
                    gpus.append(GPUInfo(index=idx, name=_clean_wmi_gpu_name(name),
                                        vendor=vendor, vram_mb=vram))
                    idx += 1
    elif sys.platform == "linux":
        out = _run_cmd(["lspci"])
        for line in out.split("\n"):
            if any(x in line for x in ["VGA", "3D", "Display"]):
                nl = line.lower()
                if any(g.vendor == "nvidia" for g in gpus) and "nvidia" in nl:
                    continue
                raw = line.split(":", 2)[-1].strip()
                cleaned = _clean_lspci_gpu_name(raw)
                if "radeon" in nl or "amd" in nl:
                    gpus.append(GPUInfo(index=idx, name=cleaned, vendor="amd"))
                    idx += 1
                elif "intel" in nl:
                    gpus.append(GPUInfo(index=idx, name=cleaned, vendor="intel"))
                    idx += 1
    return gpus

def short_gpu_name(name: str) -> str:
    """Strip marketing filler from a GPU name for the header line.

    AP-QL-78 (Delta-Audit 03.08., Fund F): woertlich aus der archivierten
    Windows-Linie zurueckgeholt. Ohne sie stand in der Kopfzeile
    "NVIDIA GeForce RTX 5090" — bei zwei Karten rund 16 Zeichen mehr in
    einer Zeile, die sich GPU, RAM, CUDA, Build und Laufzeit teilt.

    v0.65: the header packs GPU, RAM, CUDA, build and runtime into one
    line, so every character counts — "NVIDIA GeForce RTX 5090" says
    nothing "NVIDIA RTX 5090" doesn't. Only the vendor's series word is
    dropped; the model number, which is the part that identifies the
    card, is left untouched. AMD/Intel names pass through unchanged.

    Args:
        name: Raw GPU name as reported by the vendor tool.

    Returns:
        The name without the series filler word, whitespace collapsed.
    """
    out = name.replace("GeForce ", "")
    return " ".join(out.split())


def _clean_wmi_gpu_name(raw: str) -> str:
    """Turn WMI's verbose VideoController Name into something short and
    readable for the model-card label column.

    AP-QL-76 (Delta-Audit 03.08., Fund D): woertlich aus der archivierten
    Windows-Linie zurueckgeholt. Das Linux-Gegenstueck
    _clean_lspci_gpu_name kam beim Zusammenlegen mit, dieses hier ging
    verloren — der WMI-Zweig nahm seitdem den Rohnamen, der in der
    Beschriftungsspalte abgeschnitten wurde.

    WMI Name field examples:
      "NVIDIA GeForce RTX 5090"              -> "GeForce RTX 5090"
      "AMD Radeon RX 7900 XTX"               -> "Radeon RX 7900 XTX"
      "AMD Radeon(TM) Graphics"              -> "Radeon iGPU"       (APU)
      "Intel(R) UHD Graphics 770"  -> "Intel UHD Graphics 770"      (iGPU)
      "Intel(R) Iris(R) Xe Graphics"         -> "Intel Iris Xe"     (iGPU)
      "Intel(R) Arc(TM) A770 Graphics"       -> "Intel Arc A770"    (discrete)

    Zur UHD-Zeile: die Vorlage behauptete dort "Intel UHD 770". Das war
    schon in der archivierten Linie falsch — die Regel streicht "Graphics"
    nur am ZEILENENDE, und hier folgt noch die Modellnummer. Gemessen am
    03.08. beim Rueckport (AP-QL-76) und hier auf den tatsaechlichen Wert
    berichtigt; das VERHALTEN ist unveraendert woertlich uebernommen.
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

def _clean_lspci_gpu_name(raw: str) -> str:
    """Turn lspci's verbose GPU name into something short and readable.

    lspci raw output examples:
      "Advanced Micro Devices, Inc. [AMD/ATI] Raphael (rev c2)"
        -> "Radeon (Raphael)"
      "Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XTX]"
        -> "Radeon RX 7900 XTX"
      "Intel Corporation Raptor Lake-S UHD Graphics (rev 04)"
        -> "Intel UHD Graphics (Raptor Lake-S)"
      "NVIDIA Corporation GA102 [GeForce RTX 3090]"
        -> "GeForce RTX 3090"
    """
    import re
    s = raw
    # Drop "(rev XX)" suffix
    s = re.sub(r"\s*\(rev [^)]+\)\s*$", "", s)
    # Prefer the bracketed product name if present: "... [Radeon RX 7900 XTX]"
    bracketed = re.findall(r"\[([^\]]+)\]", s)
    for b in bracketed:
        bl = b.lower()
        # Skip generic vendor brackets like "[AMD/ATI]"
        if bl in ("amd/ati", "amd", "ati"):
            continue
        if any(k in bl for k in ("radeon", "geforce", "quadro", "rtx", "gtx",
                                  "arc", "iris", "uhd graphics", "hd graphics")):
            return b.strip()
    # Strip known vendor prefix noise
    s = re.sub(r"^Advanced Micro Devices,?\s*Inc\.?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^\[AMD/ATI\]\s*", "", s)
    s = re.sub(r"^NVIDIA Corporation\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^Intel Corporation\s*", "", s, flags=re.IGNORECASE)
    # Remove any remaining "[AMD/ATI]" marker mid-string
    s = re.sub(r"\s*\[AMD/ATI\]\s*", " ", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # If the codename survived alone (e.g. "Raphael", "Renoir", "Cezanne"),
    # tag it as a Radeon iGPU so the user sees what it is.
    low = s.lower()
    AMD_IGPU_CODENAMES = ("raphael", "renoir", "cezanne", "rembrandt",
                          "phoenix", "hawk point", "strix", "granite ridge",
                          "lucienne", "barcelo", "mendocino", "picasso",
                          "van gogh")
    if any(cn in low for cn in AMD_IGPU_CODENAMES):
        return f"Radeon iGPU ({s})"
    INTEL_IGPU_CODENAMES = ("raptor lake", "alder lake", "tiger lake",
                             "rocket lake", "comet lake", "meteor lake",
                             "arrow lake", "lunar lake")
    if any(cn in low for cn in INTEL_IGPU_CODENAMES):
        return f"Intel iGPU ({s})"
    # Fall back to whatever's left, capped at 28 chars (our row label width)
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

# ── AP-QL-23: Linux-Pendant zu REQUIRED_DLLS_CORE ───────────────────────────
# Bewusst genau die fuenf Kernbibliotheken, die REQUIRED_DLLS_CORE spiegeln —
# nachgemessen an allen fuenf Linux-Builds (mainline b10216, thetom 8a891f4b,
# spiritbuun 0eb1e82b, madreag f67e1357, turbo-tan f883e117), sie sind ueberall
# vorhanden. NICHT aufgenommen: die libllama-*-impl.so-Familie (fehlt bei
# madreag, das auf einer aelteren, monolithischen llama.cpp-Basis sitzt) und
# libmtmd.so (Multimodal, konzeptionell optional). Ein zu breiter Pflichtsatz
# haette ein falsches ✗ auf funktionierenden Builds erzeugt.
#
# Die Namen ohne Versionssuffix sind Symlinks auf libX.so.<major>.<minor> —
# os.path.isfile() folgt ihnen, die Pruefung greift also unveraendert.
#
# Bewusst NICHT geprueft: cuBLAS. Unter Windows liegt es als DLL im Paketordner,
# unter Linux kommt die CUDA-Runtime vom System (LD_LIBRARY_PATH / RPATH) und
# gehoert nicht neben das Binary.
REQUIRED_LIBS_CORE = [
    "libggml-cuda.so", "libggml-base.so", "libggml-cpu.so",
    "libggml.so", "libllama.so",
]

# Beschriftung der Laufzeitbibliotheks-Anzeige im Kopfbereich. Unter Linux von
# "DLLs" auf "Libs" — die Anzeige zeigt .so-Dateien, nicht DLLs.
RUNTIME_LIB_LABEL = "DLLs" if IS_WINDOWS else "Libs"

# cuBLAS DLLs: check for 12.x OR 13.x (whichever matches the build)
CUBLAS_VARIANTS = [
    ("cublas64_12.dll", "cublasLt64_12.dll"),    # CUDA 12.x
    ("cublas64_13.dll", "cublasLt64_13.dll"),    # CUDA 13.x
]

def check_required_dlls(server_dir: str) -> List[Dict]:
    """Pruefen, ob die Laufzeitbibliotheken neben llama-server liegen.

    Windows: DLLs aus REQUIRED_DLLS_CORE plus die passende cuBLAS-Variante
    (12.x oder 13.x, automatisch erkannt).

    Linux (AP-QL-23): die .so-Dateien aus REQUIRED_LIBS_CORE. Frueher gab die
    Funktion hier stumpf [] zurueck, weil "CUDA-Libs liegen auf LD_LIBRARY_PATH"
    — das stimmt fuer cuBLAS, aber die ggml-/llama-Bibliotheken liegen sehr wohl
    im Paketordner. Der leere Rueckgabewert liess die Kopfzeile faelschlich
    "server path not set" melden, obwohl der Pfad gesetzt war.

    Args:
        server_dir: Ordner, in dem das llama-server-Binary liegt.

    Returns:
        Liste von {name, found, size_mb}. Leer nur, wenn server_dir leer ist.
    """
    if not IS_WINDOWS:
        results = []
        for lib in REQUIRED_LIBS_CORE:
            path = os.path.join(server_dir, lib)
            found = os.path.isfile(path)      # folgt Symlinks (libX.so -> libX.so.N)
            size_mb = 0
            if found:
                try:
                    size_mb = round(os.path.getsize(path) / (1024 * 1024), 1)
                except OSError:
                    pass
            results.append({"name": lib, "found": found, "size_mb": size_mb})
        return results

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

# AP-QL-85: NextN-/MTP-Koepfe stehen NICHT in den Metadaten, sondern als
# eigene Tensoren (blk.N.nextn.*). Der Tensor-Abschnitt folgt direkt auf die
# Key/Value-Sektion; ein Eintrag ist Name (u64-Laenge + Bytes), n_dims (u32),
# n_dims x u64 Dimension, Typ (u32), Offset (u64). Dieselbe Erkennung benutzt
# dev/tools/nextn_audit.py.
_GGUF_NEXTN_TENSOR_MARKERS = ("nextn", ".mtp")
_GGUF_MAX_TENSOR_COUNT = 100_000
_GGUF_MAX_TENSOR_DIMS = 8

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

# Substrings in the chat template that OPEN a thinking block. Lowercased
# comparison. Only opening markers belong here: a closing "</think>" on its
# own is never evidence, because non-reasoning templates mention it purely to
# strip thinking out of past turns. See _template_opens_thinking().
_GGUF_THINKING_TEMPLATE_MARKERS = (
    "<think>",
    "<|think|>",
    "<|thinking|>",
)

# Jinja control flow and comments. Whatever survives their removal is what the
# template can actually emit.
_JINJA_STATEMENT_RE = re.compile(r"\{%-?.*?-?%\}", re.S)
_JINJA_COMMENT_RE = re.compile(r"\{#-?.*?-?#\}", re.S)


def _template_opens_thinking(template: str) -> bool:
    """Return True if the chat template can emit an opening thinking marker.

    A plain substring test over the whole template is not enough. Several
    non-reasoning models mention the markers only inside control flow that
    strips thinking out of *past* turns. LFM2.5-1.2B-Instruct is the case that
    forced the distinction: its template contains "</think>" twice, both times
    inside `{%- if "</think>" in content -%}` cleanup for previous assistant
    turns, while its generation branch emits a bare `<|im_start|>assistant`.
    MiniCPM-V-4.6 and Qwen3.8-27B, by contrast, emit `{{- '<think>\\n' }}` in
    their add_generation_prompt branch and really do think.

    So: drop Jinja statements and comments, keep only what the template can
    emit, and look for an opening marker in that. Verified 2026-08-21 against
    all 19 GGUFs in the local library (AP-QL-82).

    Args:
        template: Raw `tokenizer.chat_template` value, may be empty.

    Returns:
        True if a thinking block can be opened by this template.
    """
    if not template:
        return False
    visible = _JINJA_COMMENT_RE.sub(" ", template)
    visible = _JINJA_STATEMENT_RE.sub(" ", visible)
    low = visible.lower()
    return any(marker in low for marker in _GGUF_THINKING_TEMPLATE_MARKERS)


# GGUF metadata is a key/value section at the head of the file, but its size is
# not bounded in any useful way: the token vocabulary lives inside it as a
# string array and alone runs to several megabytes. Reading a fixed prefix
# therefore cannot work. Measured 2026-08-21 over the local library,
# "tokenizer.chat_template" sits between 1.69 MiB and 15.03 MiB into the file
# for 18 of 19 models. The walk below seeks past values instead of reading
# them, so file size is irrelevant; these caps only guard a corrupt header.
_GGUF_MAX_METADATA_BYTES = 64 * 1024 * 1024
_GGUF_MAX_STRING_BYTES = 16 * 1024 * 1024
_GGUF_MAX_ARRAY_LEN = 100_000_000
_GGUF_MAX_KEY_BYTES = 1024
_GGUF_MAX_KV_COUNT = 10_000

# GGUF value type ids -> size in bytes. Types 8 (string) and 9 (array) carry
# their own length and are handled separately.
_GGUF_FIXED_SIZES = {
    0: 1, 1: 1,       # uint8, int8
    2: 2, 3: 2,       # uint16, int16
    4: 4, 5: 4,       # uint32, int32
    6: 4,             # float32
    7: 1,             # bool
    10: 8, 11: 8,     # uint64, int64
    12: 8,            # float64
}
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9


def _read_gguf_u64(f) -> int:
    """Read one little-endian u64 from `f`.

    Args:
        f: Binary file object.

    Returns:
        The decoded value.

    Raises:
        ValueError: The file ended before eight bytes were available.
    """
    raw = f.read(8)
    if len(raw) < 8:
        raise ValueError("truncated u64")
    return struct.unpack("<Q", raw)[0]


def _read_gguf_value(f, val_type: int, depth: int = 0) -> Optional[str]:
    """Read one GGUF metadata value, returning it only if it is a string.

    The file position is always advanced past the value. Non-string values are
    skipped with seeks rather than read into memory, which is what makes a
    multi-megabyte token vocabulary nearly free to walk past.

    Args:
        f: Binary file object positioned at the start of the value.
        val_type: GGUF value type id.
        depth: Current array nesting depth; guards against crafted files.

    Returns:
        The decoded string for type 8, otherwise None.

    Raises:
        ValueError: The value is malformed, implausibly large, or of an
            unknown type — anything that makes further walking unreliable.
    """
    if val_type == _GGUF_TYPE_STRING:
        n = _read_gguf_u64(f)
        if n > _GGUF_MAX_STRING_BYTES:
            raise ValueError(f"string value of {n} bytes")
        raw = f.read(n)
        if len(raw) < n:
            raise ValueError("truncated string value")
        return raw.decode("utf-8", errors="replace")

    if val_type == _GGUF_TYPE_ARRAY:
        if depth > 4:
            raise ValueError("array nested too deeply")
        raw = f.read(12)
        if len(raw) < 12:
            raise ValueError("truncated array header")
        inner_type, count = struct.unpack("<IQ", raw)
        if count > _GGUF_MAX_ARRAY_LEN:
            raise ValueError(f"array of {count} elements")
        fixed = _GGUF_FIXED_SIZES.get(inner_type)
        if fixed is not None:
            f.seek(fixed * count, 1)          # numeric array: a single seek
        elif inner_type == _GGUF_TYPE_STRING:
            # Hot loop: this is the token vocabulary, tens of thousands of
            # strings per model, and it is walked for every model in the
            # library on every scan. Binding the methods to locals and
            # inlining the length read is worth it — going through
            # _read_gguf_u64() here measured 1580 ms against 560 ms for a
            # 19-model scan (AP-QL-82, 2026-08-21).
            read, seek, unpack = f.read, f.seek, struct.unpack
            for _ in range(count):            # each string carries its length
                raw = read(8)
                if len(raw) < 8:
                    raise ValueError("truncated array string length")
                n = unpack("<Q", raw)[0]
                if n > _GGUF_MAX_STRING_BYTES:
                    raise ValueError(f"array string of {n} bytes")
                seek(n, 1)
        else:
            for _ in range(count):
                _read_gguf_value(f, inner_type, depth + 1)
        return None

    fixed = _GGUF_FIXED_SIZES.get(val_type)
    if fixed is None:
        raise ValueError(f"unknown value type {val_type}")
    f.seek(fixed, 1)
    return None


def _read_gguf_info(path: str, max_bytes: int = _GGUF_MAX_METADATA_BYTES,
                    scan_tensors: bool = False):
    """Read GGUF header metadata and, optionally, look for NextN tensors.

    Only keys in _GGUF_KEYS_OF_INTEREST are kept; every other key is skipped
    with seeks. The walk stops as soon as all keys of interest have been seen,
    so a template stored before the vocabulary costs almost nothing.

    AP-QL-82 (2026-08-21): this used to read a fixed 1 MiB prefix and parse it
    from memory, assuming the metadata section fits there. It does not. The
    vocabulary array sits between the architecture key and the chat template,
    so the walk ran out of buffer mid-vocabulary and returned a dict without
    "tokenizer.chat_template" — for 18 of the 19 GGUFs in the local library.
    Template-based reasoning detection therefore never ran and fell back to the
    filename heuristic without saying so. Measured, not assumed.

    Supports GGUF v2 and v3. Returns None on v1 (deprecated Jun 2023, not seen
    in practice) and on any IO error.

    AP-QL-85: `scan_tensors` continues past the key/value section into the
    tensor table and reports whether the model carries NextN/MTP heads. That
    is one walk for both answers — asking separately would pay the (measured)
    ~620 ms library scan twice. Note the early exit below is skipped when
    tensors are wanted: leaving the key/value walk early would strand the file
    position short of the tensor table.

    Args:
        path: Path to the GGUF file.
        max_bytes: Backstop against a corrupt header claiming an endless
            metadata section. Not a read budget — the walk seeks.
        scan_tensors: Also scan tensor names for NextN/MTP markers.

    Returns:
        Tuple of (metadata dict, has_nextn). The dict holds the string-valued
        keys of interest that were found (possibly empty or partial), or is
        None if the file is not a readable GGUF v2/v3. `has_nextn` is False
        whenever `scan_tensors` is False or nothing was found.
    """
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None, False
            header = f.read(20)
            if len(header) < 20:
                return None, False
            version, n_tensors, kv_count = struct.unpack("<IQQ", header)
            # v1 used u32 counts (not u64) and is effectively extinct.
            # Sanity cap — real GGUFs have at most a few hundred keys.
            if version < 2 or kv_count > _GGUF_MAX_KV_COUNT:
                return None, False

            result: Dict[str, str] = {}
            wanted = set(_GGUF_KEYS_OF_INTEREST)
            for _ in range(kv_count):
                if f.tell() > max_bytes:
                    break
                try:
                    key_len = _read_gguf_u64(f)
                    if key_len > _GGUF_MAX_KEY_BYTES:
                        break
                    key = f.read(key_len).decode("utf-8", errors="replace")
                    raw = f.read(4)
                    if len(raw) < 4:
                        break
                    val_type = struct.unpack("<I", raw)[0]
                    value = _read_gguf_value(f, val_type)
                except (ValueError, struct.error, OSError):
                    # Whatever is in `result` so far stays usable, but the walk
                    # cannot continue reliably past a malformed value.
                    break
                if key in wanted and isinstance(value, str):
                    result[key] = value
                    wanted.discard(key)
                    # Nur aussteigen, wenn niemand die Tensoren braucht — sonst
                    # stuende die Datei vor dem Tensor-Abschnitt still.
                    if not wanted and not scan_tensors:
                        break

            has_nextn = False
            if scan_tensors and wanted != set(_GGUF_KEYS_OF_INTEREST):
                has_nextn = _gguf_scan_nextn(f, n_tensors)
            return result, has_nextn
    except OSError:
        return None, False


def _gguf_scan_nextn(f, n_tensors: int) -> bool:
    """Scan the tensor table for NextN/MTP tensor names.

    The file must be positioned at the start of the tensor table, i.e. right
    after the key/value section. Stops at the first hit.

    Args:
        f: Binary file object positioned at the tensor table.
        n_tensors: Tensor count from the GGUF header.

    Returns:
        True if any tensor name carries a NextN/MTP marker.
    """
    if n_tensors <= 0 or n_tensors > _GGUF_MAX_TENSOR_COUNT:
        return False
    read, seek, unpack = f.read, f.seek, struct.unpack
    for _ in range(n_tensors):
        raw = read(8)
        if len(raw) < 8:
            return False
        name_len = unpack("<Q", raw)[0]
        if name_len > _GGUF_MAX_KEY_BYTES:
            return False
        name = read(name_len).decode("utf-8", errors="replace").lower()
        if any(m in name for m in _GGUF_NEXTN_TENSOR_MARKERS):
            return True
        raw = read(4)
        if len(raw) < 4:
            return False
        n_dims = unpack("<I", raw)[0]
        if n_dims > _GGUF_MAX_TENSOR_DIMS:
            return False
        seek(8 * n_dims + 4 + 8, 1)      # Dimensionen, Typ, Offset
    return False


def _read_gguf_metadata(path: str, max_bytes: int = _GGUF_MAX_METADATA_BYTES):
    """Read GGUF header metadata. Returns dict of selected string keys or None.

    Thin wrapper around _read_gguf_info() for callers that do not care about
    tensors. See there for the walk and why it is not a fixed-prefix read.

    Args:
        path: Path to the GGUF file.
        max_bytes: See _read_gguf_info().

    Returns:
        Dict of the string-valued keys of interest, or None.
    """
    return _read_gguf_info(path, max_bytes)[0]


def _detect_model_traits(path: str, filename: str):
    """Return (is_reasoning, architecture, has_nextn) for a GGUF.

    One file walk for every trait the launcher reads off a model. Asking
    separately would pay the library scan twice — measured 2026-08-21: 620 ms
    for the metadata alone, 650 ms with the tensor scan on top.

    Args:
        path: Path to the GGUF file.
        filename: Its basename, for the legacy filename heuristic.

    Returns:
        Tuple of (is_reasoning, architecture, has_nextn). `has_nextn` says the
        model carries NextN/MTP heads, i.e. `--spec-type draft-mtp` will load;
        on a model without them llama-server aborts with exit 1.
    """
    meta, has_nextn = _read_gguf_info(path, scan_tensors=True)
    return _is_reasoning_from(meta or {}, filename) + (has_nextn,)


def _is_reasoning_from(meta: dict, filename: str):
    """Decide reasoning-capability from already-read metadata.

    Split out of _detect_is_reasoning() by AP-QL-85 so the decision rules live
    in one place while the file walk happens once.

    Args:
        meta: Metadata dict from _read_gguf_info().
        filename: GGUF basename, for the legacy heuristic.

    Returns:
        Tuple of (is_reasoning, architecture).
    """
    arch = (meta.get("general.architecture") or "").strip().lower()
    tmpl = meta.get("tokenizer.chat_template") or ""

    is_reasoning = False
    if arch in _GGUF_REASONING_ARCHITECTURES:
        is_reasoning = True
    elif _template_opens_thinking(tmpl):
        is_reasoning = True

    # Legacy fallback — ONLY when there is nothing better to judge by.
    #
    # AP-QL-87 (2026-08-21): the guard `not tmpl` is new, and it is the whole
    # point. The filename heuristic was written as a fallback for "GGUF parse
    # failed or no thinking template" — but it ran unconditionally, so it also
    # overrode a template that had been read successfully and said NO.
    #
    # That was harmless while the reader was broken (AP-QL-82: the template was
    # missing for 18 of 19 local GGUFs, so the fallback was all there was).
    # Once the reader worked, the heuristic started contradicting real evidence:
    # `Qwen3-VL-8B-Instruct` and `Qwen3VL-2B-Instruct` carry templates that open
    # no thinking block, yet the substring "qwen3" forced is_reasoning=True.
    #
    # A template that was read and opens no thinking block is EVIDENCE, not an
    # absence of evidence. A filename is neither — it is a guess that survives
    # renaming, quant suffixes and fine-tune naming only by luck. So: judge by
    # the template when there is one, guess from the name only when there is not.
    #
    # ⚠ This REMOVES warnings rather than adding them (the `--reasoning off`
    # warning fires on is_reasoning), so the failure mode flipped: before, a
    # non-thinking model got a pointless warning; now, a thinking model whose
    # template hides it would get none. The architecture rule above still
    # catches the two families where thinking is the documented default.
    if not is_reasoning and not tmpl:
        is_reasoning = _is_reasoning_model(filename)

    return is_reasoning, arch


def _detect_is_reasoning(path: str, filename: str):
    """Return (is_reasoning, architecture) for a GGUF at `path`.

    Tries GGUF metadata first (architecture + chat template), falls back
    to the legacy filename substring check if the GGUF is malformed or
    the metadata is inconclusive. This guarantees v0.57 is at worst as
    accurate as v0.56 (filename-only) and typically much more accurate
    (handles renamed files, exotic quants, fine-tunes with different
    names but intact templates).
    """
    return _is_reasoning_from(_read_gguf_metadata(path) or {}, filename)


@dataclass
class ServerInstance:
    """One running llama-server and everything that belongs to it.

    AP-QL-89 (2026-08-21). Until now this state lived in eight loose fields
    on the app — server_process, running_model, _running_gpu_key,
    _running_port, _running_kv, _running_la, _running_host,
    _log_reader_thread — read or written at 55 places. That is one object
    carried by hand, and it is the reason the launcher could only ever manage
    a single server: there was nowhere to put a second one.

    Owner decision 2026-08-21: the launcher should run the Brain and the
    Vision model side by side, each on its own port, as equal roles. This
    dataclass is the box those roles go into.

    `role` is ROLE_BRAIN / ROLE_VISION, or "" for a server the user started
    by clicking Run on a card without any role being involved.
    """
    role: str = ""
    process: Optional[subprocess.Popen] = None
    model: Optional[str] = None
    gpu_key: Optional[str] = None
    port: Optional[int] = None
    kv: Optional[str] = None
    la: Optional[int] = None
    host: Optional[str] = None
    log_thread: Optional[threading.Thread] = None
    # AP-QL-91: opaque capability of the external application that caused
    # this exact model load.  Manual/pre-existing servers keep None.  A
    # RELEASE carrying another token can therefore never stop them.
    owner_token: Optional[str] = None

    def is_alive(self) -> bool:
        """True while the process exists and has not exited."""
        return self.process is not None and self.process.poll() is None


# The primary slot: the server the whole existing UI talks about — the one a
# Run click starts, the one the status line and the Stop button refer to.
# Keeping its name separate from ROLE_BRAIN is deliberate: a role says which
# model was CHOSEN for a job, a slot says where a process is RUNNING. A user
# who clicks Run on some other model occupies the primary slot without
# changing anyone's role.
SLOT_PRIMARY = "primary"
SLOT_VISION = "vision"


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
    # AP-QL-85: traegt das Modell NextN-/MTP-Koepfe? Nur dann laesst sich
    # --spec-type draft-mtp setzen; ohne sie bricht llama-server mit Exit 1 ab
    # ("context type MTP requested but model doesn't contain MTP layers").
    # Deshalb haengt der Schalter am Modell und ist nirgends global setzbar.
    has_nextn: bool = False

# Directory name patterns to skip during recursive scanning. These are
# places where .gguf files may exist but are NOT user-selectable LLMs:
# Ollama's blob store (opaque hash filenames), conda/pip caches,
# virtualenvs, snap containers, build caches, hidden config dirs.
# Match is case-insensitive on the basename of the directory.
_SCAN_EXCLUDE_DIRS = frozenset({
    # AP-QL-101: ausgemusterte Modelle bleiben als wiederherstellbares Archiv
    # neben der aktiven Bibliothek, duerfen im rekursiven Launcher-Scan aber
    # nicht erneut als startbare Modelle erscheinen. Wer `_old` bewusst als
    # eigenen Modellpfad eintraegt, kann den Ordner weiterhin direkt scannen;
    # gepruned werden nur Unterverzeichnisse eines uebergeordneten Scan-Roots.
    "_old",
    # Python / package manager / env dirs
    ".cache", ".conda", "conda-meta", ".local", ".venv", "venv", "venvs",
    "miniconda3", "anaconda3", "__pycache__", "site-packages",
    "pip-cache", ".npm", "node_modules",
    # Ollama stores models under ~/.ollama/models/blobs/ as sha256-...
    # These are real ggufs but have no human-readable name, so they're
    # useless in the launcher UI.
    ".ollama", "ollama",
    # Snap and Flatpak containers
    "snap", ".var", "flatpak",
    # AP-QL-76 (Fund D): Windows-Systemordner. Gingen beim Zusammenlegen
    # verloren, weil die Linux-Linie kanonisch wurde — bei breit gesetztem
    # Modellpfad (z. B. "C:\") lief der rekursive Scan danach durch AppData
    # und Windows. Die Namen sind ADDITIV zu den Linux-Eintraegen; der
    # Vergleich ist ohnehin case-insensitiv auf dem Basisnamen, ein
    # Linux-Verzeichnis "windows" waere ein sehr unwahrscheinlicher
    # Fundort fuer GGUFs.
    "appdata", "$recycle.bin", "system volume information",
    "programdata", "windows",
    # Browser / IDE / misc caches
    "mozilla", ".mozilla", ".config", ".wine",
    # Compile caches produced by unsloth etc.
    "unsloth_compiled_cache",
    # Build dirs
    "build", "dist",
    # System trash
    ".trash", ".local/share/Trash",
})

# Minimum GGUF file size to be considered a "real" model. Anything
# smaller is overwhelmingly vocab-only files (ggml-vocab-*.gguf are
# typically <5 MB), stub test fixtures, or corrupted downloads. Real
# quantized LLMs start around 500 MB even for tiny 1-2B models. We
# pick 50 MB as a conservative lower bound — generous enough for
# edge cases (embedding models, TinyLlama-1B-Q2) but eliminates every
# vocab file and llama.cpp test stub.
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

    Applies three filters so that pointing the launcher at a home
    directory still yields a clean list of actual LLMs:
      1. Skip common junk directories (venvs, .ollama blob store,
         snap/flatpak containers, __pycache__, conda envs, unsloth
         caches, etc.). Full list in _SCAN_EXCLUDE_DIRS.
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
            # Filter subdirectories in-place so os.walk doesn't descend
            # into them. topdown=True (default) lets us mutate _dirs.
            for root, subdirs, files in os.walk(models_dir, topdown=True):
                # Prune junk and hidden dirs. Don't prune the scan root
                # itself (user may have pointed at a hidden dir on purpose).
                pruned = []
                for d in subdirs:
                    low = d.lower()
                    if low in _SCAN_EXCLUDE_DIRS:
                        continue
                    # Hidden dir (".git", ".cache", ...) — skip unless
                    # it's a direct child AND the user explicitly named
                    # it. Cheapest check: skip all dotdirs.
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
    is_reasoning, arch, has_nextn = _detect_model_traits(full_path, filename)
    seen[key] = ModelInfo(
        filename=filename,
        path=full_path,
        size_bytes=size,
        size_gb=round(size / (1024**3), 1),
        has_unsafe_path=("," in full_path),
        is_reasoning=is_reasoning,
        architecture=arch,
        has_nextn=has_nextn,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: Installed Non-GGUF Inventory
# ═══════════════════════════════════════════════════════════════════════════════
# AP-QL-123: installed non-GGUF inventory, never llama-server admission.
@dataclass(frozen=True)
class InstalledModelInfo:
    """A local weight directory, separate from runnable GGUF models."""
    name: str
    path: str
    format_name: str
    kind: str
    runtime: str
    size_bytes: int
    status: str


_NON_GGUF_JSON_LIMIT = 2 * 1024 * 1024
_NON_GGUF_INDEX_FORMATS = {
    "model.safetensors.index.json": "Safetensors",
    "pytorch_model.bin.index.json": "PyTorch",
    "model.bin.index.json": "PyTorch",
}


def _non_gguf_format(filename: str) -> str:
    low = filename.lower()
    if (low.startswith("partial") or low.startswith("part-")
            or ".part." in low or ".tmp." in low
            or "mmproj" in low or "vocab" in low):
        return ""
    if low.endswith(".safetensors"):
        return "Safetensors"
    if low.endswith(".onnx"):
        return "ONNX"
    if low.endswith(".bin") and (low.startswith("pytorch_model")
                                  or low.startswith("model")):
        return "PyTorch"
    return ""


def _non_gguf_regular_size(path: str) -> int:
    """Return a positive regular-file size; symlinked weights are omitted."""
    try:
        info = os.lstat(path)
    except OSError:
        return 0
    return info.st_size if stat.S_ISREG(info.st_mode) and info.st_size > 0 else 0


def _non_gguf_json(path: str) -> tuple[Optional[dict], bool]:
    """Read a small regular JSON metadata file, or report it as unreadable."""
    try:
        before = os.lstat(path)
        if not stat.S_ISREG(before.st_mode) or before.st_size > _NON_GGUF_JSON_LIMIT:
            return None, True
        flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                 | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(os.open(path, flags), "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > _NON_GGUF_JSON_LIMIT:
                return None, True
            raw = stream.read(_NON_GGUF_JSON_LIMIT + 1)
        if len(raw) > _NON_GGUF_JSON_LIMIT:
            return None, True
        value = json.loads(raw.decode("utf-8"))
        return (value, False) if isinstance(value, dict) else (None, True)
    except (OSError, UnicodeError, ValueError, RecursionError):
        return None, True


def _non_gguf_shards_missing(root: str, index: dict) -> Optional[bool]:
    """None means an unsafe or malformed index; True means missing shards."""
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return None
    shards = set()
    for tensor, filename in weight_map.items():
        if (not isinstance(tensor, str) or not isinstance(filename, str)
                or not filename or filename in (".", "..") or "\x00" in filename
                or "/" in filename or "\\" in filename or ":" in filename
                or not _non_gguf_format(filename)):
            return None
        shards.add(filename)
    return any(not _non_gguf_regular_size(os.path.join(root, name)) for name in shards)


def _non_gguf_inventory_entry(root: str, files: List[str]) -> Optional[InstalledModelInfo]:
    formats = set()
    size_bytes = 0
    for filename in files:
        format_name = _non_gguf_format(filename)
        if not format_name:
            continue
        size = _non_gguf_regular_size(os.path.join(root, filename))
        if size:
            formats.add(format_name)
            size_bytes += size

    indexes = []
    for filename, format_name in _NON_GGUF_INDEX_FORMATS.items():
        path = os.path.join(root, filename)
        if os.path.lexists(path):
            indexes.append(path)
            formats.add(format_name)
    if not size_bytes and not indexes:
        return None

    metadata_bad = False
    missing_shards = False
    config = {}
    config_path = os.path.join(root, "config.json")
    if os.path.lexists(config_path):
        config, metadata_bad = _non_gguf_json(config_path)
        if not metadata_bad:
            model_type = config.get("model_type")
            architectures = config.get("architectures")
            if ((model_type is not None and not isinstance(model_type, str))
                    or (architectures is not None and
                        (not isinstance(architectures, list)
                         or not all(isinstance(arch, str) for arch in architectures)))):
                metadata_bad = True

    for index_path in indexes:
        index, unreadable = _non_gguf_json(index_path)
        if unreadable:
            metadata_bad = True
            continue
        missing = _non_gguf_shards_missing(root, index)
        if missing is None:
            metadata_bad = True
        elif missing:
            missing_shards = True

    kind = "Model"
    runtime = "External model runtime"
    if not metadata_bad:
        architectures = config.get("architectures") or []
        if (config.get("model_type") == "nemotron3_5_asr"
                or "Nemotron3_5AsrForRNNT" in architectures):
            kind = "ASR"
            runtime = "Reachy speech recognition (Transformers)"
        elif any(arch.endswith("ForCausalLM") for arch in architectures):
            kind = "LLM"

    status = ("Metadata unreadable" if metadata_bad else
              "Incomplete: missing shards" if missing_shards else "Files present")
    return InstalledModelInfo(
        name=os.path.basename(root), path=root,
        format_name=" / ".join(name for name in ("Safetensors", "PyTorch", "ONNX")
                               if name in formats),
        kind=kind, runtime=runtime, size_bytes=size_bytes, status=status,
    )


def scan_non_gguf_models(
        models_dirs: str | os.PathLike | List[str | os.PathLike] | None,
        recursive: bool = True) -> List[InstalledModelInfo]:
    """Inventory local non-GGUF weight directories without loading weights."""
    if isinstance(models_dirs, (str, os.PathLike)):
        dirs = [models_dirs] if models_dirs else []
    else:
        dirs = models_dirs or []

    seen_dirs = set()
    found = []
    for models_dir in dirs:
        if not models_dir:
            continue
        try:
            scan_root = os.path.realpath(os.fspath(models_dir))
        except (OSError, TypeError, ValueError):
            continue
        if not os.path.isdir(scan_root):
            continue
        if recursive:
            for root, subdirs, files in os.walk(scan_root, topdown=True, followlinks=False):
                canonical = os.path.realpath(root)
                if canonical in seen_dirs:
                    subdirs[:] = []
                    continue
                seen_dirs.add(canonical)
                subdirs[:] = [name for name in subdirs
                              if name.lower() not in _SCAN_EXCLUDE_DIRS
                              and not name.startswith(".")
                              and not os.path.islink(os.path.join(root, name))]
                entry = _non_gguf_inventory_entry(canonical, files)
                if entry:
                    found.append(entry)
        elif scan_root not in seen_dirs:
            seen_dirs.add(scan_root)
            try:
                files = os.listdir(scan_root)
            except OSError:
                continue
            entry = _non_gguf_inventory_entry(scan_root, files)
            if entry:
                found.append(entry)

    return sorted(found, key=lambda item: (item.name.casefold(), item.path))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: Config Management
# ═══════════════════════════════════════════════════════════════════════════════

# ── AP-QL-31 (Portierung v0.75): Pfade nach einem Workspace-Umzug ───────────

def _reanchor_path(raw: str, workspace_root: Path, want_dir: bool) -> str:
    """Einen toten Pfad auf der aktuellen Workspace-Wurzel neu verankern.

    Probiert zuerst das laengste Suffix von ``raw`` und arbeitet sich zum
    blossen Dateinamen vor: je mehr des urspruenglichen Pfades sich unter der
    neuen Wurzel nachbilden laesst, desto geringer die Gefahr, eine
    unbeteiligte Datei gleichen Namens zu erwischen. Ein lebender Pfad wird
    nie angefasst, und eine Reparatur, die sich nicht auf der Platte belegen
    laesst, findet gar nicht statt.

    Args:
        raw: Der gespeicherte Pfad, moeglicherweise ins Leere zeigend.
        workspace_root: Verzeichnis, in dem der Launcher selbst liegt (dessen
            Elternverzeichnis).
        want_dir: True, wenn der Pfad ein Verzeichnis benennen muss, False
            fuer eine Datei.

    Returns:
        Der reparierte Pfad mit Vorwaertsslashes, oder "" wenn der Pfad noch
        gueltig ist oder sich nichts belegen liess.
    """
    raw = (raw or "").strip()
    if not raw:
        return ""
    alive = os.path.isdir(raw) if want_dir else os.path.isfile(raw)
    if alive:
        return ""
    # Laufwerksbuchstaben ("D:") und fuehrende UNC-Leerteile fallen weg — nur
    # die benannten Segmente lassen sich neu verankern. Windows-Pfade aus
    # einer geteilten Config landen genau hier: "D:/workspace/
    # llm-models" wird unter der Linux-Wurzel wiedergefunden.
    parts = [p for p in re.split(r"[\\/]+", raw) if p and not p.endswith(":")]
    for start in range(len(parts)):
        candidate = os.path.join(str(workspace_root), *parts[start:])
        found = os.path.isdir(candidate) if want_dir else os.path.isfile(candidate)
        if found:
            return candidate.replace("\\", "/")
    return ""


def _workspace_search_roots() -> List[Path]:
    """Die Wurzeln, unter denen ein toter Pfad gesucht wird — in dieser Reihenfolge.

    AP-QL-57. Bis hierher gab es genau eine Wurzel: ``LAUNCHER_DIR.parent``,
    den Workspace selbst. Das traegt, solange alles Benutzte *im* Workspace
    liegt — und genau das gilt seit dem 02.08.2026 nicht mehr: die
    GGUF-Bibliothek ist von ``<workspace>/llm-models`` nach
    ``<workspace>/../llm-models`` gezogen, weil sie von mehreren Projekten
    benutzt wird und kein Workspace-Inhalt ist. Ein Pfad eine Ebene OBERHALB
    der Wurzel liess sich unter der alten Regel prinzipiell nicht finden; die
    Selbstheilung lief ins Leere und der Owner musste von Hand nachtragen.

    Die Erweiterung ist bewusst STRIKT ADDITIV: die Workspace-Wurzel wird
    zuerst und vollstaendig durchprobiert (laengstes Suffix zuerst, siehe
    ``_reanchor_path``), erst danach die Ebene darueber. Jeder Pfad, den die
    alte Regel repariert hat, wird damit identisch repariert — die zweite
    Wurzel kommt ausschliesslich fuer Faelle zum Zug, in denen vorher gar
    nichts gefunden wurde. Die Alternative (nach Suffixlaenge ueber beide
    Wurzeln hinweg sortieren) waere praeziser, koennte aber bestehende
    Ergebnisse umdrehen, und darum geht es hier nicht.

    Zwei Wurzeln sind das Ende der Fahnenstange, nicht der Anfang einer
    Kette: je hoeher der Anker, desto groesser die Gefahr, eine unbeteiligte
    Datei gleichen Namens zu erwischen (unter ``/srv/projects`` liegen
    z. B. weitere Projekte mit eigenen QLauncher-Kopien).

    Der Dateisystem-Wurzel wird nie zur Suchwurzel — dort waere "eine Ebene
    hoeher" das ganze Laufwerk. Das ist kein theoretischer Fall: auf Windows
    liegt der Workspace als ``D:\\workspace`` direkt auf dem
    Laufwerk, die zweite Wurzel waere ``D:\\``. Dort bleibt diese Erweiterung
    also folgenlos und die Funktion verhaelt sich exakt wie vorher.

    Returns:
        Ein bis zwei existierende Verzeichnisse, spezifischste Wurzel zuerst.
    """
    root = LAUNCHER_DIR.parent
    roots = [root]
    parent = root.parent
    # parent.parent == parent trifft genau die Dateisystem-/Laufwerkswurzel
    # ("/" bzw. "D:\\") — die faellt damit raus, ohne Sonderfall je Plattform.
    if parent != root and parent.parent != parent and parent.is_dir():
        roots.append(parent)
    return roots


def repair_moved_workspace_paths(cfg: dict) -> List[str]:
    """Config-Pfade reparieren, die ein Workspace-Umzug ins Leere hat laufen lassen.

    Wird der ganze Workspace verschoben — anderes Laufwerk, anderer
    Elternordner, umbenannt, oder von Windows auf Linux —, sind schlagartig
    alle absoluten Pfade der Config ungueltig: die Modellverzeichnisse finden
    null Dateien, die Quick-Switch-Slots zeigen nirgendwohin, und die aktive
    Engine meldet "llama-server fehlt", obwohl der Build direkt danebenliegt.
    Der Launcher liegt *im* Workspace, also ist ``LAUNCHER_DIR.parent`` per
    Definition die neue Wurzel, und ein toter Pfad laesst sich daran
    deterministisch neu verankern.

    AP-QL-57: gesucht wird nicht mehr nur unter dieser einen Wurzel, sondern
    zusaetzlich eine Ebene darueber — fuer Daten, die neben dem Workspace
    liegen statt darin (die GGUF-Bibliothek ist am 02.08.2026 genau dorthin
    gezogen). Die Reihenfolge und die Grenzen dieser Ausweitung stehen in
    ``_workspace_search_roots``.

    Konservativ gebaut: umgeschrieben wird nur, was tot ist UND dessen
    reparierte Form auf der Platte existiert. Ein toter Pfad ist nie eine
    absichtliche Einstellung, das kann also keine Absicht ueberschreiben — und
    ein lebender Pfad bleibt unangetastet, selbst wenn unter der neuen Wurzel
    eine gleichnamige Datei liegt.

    Args:
        cfg: Die geladene Konfiguration; wird an Ort und Stelle repariert.

    Returns:
        Je reparierten Pfad eine lesbare Notiz, leer wenn nichts tot war oder
        sich nichts belegen liess. Der Aufrufer speichert die Config und
        meldet die Notizen.
    """
    roots = _workspace_search_roots()
    notes: List[str] = []

    def _fix(path: str, want_dir: bool) -> str:
        """Denselben Pfad unter jeder Suchwurzel probieren, erste Fundstelle gewinnt.

        Die Reihenfolge kommt aus _workspace_search_roots(): Workspace zuerst,
        dann die Ebene darueber. Der Helfer bleibt einwurzelig — die Politik,
        WELCHE Wurzeln in welcher Reihenfolge gelten, gehoert hierher.
        """
        for root in roots:
            fixed = _reanchor_path(path, root, want_dir)
            if fixed:
                return fixed
        return ""

    models = list(cfg.get("llm_models_paths") or [])
    for i, path in enumerate(models):
        fixed = _fix(path, want_dir=True)
        if fixed:
            notes.append(f"Modellpfad #{i + 1} neu verankert: {path} → {fixed}")
            models[i] = fixed
    if notes:
        cfg["llm_models_paths"] = models

    slots = list(cfg.get("server_slots") or [])
    slots_before = len(notes)
    for i, path in enumerate(slots):
        fixed = _fix(path, want_dir=False)
        if fixed:
            notes.append(f"Quick-Switch-Slot #{i + 1} neu verankert: {path} → {fixed}")
            slots[i] = fixed
    if len(notes) > slots_before:
        cfg["server_slots"] = slots

    active = cfg.get("llama_server_path") or ""
    fixed = _fix(active, want_dir=False)
    if fixed:
        notes.append(f"Aktive Engine neu verankert: {active} → {fixed}")
        cfg["llama_server_path"] = fixed

    return notes


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                cfg.update(saved)
    except Exception:
        pass
    # AP-QL-64 (port of AP-QL-10d, verbatim from the archived Windows line):
    # the API key must not live in the main config. Move a key written by an
    # earlier build into key.json and rewrite the config without it, so the
    # secret does not linger in a file that gets shared.
    legacy_key = str(cfg.pop("api_key", "") or "").strip()
    if legacy_key:
        if not load_api_key():
            save_api_key(legacy_key)
        try:
            if CONFIG_FILE.exists():
                save_config(cfg)
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
    "f16-K + turbo4-V",    # 0.625 — TheToms "sicherer Einstieg"
    "q8_0 / q8_0",         # 0.500
    "q8_0-K + turbo4-V",   # 0.375
    "q8_0-K + turbo3-V",   # 0.350 — typischer Neutral-Kandidat
    "q8_0-K + turbo2-V",   # 0.328 — Boundary V (Auto ODER explizit LA=7)
    "turbo4 / turbo4",     # 0.250
    "turbo3 / turbo3",     # 0.200 — Context-Extrem
)

# AP-QL-16: aus Windows v0.62 portiert (woertlich).
# AP-QL-05: qnut leitet seine LA-Modi nicht mehr aus einer festen Liste ab.
# Die alte NUT_LA_MODES = (0, 1, 5, 7) kannte Auto nicht und wurde kartesisch
# mit den KV-Presets gekreuzt — dabei entstand u. a. LA=1 auf q8_0-K-Presets,
# obwohl Modus 1 einen Turbo-K braucht und dort wirkungslos ist. Maßgeblich
# ist jetzt allein allowed_la_modes(kv_name, family) aus AP-QL-02.
#
# Ebenso ist NUT_KV_ORDER nur noch die ORDNUNG (Quality → Context), nicht die
# Auswahl: welche Presets qnut wirklich misst, entscheidet die Serverfamilie
# über allowed_kv_names(). Auf Mainline wären Turbo-Tupel sonst unter falscher
# Bezeichnung als f16 gemessen worden.

def nut_kv_order(family: str) -> list:
    """KV presets qnut measures for `family`, ordered quality → context."""
    allowed = set(allowed_kv_names(family))
    return [kv for kv in NUT_KV_ORDER if kv in allowed]

def nut_la_extras(kv_name: str, family: str) -> list:
    """Non-default LA modes qnut sweeps for `kv_name` in phase 2.

    LA=0 is already covered by phase 0 (baseline) and phase 1 (KV sweep),
    so phase 2 only iterates the remaining modes that have a real, distinct
    effect for this tuple — including Auto, which differs from off only for
    turbo2-V (there it enables Boundary V).
    """
    return [la for la in allowed_la_modes(kv_name, family) if la != 0]

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


def _nut_http_chat(port: int, prompt: str, timeout: int = 60,
                   api_key: str = "") -> str:
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
        # AP-QL-26 (AP-QL-10c): mit gesetztem Schluessel wuerde der
        # Server unsere eigenen Probes sonst mit 401 abweisen.
        headers=api_auth_headers(api_key,
                                 {"Content-Type": "application/json"}),
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
                      progress_cb=None,
                      api_key: str = "") -> tuple:
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
                                  timeout=probe["timeout"],
                                  api_key=api_key)
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
#   Client -> "SHUTDOWN\n"       Server -> "OK shutting down\n"
#   Client -> "AUTOSTART\n"      Server -> "OK starting\n"  (or ERR ...)
#   Client -> "ACQUIRE <token>\n" Server -> "OK acquiring\n"
#   Client -> "RELEASE <token>\n" Server -> "OK releasing\n"
#   Client -> "LEASE_STATUS <token>\n" -> JSON list of owned slots
#   Client -> "STATUS\n"         Server -> "<JSON status>\n"
#   Client -> anything else  Server -> "ERR unknown command\n"
#
# Not a general-purpose RPC. Each command has a single line reply. The two
# lease commands accept only one tightly validated opaque token. The server
# closes the connection after one
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


# ────────────────────────────────────────────────────────────────────
# AP-QL-105: name the process that holds the port.
#
# The v0.57 check above can only say *that* a port is taken; the dialog
# then listed guesses ("another llama-server", "Ollama, vLLM, …"). On
# 2026-09-16 the guess was wrong in the most expensive direction: the
# occupant of 8080 was our own llama-server, spawned by ReachyMini's
# deploy copy of this launcher, serving the robot's brain. A plain kill
# button would have taken it down mid-session. So the occupant is named
# first, and a free port is offered beside the kill.
#
# stdlib only, both platforms:
#   Linux    /proc/net/tcp{,6} carries the socket inode and the owning
#            uid; /proc/<pid>/fd is scanned for that inode.
#   Windows  netstat -ano gives the PID, tasklist the image name.
#
# A socket owned by another user hides its PID from an unprivileged fd
# scan — `ss -ltnp` does not show it either. The uid stays readable, so
# the dialog can say "belongs to ollama, stopping it needs root"
# instead of offering a button that would fail. Measured: port 11434
# yields uid 997 -> ollama, with no PID visible.
# ────────────────────────────────────────────────────────────────────

#: Every llama-server built into a slot folder carries this in its path.
_SLOT_BINARY_MARKER = "llama-server_"

#: What _port_occupant() may report as "kind".
OCCUPANT_NONE = "none"                    # nothing is listening
OCCUPANT_UNKNOWN = "unknown"              # listening, but not identifiable
OCCUPANT_HIDDEN = "hidden"                # another user's process
OCCUPANT_FOREIGN = "foreign"              # someone else's program
OCCUPANT_FOREIGN_SERVER = "foreign-server"    # a llama-server, not from a slot
OCCUPANT_OWN_SERVER = "own-server"            # our slot binary, no launcher parent
OCCUPANT_LAUNCHER_CHILD = "launcher-child"    # our slot binary, another launcher started it
OCCUPANT_QLAUNCHER = "qlauncher"              # another QLauncher itself


def _proc_cmdline(pid: int) -> str:
    """Return the full command line of a process, or "" if unreadable.

    Args:
        pid: Process id to inspect.

    Returns:
        The command line with arguments separated by single spaces.
    """
    if IS_WINDOWS:
        try:
            out = subprocess.run(
                ["wmic", "process", "where", f"processid={pid}",
                 "get", "commandline", "/value"],
                capture_output=True, text=True, timeout=5,
                **no_console_kwargs()).stdout
        except (OSError, subprocess.SubprocessError):
            return ""
        for line in out.splitlines():
            if line.lower().startswith("commandline="):
                return line.split("=", 1)[1].strip()
        return ""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            raw = fh.read()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()


def _proc_ppid(pid: int) -> Optional[int]:
    """Return the parent pid of a process, or None if unreadable.

    Only implemented on POSIX; the Windows dialog does without the
    parent and classifies on the image name alone.
    """
    if IS_WINDOWS:
        return None
    try:
        with open(f"/proc/{pid}/stat") as fh:
            # The executable name sits in parentheses and may itself
            # contain spaces, so split after the last ')'.
            return int(fh.read().rsplit(")", 1)[1].split()[1])
    except (OSError, IndexError, ValueError):
        return None


def _listen_sockets(port: int) -> Dict[str, object]:
    """Return inodes and owning uid of every LISTEN socket on a port.

    Reads /proc/net/tcp and /proc/net/tcp6 directly instead of shelling
    out to `ss`, so the lookup works with nothing installed and returns
    the uid even when the owning process stays invisible.

    Args:
        port: TCP port to look up.

    Returns:
        ``{"inodes": [str, ...], "uid": int | None}``.
    """
    inodes: List[str] = []
    uid: Optional[int] = None
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(path) as fh:
                next(fh, None)
                for line in fh:
                    fields = line.split()
                    if len(fields) < 10 or fields[3] != "0A":
                        continue          # 0A == TCP_LISTEN
                    try:
                        if int(fields[1].split(":")[1], 16) != port:
                            continue
                    except (IndexError, ValueError):
                        continue
                    inodes.append(fields[9])
                    if uid is None:
                        try:
                            uid = int(fields[7])
                        except ValueError:
                            pass
        except OSError:
            continue
    return {"inodes": inodes, "uid": uid}


def _pid_holding_inodes(inodes: List[str]) -> Optional[int]:
    """Return the pid owning one of the given socket inodes.

    Scans /proc/<pid>/fd. Processes of other users deny the listing, so
    a None here means "not visible", never "does not exist".
    """
    wanted = {f"socket:[{inode}]" for inode in inodes}
    if not wanted:
        return None
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        fd_dir = f"/proc/{entry}/fd"
        try:
            for fd in os.listdir(fd_dir):
                try:
                    if os.readlink(f"{fd_dir}/{fd}") in wanted:
                        return int(entry)
                except OSError:
                    continue          # fd vanished mid-scan
        except OSError:
            continue                  # not our process
    return None


def _uid_name(uid: Optional[int]) -> str:
    """Return the account name for a uid, falling back to the number."""
    if uid is None:
        return "unknown"
    try:
        import pwd                    # POSIX only, imported where used
        return pwd.getpwuid(uid).pw_name
    except (ImportError, KeyError):
        return f"uid {uid}"


def _classify_occupant(exe: str, cmdline: str,
                       parent_cmdline: str = "") -> Dict[str, object]:
    """Decide what kind of process holds the port.

    Kept free of I/O so it can be tested against synthetic command
    lines — the AP-QL-93/99 lesson applies to the dialog too: a check
    that anchors on live machine state proves nothing on a quiet
    machine.

    Args:
        exe: Path of the running binary.
        cmdline: Its full command line.
        parent_cmdline: The parent's command line, "" if unknown.

    Returns:
        ``{"kind": str, "detail": str, "stoppable": bool}``.
    """
    name = os.path.basename(exe).lower()
    is_llama_server = name.startswith("llama-server")
    from_slot = _SLOT_BINARY_MARKER in exe

    if "turboquant_qlauncher" in cmdline.lower():
        return {"kind": OCCUPANT_QLAUNCHER,
                "detail": "another QLauncher instance — close it in its own "
                          "window rather than killing it here, or its "
                          "servers lose their owner",
                "stoppable": False}

    if is_llama_server and from_slot:
        if "turboquant_qlauncher" in parent_cmdline.lower():
            return {"kind": OCCUPANT_LAUNCHER_CHILD,
                    "detail": "started by another QLauncher instance",
                    "stoppable": True}
        return {"kind": OCCUPANT_OWN_SERVER,
                "detail": "a llama-server from a TurboQuant slot",
                "stoppable": True}

    if is_llama_server:
        return {"kind": OCCUPANT_FOREIGN_SERVER,
                "detail": "a llama-server, but not from a TurboQuant slot",
                "stoppable": True}

    return {"kind": OCCUPANT_FOREIGN,
            "detail": "not a TurboQuant process",
            "stoppable": True}


def _model_from_cmdline(cmdline: str) -> str:
    """Return the basename of the model a llama-server was started with."""
    argv = cmdline.split()
    for flag in ("-m", "--model"):
        if flag in argv:
            index = argv.index(flag) + 1
            if index < len(argv):
                return os.path.basename(argv[index])
    return ""


def _occupant_windows(port: int) -> Dict[str, object]:
    """Identify a port's owner through netstat and tasklist.

    Untested on Windows as of AP-QL-105 — this machine has no Windows
    bench. Every failure degrades to OCCUPANT_UNKNOWN, so the dialog
    falls back to its old text instead of showing a wrong owner.
    """
    pid: Optional[int] = None
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "TCP"],
                             capture_output=True, text=True, timeout=8,
                             **no_console_kwargs()).stdout
    except (OSError, subprocess.SubprocessError):
        out = ""
    for line in out.splitlines():
        fields = line.split()
        if len(fields) < 5 or fields[3].upper() != "LISTENING":
            continue
        local = fields[1]
        if local.rsplit(":", 1)[-1] != str(port):
            continue
        try:
            pid = int(fields[4])
        except ValueError:
            pid = None
        break
    if pid is None:
        return {"kind": OCCUPANT_UNKNOWN, "pid": None, "name": "", "model": "",
                "detail": "the owning process could not be determined",
                "stoppable": False, "user": ""}

    name = ""
    try:
        listing = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=8,
            **no_console_kwargs()).stdout.strip()
        if listing.startswith('"'):
            name = listing.split('","')[0].strip('"')
    except (OSError, subprocess.SubprocessError):
        name = ""

    cmdline = _proc_cmdline(pid)
    verdict = _classify_occupant(cmdline.split()[0] if cmdline else name,
                                 cmdline or name)
    verdict.update({"pid": pid, "name": name or "unknown",
                    "model": _model_from_cmdline(cmdline), "user": ""})
    return verdict


def _port_occupant(port: int) -> Dict[str, object]:
    """Describe the process listening on a TCP port.

    Args:
        port: The port the user is trying to start a server on.

    Returns:
        A dict with ``kind``, ``detail``, ``stoppable`` and, where they
        could be determined, ``pid``, ``name``, ``model`` and ``user``.
        ``kind`` is OCCUPANT_NONE when the port is in fact free.
    """
    if IS_WINDOWS:
        return _occupant_windows(port)

    sockets = _listen_sockets(port)
    inodes = sockets["inodes"]                       # type: ignore[index]
    uid = sockets["uid"]                             # type: ignore[index]
    if not inodes:
        return {"kind": OCCUPANT_NONE, "pid": None, "name": "", "model": "",
                "detail": "nothing is listening on this port",
                "stoppable": False, "user": ""}

    pid = _pid_holding_inodes(inodes)
    if pid is None:
        owner = _uid_name(uid)
        mine = uid is not None and uid == os.getuid()
        return {"kind": OCCUPANT_HIDDEN, "pid": None, "name": "", "model": "",
                "user": owner,
                "detail": (f"the socket belongs to {owner}; its process is "
                           f"not visible from here"
                           if not mine else
                           "the owning process disappeared during the scan"),
                "stoppable": False}

    cmdline = _proc_cmdline(pid)
    exe = cmdline.split()[0] if cmdline else ""
    parent = _proc_ppid(pid)
    verdict = _classify_occupant(exe, cmdline,
                                 _proc_cmdline(parent) if parent else "")
    verdict.update({
        "pid": pid,
        "name": os.path.basename(exe) or "unknown",
        "model": _model_from_cmdline(cmdline),
        "user": _uid_name(uid),
        "parent": parent,
    })
    return verdict


def _kill_pid(pid: int, timeout: float = 6.0) -> Dict[str, object]:
    """Stop a process this launcher did not start: politely, then hard.

    Mirrors the escalation of _terminate_process(), which can only act
    on a Popen we own. Success is *not* claimed from the signal call —
    the caller re-probes the port, because a process that ignores
    SIGTERM and a port that stays busy look identical from here.

    Args:
        pid: The process to stop.
        timeout: Seconds to wait for the polite signal before forcing.

    Returns:
        ``{"ok": bool, "message": str, "forced": bool}``.
    """
    if _pid_is_gone(pid):
        return {"ok": True, "message": f"PID {pid} was already gone",
                "forced": False}
    try:
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/PID", str(pid), "/T"],
                           capture_output=True, timeout=10,
                           **no_console_kwargs())
        else:
            os.kill(pid, signal.SIGTERM)
    except PermissionError:
        return {"ok": False, "forced": False,
                "message": f"not allowed to stop PID {pid} — it belongs to "
                           f"another user"}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "forced": False,
                "message": f"could not signal PID {pid}: {exc}"}

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _pid_is_gone(pid):
            return {"ok": True, "forced": False,
                    "message": f"PID {pid} stopped"}
        time.sleep(0.2)

    try:
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           capture_output=True, timeout=10,
                           **no_console_kwargs())
        else:
            os.kill(pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "forced": True,
                "message": f"force kill of PID {pid} failed: {exc}"}

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if _pid_is_gone(pid):
            return {"ok": True, "forced": True,
                    "message": f"PID {pid} force-killed"}
        time.sleep(0.2)
    return {"ok": False, "forced": True,
            "message": f"PID {pid} survived both signals"}


def _next_free_port(start: int, limit: int = 32,
                    avoid: Optional[List[int]] = None) -> Optional[int]:
    """Return the first free TCP port at or above ``start``.

    Args:
        start: First candidate, usually the port that is taken plus one.
        limit: How many candidates to try before giving up.
        avoid: Ports to skip although free — our own IPC port, for one.

    Returns:
        A free port, or None if the whole window is occupied.
    """
    blocked = set(avoid or ())
    for candidate in range(max(1, start), min(65535, start + limit) + 1):
        if candidate in blocked:
            continue
        if _port_is_free(candidate):
            return candidate
    return None


def _pid_is_alive(pid: int) -> bool:
    """Return True if a process with the given PID currently exists.

    Cross-platform: uses OpenProcess/GetExitCodeProcess on Windows and
    `os.kill(pid, 0)` on POSIX. Safe on both — neither call kills the
    target process. Returns False on any failure (including permission
    denied — which practically still means "not a process we can talk to").
    """
    if pid <= 0:
        return False
    if IS_WINDOWS:
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

def _pid_is_gone(pid: int) -> bool:
    """Return True if a pid no longer runs, counting a zombie as gone.

    AP-QL-105, found by its own suite: on POSIX a process that has been
    killed but not yet reaped by its parent keeps answering
    `os.kill(pid, 0)`. _kill_pid() therefore waited out both timeouts
    and then reported failure for a process that was in fact dead. Only
    a real run caught this — the signal succeeded, the bookkeeping lied.
    """
    if not _pid_is_alive(pid):
        return True
    if IS_WINDOWS:
        return False
    try:
        with open(f"/proc/{pid}/stat") as fh:
            state = fh.read().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return True
    return state == "Z"

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

_MODEL_LEASE_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{16,128}\Z")


def valid_model_lease_token(token: object) -> bool:
    """Whether ``token`` is safe as one opaque local IPC capability."""
    return bool(_MODEL_LEASE_TOKEN_RE.fullmatch(str(token or "")))


def valid_ipc_argument(value: object, allow_spaces: bool = False) -> bool:
    """Whether ``value`` survives the line-based IPC protocol unharmed.

    The protocol is one command per line, so a control character in a field
    is not a cosmetic problem: an embedded newline would end the command and
    let the remainder be read as a second one.  Spaces are only tolerated in
    the field that is parsed as the trailing remainder (the GPU key, which
    is legitimately "GPU 0").

    Args:
        value: The candidate field, converted to ``str`` before checking.
        allow_spaces: Permit inner spaces (for the remainder field only).

    Returns:
        True if the field is non-empty, short enough, and free of control
        characters — and, unless ``allow_spaces``, free of whitespace.
    """
    text = str(value or "")
    if not text or len(text) > 255:
        return False
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        return False
    if not allow_spaces and any(ch.isspace() for ch in text):
        return False
    return text == text.strip()


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

    # ─── AP-QL-89: server slots ───────────────────────────────────────────
    #
    # The eight fields below used to be plain attributes, written and read at
    # 55 places. They are now views onto self._servers[SLOT_PRIMARY], so the
    # launcher can hold more than one server WITHOUT touching a single one of
    # those 55 call sites — and the existing suite is the proof that nothing
    # moved. The second slot (SLOT_VISION) is filled by its own controls and
    # deliberately does not shadow these names: code that says
    # `self.server_process` means the primary server and should keep meaning
    # exactly that.
    #
    # Doing it as properties instead of a rename is the whole point. A rename
    # would have touched every guard, every status update and every test in
    # one commit, and a mistake in any of them would look like a server bug.

    def slot(self, name: str = SLOT_PRIMARY) -> "ServerInstance":
        """Return the slot with this name, creating an empty one on demand.

        Never returns None: callers ask about a slot far more often than one
        is running, and `self.slot().process is None` reads better than a
        None-check on the slot itself.

        The register is read through ``self.__dict__`` rather than as a plain
        attribute, and that is not style. ``tk.Tk.__getattr__`` forwards every
        unknown attribute to the embedded interpreter object, which forwards
        it again — so a missing ``_servers`` does not raise AttributeError
        here, it recurses until the stack is gone. That turns "the property
        was touched a moment too early" into a crash with no usable message.
        Reading the instance dict cannot recurse, and a missing register is
        simply created.

        Found the hard way in AP-QL-89: two suites build a bare stub that owns
        the methods but never ran __init__, and both died with 993 frames of
        tkinter instead of a readable error.
        """
        servers = self.__dict__.get("_servers")
        if servers is None:
            servers = {}
            self.__dict__["_servers"] = servers
        inst = servers.get(name)
        if inst is None:
            inst = ServerInstance(role=name)
            servers[name] = inst
        return inst

    @property
    def server_process(self):
        return self.slot().process

    @server_process.setter
    def server_process(self, value):
        self.slot().process = value

    @property
    def running_model(self):
        return self.slot().model

    @running_model.setter
    def running_model(self, value):
        self.slot().model = value

    @property
    def _running_gpu_key(self):
        return self.slot().gpu_key

    @_running_gpu_key.setter
    def _running_gpu_key(self, value):
        self.slot().gpu_key = value

    @property
    def _running_port(self):
        return self.slot().port

    @_running_port.setter
    def _running_port(self, value):
        self.slot().port = value

    @property
    def _running_kv(self):
        return self.slot().kv

    @_running_kv.setter
    def _running_kv(self, value):
        self.slot().kv = value

    @property
    def _running_la(self):
        return self.slot().la

    @_running_la.setter
    def _running_la(self, value):
        self.slot().la = value

    @property
    def _running_host(self):
        return self.slot().host

    @_running_host.setter
    def _running_host(self, value):
        self.slot().host = value

    @property
    def _log_reader_thread(self):
        return self.slot().log_thread

    @_log_reader_thread.setter
    def _log_reader_thread(self, value):
        self.slot().log_thread = value

    def __init__(self, autostart_override: Optional[bool] = None,
                 autostart_owner_token: Optional[str] = None,
                 autostart_model: Optional[str] = None,
                 autostart_gpu: Optional[str] = None):
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
        autostart_owner_token:
          Optional AP-QL-91 capability.  When this one-shot autostart really
          creates a model process, the slot records this token.  If a server
          was already present it remains unowned and a later matching release
          cannot touch it.
        autostart_model / autostart_gpu:
          AP-QL-92.  The model (and GPU row) this one-shot autostart should
          start, for the cold path where there is no running instance to
          receive an IPC command.  Both are consumed exactly once and
          outrank the assigned Brain role for that single run — see
          _trigger_autoload_if_eligible.
        """
        super().__init__()
        self.title(f"TurboQuant QLauncher v{APP_VERSION}")
        self._first_run = not CONFIG_FILE.exists()
        self.cfg = load_config()
        # AP-QL-31 (v0.75): ein verschobener Workspace laesst schlagartig
        # jeden gespeicherten Pfad ins Leere laufen — null Modelle im Scan,
        # Slots ohne Ziel, "llama-server fehlt" fuer einen Build, der
        # danebenliegt. Neu verankern, BEVOR irgendetwas die Config liest;
        # die Notizen werden geloggt, sobald das Log-Widget existiert.
        self._path_repairs: List[str] = repair_moved_workspace_paths(self.cfg)
        if self._path_repairs:
            save_config(self.cfg)
        self._autostart_owner_token = (
            autostart_owner_token
            if valid_model_lease_token(autostart_owner_token) else None)
        # AP-QL-92: the cold-start twin of the IPC tail.  Kept as plain
        # one-shot fields next to the token because they share its lifetime
        # exactly: _do_initial_scan consumes all three and clears them, so a
        # later rescan (Paths dialog, quick switch) cannot replay the wish.
        self._autostart_model = (
            autostart_model
            if valid_ipc_argument(autostart_model) else None)
        self._autostart_gpu = (
            autostart_gpu
            if valid_ipc_argument(autostart_gpu, allow_spaces=True) else None)
        # A release can race the initial scan of a freshly spawned launcher.
        # Remember it before Tk executes the queued autostart callback, so the
        # late callback cannot resurrect a model after its owner has gone.
        #
        # AP-QL-99 (2026-08-31): entries are NEVER removed, and that is the
        # design, not an oversight. The set is what makes --release honest:
        # exit 0 promises the GPU memory is back, and a start that was still
        # in flight must not be adopted afterwards. Un-revoking a token would
        # reopen exactly that window.
        #
        # The contract that follows is therefore "a token is SINGLE-USE" —
        # mint a fresh one per acquire. Documented in README.md and in the
        # --acquire help, because a consumer read "use the same token for both
        # commands" as licence to re-acquire a released one (ReachyMini
        # briefing, 2026-08-31). Its own code already mints per acquire
        # (`main.py:14072`, secrets.token_hex(16)), so nothing was broken —
        # the promise was just never written down.
        self._revoked_model_leases: set[str] = set()
        # ACQUIRE/RELEASE arrive on IPC threads while process creation runs on
        # Tk's thread.  One lock makes revocation + Popen + immediate slot
        # adoption a single ownership transition.  Pending counts cover the
        # earlier queued-autoload window, so RELEASE cannot report success
        # while a matching start is still capable of materialising.
        self._model_lease_state_lock = threading.RLock()
        self._pending_model_leases: Dict[str, int] = {}
        if self._autostart_owner_token is not None:
            self._pending_model_leases[self._autostart_owner_token] = 1
        self._model_lease_release_retries: Dict[str, int] = {}
        # v0.56 — CLI override from `--autostart` / `--no-autostart`. None
        # means "use the config flag", True/False force a one-shot
        # decision for this session only (never written back to cfg).
        self._autostart_override = autostart_override
        # v0.56 — IPC server state. Populated by _start_ipc_server when
        # (and only when) Autoload is enabled. Stays None otherwise.
        self._ipc_socket: Optional[socket.socket] = None
        self._ipc_port: int = 0
        self._ipc_accept_thread: Optional[threading.Thread] = None
        # Held for the entire IPC lifetime.  The OS lock, unlike a sequence of
        # JSON existence checks, is an actual cross-process election and is
        # released automatically if the launcher crashes.
        self._ipc_claim_file = None
        self._closing = False
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
        # _refresh_fonts() (called from _configure_theme on Linux) also
        # routes through apply_font_scale, so a font-family upgrade
        # after Tk root init preserves this zoom. Clamped + sanitised
        # by apply_font_scale itself, so a garbage value in the config
        # can't crash the build.
        try:
            self.cfg["font_scale"] = apply_font_scale(
                self.cfg.get("font_scale", 1.0))
        except Exception:
            # Defensive: if anything goes sideways, fall back to 1.0 so
            # the launcher still comes up with readable default fonts.
            self.cfg["font_scale"] = apply_font_scale(1.0)

        self.gpus: List[GPUInfo] = []
        self.models: List[ModelInfo] = []
        # Inventory-only entries never enter GGUF selection or server paths.
        self.installed_models: List[InstalledModelInfo] = []
        # AP-QL-89: muss VOR der ersten Zuweisung an server_process stehen —
        # die ist jetzt eine Eigenschaft und greift hier hinein.
        self._servers: Dict[str, ServerInstance] = {}
        self.server_process = None
        self.running_model = None
        # AP-QL-73 (Delta-Audit 03.08., Fund C): idle | loading | running.
        # Woertlich aus der archivierten Windows-Linie zurueckgeholt — beim
        # Zusammenlegen war das Feld ersatzlos entfallen (8 Verwendungen -> 0).
        # Ein Theme-Wechsel oder Schriftzoom baut den Widget-Baum NEU auf und
        # liefert das Statuslabel im Vorgabezustand zurueck; ohne dieses Feld
        # laesst sich hinterher nicht mehr sagen, ob "Loading" oder "Running"
        # gilt. Liegt bewusst hier und nicht in _build_settings_bar: die
        # Methode laeuft bei JEDEM Neuaufbau und wuerde genau den Zustand
        # zuruecksetzen, den wir wiederherstellen wollen.
        self._status_state: str = "idle"
        self._log_reader_thread: Optional[threading.Thread] = None
        self._model_cards: Dict[str, dict] = {}
        # AP-QL-88: je Rolle ein Knopf im Modell-Kopf.
        self._role_buttons: Dict[str, HoverButton] = {}
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
        # AP-QL-50: _mode_score_labels ist entfallen — die Punktzahl steht
        # jetzt im Tooltip, weil die eigene Etage ueber dem Knopf die
        # Settings-Zeile aus der Flucht brachte.
        self._mode_tooltips: dict = {}
        self._mode_base_tips: dict = {}
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
        # 60 px margin reserves room for taskbar / panel + window
        # decorations. Better to undershoot than to overshoot.
        max_w = int((screen_w - 60) / tk_scale)
        max_h = int((screen_h - 100) / tk_scale)

        w = min(self.cfg.get("window_w", 1240), max_w)
        h = min(self.cfg.get("window_h", 1150), max_h)
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

        # AP-QL-15: Geometrie automatisch sichern, sobald der Nutzer das
        # Fenster losgelassen hat. Bisher wurde sie NUR bei sauberem
        # Schliessen, Theme-Wechsel, Font-Zoom und Serverstart geschrieben —
        # nach einem Absturz oder einem Abschuss ueber den Task-Manager war
        # die Groesse weg. (Eine verwaiste Lock-Datei belegt, dass unsaubere
        # Beendigungen real vorkommen.)
        #
        # _geom_ready sperrt die Automatik bis der Startaufbau durch ist:
        # waehrend __init__ und jedem Widget-Rebuild feuert <Configure>
        # laufend, und ohne die Sperre wuerde ein halbfertiges Fenster seine
        # Uebergangsgroesse in die Config schreiben.
        self._geom_ready = False
        # AP-QL-97: server log lines travel through a queue, not through
        # one after() per line. See _drain_log_queue for the measurement
        # that forced this.
        self._log_queue: "queue.Queue" = queue.Queue()
        # AP-QL-98: dieselbe Bauform fuer AUFTRAEGE aus Arbeiter-Faeden.
        # Unter Tcl 9.0.3 wird `after()` aus einem fremden Faden
        # angenommen und nie ausgefuehrt — siehe _drain_main_queue.
        self._main_queue: "queue.Queue" = queue.Queue()
        self._geom_save_after = None
        self._hover_resync_after = None      # AP-QL-54
        self.bind("<Configure>", self._on_configure_geometry)
        self.after_idle(lambda: setattr(self, "_geom_ready", True))
        # AP-QL-97: start the single log drain once the UI exists.
        # after_idle, not here: _log_text is built later in __init__,
        # and the drain writes into it.
        self.after_idle(self._start_log_drain)

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

        # AP-QL-55 (Portierung von AP-QL-20): AltGr-Zeichen in allen
        # Eingabefeldern tippbar machen. Hier in __init__ aus demselben
        # Grund wie die Tastenkuerzel darueber: bind_class haengt an der
        # Widget-KLASSE, ueberlebt also Theme-Wechsel, Font-Zoom-Neubauten
        # und spaeter erzeugte Dialoge. Die Methode gattert sich selbst auf
        # Windows — der Aufruf bleibt darum bedingungslos.
        self._enable_altgr_input()

        # AP-QL-31 (v0.75): die Reparaturen von oben melden. Verzoegert,
        # weil __init__ erst das Log-Widget baut, in das diese Zeilen gehen.
        if self._path_repairs:
            self.after(50, self._log_path_repairs)

        if self._first_run:
            self.after(100, self._first_run_setup)
        else:
            self.after(100, self._do_initial_scan)
        if self.cfg.get("sash_pos") is not None:
            self.after(150, self._restore_sash)

        # v0.56 — Normally the user's persisted Autoload choice owns IPC.
        # AP-QL-91 adds one narrow exception: a leased one-shot start MUST
        # publish its control socket even when Autoload is off. Otherwise the
        # matching RELEASE could never find the fresh launcher and the model
        # would stay in VRAM after its external owner exited. The token does
        # not change the persisted toggle; closing this session removes IPC.
        if (self.cfg.get("autoload", False)
                or self._autostart_owner_token is not None):
            ipc_ready = self._start_ipc_server()
            if self._autostart_owner_token is not None and not ipc_ready:
                # A leased process without a discoverable control plane could
                # never be released. Fail before autoload, while a normal GUI
                # remains usable exactly as before.
                self._revoke_model_lease(self._autostart_owner_token)
                self._log("Leased autostart cancelled — no exclusive IPC "
                          "control plane could be published.", "error")

        # Note: autoload triggering is done at the tail of
        # _do_initial_scan, once self.models is guaranteed to be populated.
        # See _trigger_autoload_if_eligible for the gating logic.

    # ─── Input ────────────────────────────────────────────────────────────

    def _enable_altgr_input(self) -> None:
        """AltGr-Zeichen bis in die Eingabefelder durchlassen (AP-QL-55).

        Windows meldet AltGr als Strg+Alt, und Tk beantwortet
        ``<Control-Key>`` und ``<Alt-Key>`` in den Klassen-Bindings von
        Entry/TEntry mit ``# nothing``. Auf jedem AltGr-Layout verschluckt
        das lautlos ``\\ @ | ~ { } [ ] EUR µ`` — auf deutscher Tastatur ist
        der Backslash AltGr+ß, ein Windows-Pfad liess sich im
        Configure-Paths-Dialog also ueberhaupt nicht von Hand eintippen;
        nur der "..."-Knopf funktionierte.

        Zwei Muster sind noetig, nicht eines:

        * ``<Control-Alt-Key>`` schlaegt die generischen
          ``<Control-Key>`` / ``<Alt-Key>``-Klassen-Bindings (mehr
          Modifier gewinnen) und deckt den ganzen AltGr-Satz ab.
        * ``<Control-Alt-Key-backslash>`` braucht es zusaetzlich, weil Tk
          Strg+Backslash auf das virtuelle Event ``<<SelectNone>>``
          abbildet und ein Virtual-Event-Treffer ein Muster ohne Detail
          schlaegt. Ohne dieses zweite Muster bliebe ausgerechnet der
          Backslash gefressen — das eine Zeichen, das ein Windows-Pfad
          wirklich braucht. Geprueft auf Tk 8.6.13, Layout 0x0407.

        Plattform-Gatter (AP-QL-55): X11 schickt fuer AltGr
        ``ISO_Level3_Shift``, nicht Control — unter Linux ist die Bindung
        also wirkungslos und wuerde bestenfalls totes Gewicht sein.
        Schlimmstenfalls faenge sie echte Strg+Alt-Kuerzel ab. Da der
        Owner den Linux-Stand am 01.08.2026 live abgenommen hat, bleibt
        dessen Verhalten hier bitweise unangetastet: ausserhalb von
        Windows kehrt die Methode sofort um, ohne irgendetwas zu binden.
        """
        if not IS_WINDOWS:
            return

        def _insert_altgr(event) -> Optional[str]:
            """Das Zeichen einfuegen, das Tk fuer den Strg+Alt-Anschlag las."""
            # Reine Strg+Alt-Kuerzel tragen kein Zeichen — die durchlassen,
            # statt sie anderen Zuhoerern wegzuschnappen.
            if not event.char or not event.char.isprintable():
                return None
            widget = event.widget
            try:
                # ttk.Entry haelt seine Editierbarkeit in den State-Flags,
                # das klassische tk.Entry in der -state-Option. Niemals in
                # ein gesperrtes oder nur lesbares Feld schreiben.
                if hasattr(widget, "instate"):
                    if not widget.instate(["!disabled", "!readonly"]):
                        return None
                elif str(widget.cget("state")) != "normal":
                    return None
                if widget.selection_present():
                    widget.delete("sel.first", "sel.last")
                widget.insert("insert", event.char)
            except (tk.TclError, AttributeError):
                return None      # doch kein Entry — aus dem Weg bleiben
            return "break"

        for widget_class in ("TEntry", "Entry"):
            self.bind_class(widget_class, "<Control-Alt-Key>", _insert_altgr)
            self.bind_class(widget_class, "<Control-Alt-Key-backslash>",
                            _insert_altgr)

    # ─── Theme ────────────────────────────────────────────────────────────

    def _configure_theme(self):
        t = self.theme
        self.configure(bg=t.bg)

        # tk_setPalette: the nuclear option. Sets background /
        # foreground / activeBackground / activeForeground /
        # selectBackground / selectForeground / highlightBackground /
        # highlightColor / insertBackground / disabledForeground /
        # troughColor for EVERY Tk widget class at once — including
        # the obscure ones that show up in file dialogs (Spinbox,
        # legacy internal popup Frames, tearoff Menu stubs, etc.)
        # that we'd otherwise have to enumerate by hand.
        #
        # Must run BEFORE any widget is created, or only new widgets
        # pick it up. Here it runs as the first thing on rebuild
        # (re-config runs before _build_header).
        try:
            self.tk.call(
                "tk_setPalette",
                "background",          t.bg,
                "foreground",          t.fg,
                "activeBackground",    t.accent,
                "activeForeground",    "#ffffff",
                "selectBackground",    t.accent,
                "selectForeground",    "#ffffff",
                "selectColor",         t.accent,
                "highlightBackground", t.bg,
                "highlightColor",      t.accent,
                "insertBackground",    t.fg,
                "disabledForeground",  t.fg_dim,
                "troughColor",         t.bg_secondary,
            )
        except Exception:
            pass

        # Upgrade fonts now that a real Tk root exists. On Linux this
        # may swap DejaVu Sans Mono for JetBrains Mono / Fira Code /
        # Cascadia / etc. if installed. Safe to call repeatedly — it
        # only rebuilds the FONT_* globals if the result differs, and
        # widgets constructed after this point will see the new fonts.
        if not getattr(self, "_fonts_refreshed", False):
            _refresh_fonts(self)
            self._fonts_refreshed = True

        # HiDPI auto-scaling. Tk has no built-in DPI awareness, so on a
        # 4K monitor every widget renders at ~50% of its intended size.
        # We detect the screen height once and bump the Tk scaling
        # factor accordingly. Runs only on first theme-configure (not
        # on the live-swap rebuild) so repeated toggles don't compound.
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
        # ttk.Entry styling — used in Settings/Paths dialog. Clam's
        # default renders a near-invisible 1px border on dark themes;
        # we thicken it and tint it in the accent colour when focused.
        s.configure("TEntry",
                    fieldbackground=t.entry_bg, foreground=t.fg,
                    insertcolor=t.fg,
                    bordercolor=t.border, lightcolor=t.border, darkcolor=t.border,
                    borderwidth=1, padding=(4, 3))
        s.map("TEntry",
              bordercolor=[("focus", t.accent)],
              lightcolor=[("focus", t.accent)],
              darkcolor=[("focus", t.accent)])
        # ttk.Checkbutton — the clam default indicator is barely visible
        # against bg_secondary (~3 values apart on dark). Use the border
        # colour for the frame and a solid accent fill when selected.
        s.configure("TCheckbutton",
                    background=t.bg, foreground=t.fg,
                    indicatorbackground=t.entry_bg,
                    indicatorforeground=t.fg,
                    indicatorcolor=t.entry_bg,
                    bordercolor=t.border,
                    lightcolor=t.border, darkcolor=t.border,
                    focusthickness=1,
                    padding=(2, 1),
                    font=FONT_BODY)
        s.map("TCheckbutton",
              background=[("active", t.bg)],
              foreground=[("disabled", t.fg_dim)],
              indicatorcolor=[("selected", t.accent),
                              ("pressed", t.accent),
                              ("!selected", t.entry_bg)],
              bordercolor=[("focus", t.accent)])
        # ttk.Scrollbar — styled so the model-list scrollbar matches the
        # dark theme. Clam honours these vars; on Windows the native
        # theme wins but it still looks fine there.
        s.configure("Vertical.TScrollbar",
                    background=t.bg_secondary,
                    troughcolor=t.bg,
                    bordercolor=t.bg,
                    arrowcolor=t.fg_secondary,
                    lightcolor=t.bg_secondary,
                    darkcolor=t.bg_secondary,
                    gripcount=0, relief="flat")
        s.map("Vertical.TScrollbar",
              background=[("active", t.border),
                          ("pressed", t.accent)],
              arrowcolor=[("active", t.fg),
                          ("disabled", t.fg_dim)])
        s.configure("Horizontal.TScrollbar",
                    background=t.bg_secondary,
                    troughcolor=t.bg,
                    bordercolor=t.bg,
                    arrowcolor=t.fg_secondary,
                    lightcolor=t.bg_secondary,
                    darkcolor=t.bg_secondary,
                    gripcount=0, relief="flat")
        s.map("Horizontal.TScrollbar",
              background=[("active", t.border),
                          ("pressed", t.accent)],
              arrowcolor=[("active", t.fg),
                          ("disabled", t.fg_dim)])
        s.configure("TSeparator", background=t.border)

        # ── ttk.Combobox + Treeview theming ─────────────────────────
        # Tk's filedialog on Linux builds its "directory breadcrumb"
        # row using a ttk.Combobox, and on Tk 8.6+ the file list is a
        # ttk.Treeview. Neither widget reads the plain option database,
        # they only honour ttk.Style() — so we have to style them here
        # even though the launcher itself doesn't use them directly.
        s.configure("TCombobox",
                    fieldbackground=t.entry_bg,
                    background=t.bg_secondary,
                    foreground=t.fg,
                    arrowcolor=t.fg,
                    bordercolor=t.border,
                    lightcolor=t.border, darkcolor=t.border,
                    insertcolor=t.fg,
                    selectbackground=t.accent,
                    selectforeground="#ffffff",
                    padding=(4, 2))
        s.map("TCombobox",
              fieldbackground=[("readonly", t.entry_bg),
                               ("disabled", t.bg_secondary),
                               ("focus",    t.entry_bg)],
              foreground=[("readonly", t.fg),
                          ("disabled", t.fg_dim)],
              background=[("active", t.bg_secondary),
                          ("readonly", t.bg_secondary)],
              bordercolor=[("focus", t.accent)],
              arrowcolor=[("disabled", t.fg_dim),
                          ("active", t.fg)])
        # The Combobox dropdown popup is a Listbox — it reads the
        # option database, NOT ttk.Style. So we set it via option_add
        # with the TCombobox listbox pattern.
        self.option_add("*TCombobox*Listbox.background",       t.bg_secondary)
        self.option_add("*TCombobox*Listbox.foreground",       t.fg)
        self.option_add("*TCombobox*Listbox.selectBackground", t.accent)
        self.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        self.option_add("*TCombobox*Listbox.borderWidth",      "0")
        self.option_add("*TCombobox*Listbox.relief",           "flat")
        self.option_add("*TCombobox*Listbox.font",             FONT_BODY)

        # Treeview = file list inside the filedialog on modern Tk.
        s.configure("Treeview",
                    background=t.bg_secondary,
                    foreground=t.fg,
                    fieldbackground=t.bg_secondary,
                    bordercolor=t.border,
                    lightcolor=t.border, darkcolor=t.border,
                    borderwidth=0,
                    rowheight=22)
        s.map("Treeview",
              background=[("selected", t.accent)],
              foreground=[("selected", "#ffffff")])
        s.configure("Treeview.Heading",
                    background=t.bg_header,
                    foreground=t.fg,
                    relief="flat",
                    borderwidth=0,
                    font=FONT_SMALL_B)
        s.map("Treeview.Heading",
              background=[("active", t.border)])
        # TButton: used by the filedialog's OK/Cancel row.
        s.configure("TButton",
                    background=t.bg_secondary,
                    foreground=t.fg,
                    bordercolor=t.border,
                    lightcolor=t.border, darkcolor=t.border,
                    borderwidth=1, padding=(8, 4), relief="flat")
        s.map("TButton",
              background=[("active", t.accent),
                          ("pressed", t.accent),
                          ("disabled", t.bg_secondary)],
              foreground=[("active", "#ffffff"),
                          ("pressed", "#ffffff"),
                          ("disabled", t.fg_dim)],
              bordercolor=[("focus", t.accent)])
        # Plain tk Menu widgets inherit via the option database.
        # (Note: there are currently no ttk.Combobox / tk.OptionMenu
        # widgets in the launcher — KV/LA/Q-N-C selectors are custom
        # HoverButton grids — so we don't style TCombobox here.)
        self.option_add("*Menu.background", t.bg_secondary)
        self.option_add("*Menu.foreground", t.fg)
        self.option_add("*Menu.selectColor", t.accent)
        self.option_add("*Menu.activeBackground", t.accent)
        self.option_add("*Menu.activeForeground", t.bg)

        # ── Tk filedialog theming ────────────────────────────────────
        # The built-in filedialog (tk.filedialog.askopenfilename /
        # askdirectory) is implemented in Tcl using bare tk.Frame,
        # tk.Listbox, tk.Entry, tk.Button, tk.Label, tk.Scrollbar —
        # NOT ttk widgets. They ignore all ttk.Style() config and
        # default to the Motif look (white background, black 3D
        # borders, unreadable against our app). The only way to
        # theme them is Tk's classic option database: we push bg/fg
        # for every widget class the dialog uses, then the dialog
        # picks those up at construction time.
        #
        # Patterns use "*Class" so they only match widgets that
        # don't set the attribute explicitly. Our own widgets pass
        # bg=/fg= directly and take precedence, so this is safe to
        # apply globally — it only fills in where nothing else was
        # said, which is exactly inside the filedialog.
        fg_on_accent = "#ffffff"
        _opts = [
            # Generic containers
            ("*Frame.background",           t.bg),
            ("*Toplevel.background",        t.bg),
            ("*Label.background",           t.bg),
            ("*Label.foreground",           t.fg),
            # Buttons (the "Open"/"Cancel"/"Save" row)
            ("*Button.background",          t.bg_secondary),
            ("*Button.foreground",          t.fg),
            ("*Button.activeBackground",    t.accent),
            ("*Button.activeForeground",    fg_on_accent),
            ("*Button.highlightBackground", t.bg),
            ("*Button.highlightColor",      t.accent),
            ("*Button.borderWidth",         "1"),
            ("*Button.relief",              "flat"),
            # Entry (the filename / path bar)
            ("*Entry.background",           t.entry_bg),
            ("*Entry.foreground",           t.fg),
            ("*Entry.insertBackground",     t.fg),
            ("*Entry.selectBackground",     t.accent),
            ("*Entry.selectForeground",     fg_on_accent),
            ("*Entry.highlightBackground",  t.bg),
            ("*Entry.highlightColor",       t.accent),
            ("*Entry.borderWidth",          "1"),
            ("*Entry.relief",               "flat"),
            # Listbox (the file list). This is the big ugly one on
            # Motif — white canvas, black text, no theme awareness.
            ("*Listbox.background",         t.bg_secondary),
            ("*Listbox.foreground",         t.fg),
            ("*Listbox.selectBackground",   t.accent),
            ("*Listbox.selectForeground",   fg_on_accent),
            ("*Listbox.highlightBackground", t.bg),
            ("*Listbox.highlightColor",     t.accent),
            ("*Listbox.borderWidth",        "0"),
            ("*Listbox.relief",             "flat"),
            # Scrollbar on the listbox
            ("*Scrollbar.background",       t.bg_secondary),
            ("*Scrollbar.troughColor",      t.bg),
            ("*Scrollbar.activeBackground", t.border),
            ("*Scrollbar.highlightBackground", t.bg),
            ("*Scrollbar.borderWidth",      "0"),
            ("*Scrollbar.relief",           "flat"),
            # The directory dropdown (tk_chooseDirectory uses a
            # Menubutton with an attached Menu)
            ("*Menubutton.background",      t.bg_secondary),
            ("*Menubutton.foreground",      t.fg),
            ("*Menubutton.activeBackground", t.accent),
            ("*Menubutton.activeForeground", fg_on_accent),
            # Checkbuttons and Radiobuttons that appear inside some
            # filedialog variants ("Show hidden files")
            ("*Checkbutton.background",     t.bg),
            ("*Checkbutton.foreground",     t.fg),
            ("*Checkbutton.selectColor",    t.accent),
            ("*Checkbutton.activeBackground", t.bg),
            ("*Checkbutton.activeForeground", t.fg),
            ("*Radiobutton.background",     t.bg),
            ("*Radiobutton.foreground",     t.fg),
            ("*Radiobutton.selectColor",    t.accent),
        ]
        for pattern, value in _opts:
            self.option_add(pattern, value)

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
        # mode — "☾ Dark" while dark, "☀ Light" while light — so the
        # user sees immediate visual feedback after clicking. Backed by
        # self._light_var so the rest of _on_toggle_theme can read it
        # without caring which widget triggered the change.
        # Note: we use ☾ U+263E (BMP) instead of 🌙 U+1F319 (Supplementary
        # Plane). Linux distros without Noto Color Emoji render 🌙 as a
        # tofu box; ☾ is covered by DejaVu Sans Mono and JetBrains Mono.
        self._light_var = tk.BooleanVar(value=not self.is_dark)
        theme_lbl = tk.Label(
            row1,
            text=("\u2600 Light" if not self.is_dark else "\u263E Dark"),
            font=FONT_SMALL,
            bg=t.bg_header, fg=t.fg_secondary,
            cursor="hand2",
            padx=6, pady=2,
        )
        theme_lbl.pack(side="right")
        self._theme_lbl = theme_lbl

        # AP-QL-15: expliziter Layout-Speicher. Die Automatik (entprelltes
        # <Configure>) deckt den Normalfall ab; dieser Klick ist die
        # Rueckfalloption, wenn man sich nicht darauf verlassen will.
        # Gleiche Optik wie der Theme-Umschalter, damit die Kopfzeile
        # einheitlich bleibt.
        geom_lbl = tk.Label(
            row1,
            text="💾 Layout",
            font=FONT_SMALL,
            bg=t.bg_header, fg=t.fg_secondary,
            cursor="hand2",
            padx=6, pady=2,
        )
        geom_lbl.pack(side="right", padx=(0, 8))
        geom_lbl.bind("<Button-1>", self._on_save_geometry_click)
        ToolTip(geom_lbl,
                "Save window size and position now.\n"
                "They are also saved automatically ~0.6 s after you finish "
                "dragging, and on a clean exit — this button is for when you "
                "want it written immediately.", t)
        self._geom_lbl = geom_lbl

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
        # AP-QL-36 (v0.65): GPU/RAM und CUDA/Build/Libs teilen sich EINE
        # Zeile. Zwei Widgets bleiben es trotzdem, kein zusammengefuegtes
        # Label: die CUDA-Haelfte traegt eine eigene Vordergrundfarbe
        # (dll_color — gruen / gedimmt / gelb je nach Laufzeitzustand), und
        # ein einzelnes Label koennte nur EINE Farbe haben.
        info_row = ttk.Frame(hdr, style="H.TFrame")
        info_row.pack(fill="x", padx=16, pady=(0, 6))
        self._gpu_label = ttk.Label(info_row, text=" ", font=FONT_SMALL,
                                     foreground=t.fg_secondary, style="H.TLabel")
        self._gpu_label.pack(side="left")
        # Der verbindende "  •  " steckt im TEXT von _cuda_label (gesetzt an
        # den drei _refresh-Stellen), nicht in einem eigenen Trennerwidget:
        # solange der Scan noch laeuft, haelt _cuda_label den " "-Platzhalter,
        # und ein eigenstaendiger Trenner liesse bis dahin einen verwaisten
        # Punkt in der Kopfzeile stehen.
        self._cuda_label = ttk.Label(info_row, text=" ", font=FONT_SMALL,
                                      foreground=t.fg_dim, style="H.TLabel")
        self._cuda_label.pack(side="left")

        # AP-QL-25 (Portierung von AP-QL-13): Wirt fuer Run / Stop / Status.
        # Diese drei sassen bisher in `row` innerhalb des horizontal
        # scrollbaren bar_canvas der Settings-Leiste — auf schmalen Fenstern
        # scrollten sie nach rechts weg, und ein laufender Server liess sich
        # nicht mehr stoppen, ohne die Leiste erst zu verschieben. Die
        # Kopfzeile scrollt nie. Sie wird ausserdem immer VOR
        # _build_settings_bar gebaut (siehe die drei Aufrufpaare in __init__,
        # _on_toggle_theme und _refresh_fonts), der Wirt existiert dort also
        # bereits.
        self._header_action_host = ttk.Frame(hdr, style="H.TFrame")
        self._header_action_host.pack(fill="x", padx=16, pady=(0, 8))
        # AP-QL-39 (Windows-v0.67): drei Spalten, gleiches Gewicht auf den
        # beiden aeusseren. Die mittlere traegt Run/Stop, das Paar sitzt damit
        # exakt in der waagerechten Fenstermitte; der Statustext lebt in
        # Spalte 2 und haengt nach rechts weg, ohne die Knoepfe aus der Mitte
        # zu ziehen. (pack side="left" konnte das nicht — es zentrierte die
        # ganze Gruppe INKLUSIVE des unterschiedlich langen Statustextes.)
        #
        # uniform= ist das, was tatsaechlich zentriert: weight=1 allein
        # verteilt nur den UEBERSCHUSS gleichmaessig, und Spalte 2 bringt den
        # Statustext als eigene natuerliche Breite mit — die Mitte driftete
        # dadurch um die halbe Textbreite nach links (in Windows mit -32 px
        # gemessen). Dieselbe uniform-Gruppe zwingt beide Aussenspalten auf
        # identische Breite, egal was drinsteht.
        self._header_action_host.columnconfigure(0, weight=1,
                                                 uniform="hdr_side")
        self._header_action_host.columnconfigure(2, weight=1,
                                                 uniform="hdr_side")
        self._header_btn_group = ttk.Frame(self._header_action_host,
                                           style="H.TFrame")
        self._header_btn_group.grid(row=0, column=1)

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
                    # AP-QL-36 (v0.65): kein VRAM mehr hier — jede Modellzeile
                    # zeigt es ohnehin pro GPU, und die Kopfzeile muss sich
                    # jetzt eine Zeile mit CUDA/Build/Libs teilen.
                    parts.append(f"GPU {g.index}: {short_gpu_name(g.name)}")   # AP-QL-78
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
            # AP-QL-48: KEINE Build-Angabe mehr. Der laufende Build steht
            # bereits gruen in der Slot-Leiste (AP-QL-43) — zweimal dieselbe
            # Information macht die Kopfzeile nur voll. Owner-Entscheidung
            # 01.08.: immer weg, auch bei einem frei gewaehlten Pfad ohne
            # Slot. Was der Build KANN, sagt weiterhin Libs: x/y.
            if dll_total > 0:
                if dll_found == dll_total:
                    cuda_parts.append(f"{RUNTIME_LIB_LABEL}: {dll_found}/{dll_total} ✓")
                    dll_color = self.theme.fg_secondary
                else:
                    cuda_parts.append(f"{RUNTIME_LIB_LABEL}: {dll_found}/{dll_total} ✗")
                    dll_color = self.theme.yellow
            else:
                dll_color = self.theme.fg_dim
                cuda_parts.append(f"{RUNTIME_LIB_LABEL}: server path not set")
            # AP-QL-36 (v0.65): fuehrender Trenner — dieses Label setzt die
            # GPU/RAM-Zeile links davon fort, statt eine eigene zu sein.
            self._cuda_label.config(
                text="  •  " + "  •  ".join(cuda_parts), foreground=dll_color)
        except Exception:
            pass

        # GPU buttons (uses cached self.gpus).
        try:
            self._build_gpu_buttons()
            self._build_unload_buttons()   # AP-QL-111
        except Exception:
            pass

        # Model cards and inventory (uses cached scan results).
        try:
            self._refresh_models_header()
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
        #
        # AP-QL-73 (Fund C): der Block stellte NUR die Zeilentoenung wieder
        # her. Die neu gebauten Run/Stop-Knoepfe kamen im Vorgabezustand
        # zurueck (Run blau, Stop grau) und das Statuslabel las "● Idle" —
        # ein Theme-Wechsel bei laufendem Server behauptete also, es laufe
        # nichts, und der Server war ueber die Oberflaeche nicht mehr
        # stoppbar, obwohl er weiter VRAM hielt. Woertlich aus der
        # archivierten Windows-Linie zurueckgeholt (dort v0.68 und v0.74).
        if self.running_model:
            try:
                self.after(70, self._update_running_indicator)
                self.after(70, self._refresh_action_btns)
                # Loading vs Running kommt aus _status_state — "Running" fuer
                # ein Modell zu behaupten, das noch laedt, waere dieselbe
                # Luege in die andere Richtung.
                fn = self.running_model
                prt = getattr(self, "_running_port", "8080")
                if self._status_state == "running":
                    self.after(70, lambda: self._update_status_running(fn, prt))
                elif self._status_state == "loading":
                    self.after(70, lambda: self._update_status_loading(fn, prt))
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
                    # AP-QL-36 (v0.65): kein VRAM mehr hier — jede Modellzeile
                    # zeigt es ohnehin pro GPU, und die Kopfzeile muss sich
                    # jetzt eine Zeile mit CUDA/Build/Libs teilen.
                    parts.append(f"GPU {g.index}: {short_gpu_name(g.name)}")   # AP-QL-78
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
            # AP-QL-48: keine Build-Angabe — siehe Begruendung oben.
            if dll_total > 0:
                if dll_found == dll_total:
                    cuda_parts.append(f"{RUNTIME_LIB_LABEL}: {dll_found}/{dll_total} ✓")
                    dll_color = self.theme.fg_secondary
                else:
                    cuda_parts.append(f"{RUNTIME_LIB_LABEL}: {dll_found}/{dll_total} ✗")
                    dll_color = self.theme.yellow
            else:
                dll_color = self.theme.fg_dim
                cuda_parts.append(f"{RUNTIME_LIB_LABEL}: server path not set")
            # AP-QL-36 (v0.65): fuehrender Trenner — dieses Label setzt die
            # GPU/RAM-Zeile links davon fort, statt eine eigene zu sein.
            self._cuda_label.config(
                text="  •  " + "  •  ".join(cuda_parts), foreground=dll_color)
        except Exception:
            pass

        try:
            self._build_gpu_buttons()
            self._build_unload_buttons()   # AP-QL-111
        except Exception:
            pass

        try:
            self._refresh_models_header()
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

        # AP-QL-73 (Fund C): derselbe Block wie im Theme-Wechsel. Der
        # Schriftzoom baut den Widget-Baum genauso neu auf und hatte die
        # Wiederherstellung nie — auch in der archivierten Windows-Linie
        # nicht. Das ist also eine LUECKE, kein Verlust; sie wird hier
        # mitgeschlossen, weil sie exakt dasselbe Symptom erzeugt
        # (Ctrl+Plus bei laufendem Server -> "● Idle", Stop grau).
        if self.running_model:
            try:
                self.after(70, self._update_running_indicator)
                self.after(70, self._refresh_action_btns)
                fn = self.running_model
                prt = getattr(self, "_running_port", "8080")
                if self._status_state == "running":
                    self.after(70, lambda: self._update_status_running(fn, prt))
                elif self._status_state == "loading":
                    self.after(70, lambda: self._update_status_loading(fn, prt))
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
        # AP-QL-32: die Adresse, an die der LAUFENDE Server gebunden hat.
        # Nicht _effective_server_host() zur Anzeigezeit nehmen — das liest
        # die LAN-Checkbox, und die ist der Wert fuer den NAECHSTEN Start.
        # Nach einem Klick auf LAN wuerde die Statuszeile sonst eine
        # Bindung behaupten, die der laufende Prozess gar nicht hat.
        self._running_host: Optional[str] = None

        # v0.55: settings bar lives inside a horizontally scrollable canvas.
        # On narrow windows the KV / LA / Probe / Port / Ctx / Timeout /
        # Run / Stop row used to clip — controls on the right disappeared
        # with no way to reach them. Now a horizontal scrollbar appears
        # on demand (only when content exceeds canvas width).
        bar_outer = tk.Frame(self, bg=t.bg)
        bar_outer.pack(fill="x", padx=16, pady=(8, 0))

        # AP-QL-27: Umbruch statt Scroll. Vorher lag diese Leiste in einem
        # horizontal scrollbaren Canvas — auf schmalen Fenstern verschwanden
        # die rechten Bedienelemente und waren nur ueber Shift+Mausrad
        # erreichbar (eine Geste, die man kennen muss). Jetzt fliesst der
        # Inhalt in so viele Zeilen um, wie die Fensterbreite verlangt.
        # Der Zeilenumbruch macht das Fenster erst schmal skalierbar.
        row = tk.Frame(bar_outer, bg=t.bg)
        row.pack(side="top", fill="x", expand=True)

        # AP-QL-49: DREI Container, neu geschnitten (Owner-Entscheidung 01.08.).
        #
        # AP-QL-37 trennte nach "bestimmt den naechsten Serverstart" (KV, LA,
        # Port, Ctx, LAN, Key, No Thinking) gegen "liest nur ein Benchmarklauf"
        # (Q/N/C, Probe, Timeout, Benchmark, Bench All). Diese Linie lief quer
        # durch eine zusammengehoerige Gruppe: KV, LA und alles, womit man
        # deren Wirkung misst, ist EIN Thema — Q/N/C setzt sogar direkt KV und
        # LA (_apply_nut_mode). Getrennt auf zwei Zeilen wirkte die Leiste
        # deshalb wirr.
        #
        # Neu wird nach Thema geschnitten:
        #   row        — Cache & Messung: KV, LA, Q/N/C, Probe, Timeout,
        #                Benchmark, Bench All
        #   net_row    — Server & Verbindung: Port, Ctx, Key (die drei
        #                Label+Feld-Paare, die ruhigste Zeile)
        #   toggle_row — Schalter: LAN, No Thinking
        #
        # Weiterhin eigene Fluss-Container statt fester Zeilen: jeder bricht
        # fuer sich um (AP-QL-27), die Themen vermischen sich also auch im
        # schmalen Fenster nie.
        net_row = tk.Frame(bar_outer, bg=t.bg)
        net_row.pack(side="top", fill="x", expand=True, pady=(4, 0))

        toggle_row = tk.Frame(bar_outer, bg=t.bg)
        toggle_row.pack(side="top", fill="x", expand=True, pady=(4, 0))

        kv_label = tk.Label(row, text="KV:", font=FONT_BODY_B, bg=t.bg, fg=t.fg)
        kv_label.pack(side="left")
        ToolTip(kv_label, "KV-Cache quantization for context compression", t)

        self._kv_var = tk.StringVar(value=self.cfg.get("kv_cache", "q8_0-K + turbo4-V"))
        self._kv_buttons = {}
        kv_frame = tk.Frame(row, bg=t.bg)
        kv_frame.pack(side="left", padx=(4, 0))

        kv_config = {
            "f16 (default)":     ("f16",      "16-bit float — full precision, no compression"),
            "f16-K + turbo4-V":  ("f16+t4",   "Keys at f16, Values at turbo4 — TheTom's recommended "
                                              "safe entry point: K stays uncompressed (K is far more "
                                              "quality-sensitive than V), only V is compressed. "
                                              "TurboQuant builds only."),
            "q8_0-K + turbo4-V": ("q8₀+t4",   "Keys at q8_0, Values at turbo4 — best for Q4_K_M models"),
            "turbo3 / turbo3":   ("t3/t3",    "Symmetric turbo3 — 5x compression, max context"),
            "turbo4 / turbo4":   ("t4/t4",    "Symmetric turbo4 — 4x compression"),
            "q8_0-K + turbo3-V": ("q8₀+t3",   "Keys at q8_0, Values at turbo3 — high compression"),
            "q8_0-K + turbo2-V": ("q8₀+t2",   "Keys at q8_0, Values at turbo2 — Boundary V. "
                                              "Auto enables mode 7 automatically (first/last 2 "
                                              "layers at q8_0-V); explicit LA=7 forces the same "
                                              "strategy. Explicit off (0) allows uniform turbo2-V, "
                                              "which can degrade PPL badly. TheTom 08.04.2026."),
            "q8_0 / q8_0":       ("q8₀/q8₀",  "Symmetric 8-bit quantization, 2x compression"),
        }
        # AP-QL-58: die Breite folgt der Beschriftung, statt fest zu sein.
        # Vorher stand hier width=68 — bemessen an "q8₀/q8₀" bei Zoomstufe
        # 100 %, wo dieses Label 70 px braucht. Schon bei 100 % lief es also
        # ueber, bei 250 % (seit AP-QL-56 erreichbar) braucht es 175 px.
        # HoverButton bekommt hier keine eigene Schrift, benutzt also seine
        # Vorgabe FONT_SMALL_B — genau in der wird gemessen.
        kv_btn_w = self._measure_label_group_width(
            [s for (s, _) in kv_config.values()],
            size=FONT_SMALL_B[1], bold=True, pad_chars=1.0)
        for full_name, (short, tip) in kv_config.items():
            is_active = (full_name == self._kv_var.get())
            color = ACCENT_TURBO if is_active else t.border
            btn = HoverButton(kv_frame, t, text=short, color=color,
                              width=kv_btn_w, height=24,
                              command=lambda n=full_name: self._select_kv(n))
            btn.pack(side="left", padx=1)
            ToolTip(btn, tip, t)
            self._kv_buttons[full_name] = btn
        # AP-QL-28: LA steht jetzt direkt neben KV. Beide bestimmen
        # gemeinsam die Cache-Konfiguration und gaten sich gegenseitig
        # (AP-QL-01/02) — sie gehoeren nebeneinander. Vorher stand LA
        # ganz am Ende der Leiste und landete beim Umbruch (AP-QL-27)
        # regelmaessig allein in einer eigenen Zeile.

        # AP-QL-16: von 'la_label' auf 'la_caption' umbenannt. Der Name
        # verdeckte sonst die portierte Modulfunktion la_label() in genau
        # dieser Methode — und die brauchen wir hier fuer den auto-Button.
        la_caption = tk.Label(row, text="  LA:", font=FONT_BODY_B, bg=t.bg, fg=t.fg)
        la_caption.pack(side="left", padx=(8, 0))
        ToolTip(la_caption, "Layer-Adaptive mode (TURBO_LAYER_ADAPTIVE)", t)

        self._la_var = tk.IntVar(value=self.cfg.get("layer_adaptive", 0))
        self._la_buttons = {}
        la_frame = tk.Frame(row, bg=t.bg)
        la_frame.pack(side="left", padx=(4, 0))

        # AP-QL-16: aus Windows v0.62 portiert (woertlich).
        la_config = {
            LA_MODE_AUTO: ("auto",
                 "Auto: TURBO_LAYER_ADAPTIVE unset — TheTom decides.\n"
                 "With turbo2-V (and ≥8 layers) mode 7 activates\n"
                 "automatically; otherwise behaves like off.\n"
                 "Only offered for turbo2-V presets."),
            0: ("off",
                 "Off: TURBO_LAYER_ADAPTIVE=0 set explicitly —\n"
                 "uniform quantization, no layer-adaptive behavior."),
            1: ("1",
                 "First 4 + last 4 layers K+V at q8_0 — only effective\n"
                 "with a Turbo-K cache (no effect with q8_0-K presets)."),
            5: ("5",
                 "First 2 + last 2 layers V=turbo4, remaining V=turbo2 —\n"
                 "only effective with a Turbo-V cache. Overrides the\n"
                 "selected V type layer-dependently."),
            7: ("7",
                 "Explicit Boundary V: first 2 + last 2 layers V=q8_0,\n"
                 "remaining V=turbo2 — only effective with a Turbo-V cache.\n"
                 "With turbo2-V, Auto selects the same mode automatically;\n"
                 "choose 7 to force it explicitly."),
        }
        # AP-QL-58: wie bei KV folgt die Breite der Beschriftung. Vorher
        # width=34 fuer ALLE fuenf Knoepfe — "auto" braucht aber schon bei
        # der Owner-Einstellung 0.75 volle 32 px und stand damit randlos
        # gequetscht da (Owner-Befund 02.08.2026, am Bildschirmfoto belegt);
        # bei 100 % sind es 40 px, bei 250 % 100 px.
        la_btn_w = self._measure_label_group_width(
            [s for (s, _) in la_config.values()],
            size=FONT_SMALL_B[1], bold=True, pad_chars=1.0)
        for mode, (short, tip) in la_config.items():
            is_active = (mode == self._la_var.get())
            color = ACCENT_TURBO if is_active else t.border
            btn = HoverButton(la_frame, t, text=short, color=color,
                              width=la_btn_w, height=24,
                              command=lambda m=mode: self._select_la(m))
            btn.pack(side="left", padx=1)
            ToolTip(btn, tip, t)
            self._la_buttons[mode] = btn


        # qnut v0.50: three mode buttons Q (Quality), N (Neutral),
        # C (Context). Sit between the KV row and the 🐿️qnut trigger
        # button. Disabled (greyed) when no qnut profile exists for
        # the current (model, server_slot) combination. When the user
        # clicks one, the corresponding (kv, la) tuple from the profile
        # is applied to the KV/LA buttons via _select_kv / _select_la
        # so the user sees what the mode means.
        #
        # AP-QL-50: EINGEREIHT, wie KV und LA (Owner-Vorgabe 01.08.).
        #
        # v0.50.2 gab jedem Knopf ein winziges Punktelabel DARUEBER ("3/3",
        # "2/3"). Solange keine Probe gelaufen ist, blieb dieses Label leer —
        # der Fluss richtet aber alles oben aus, also standen Q/N/C eine
        # halbe Etage unter KV/LA/Probe und rissen dazwischen eine Luecke
        # auf. Genau das machte die Zeile unruhig.
        #
        # Jetzt sitzen die drei Knoepfe direkt in der Reihe. Ob es etwas
        # anzuklicken gibt, sagt ihr Zustand: grau/deaktiviert ohne Profil,
        # klickbar sobald Probe-Ergebnisse vorliegen
        # (_refresh_mode_button_state).
        # Die Punktzahl ist nicht verloren — sie steht jetzt im Tooltip,
        # zusammen mit der Erklaerung des Modus.
        mode_frame = tk.Frame(row, bg=t.bg)
        mode_frame.pack(side="left", padx=(8, 0))
        self._mode_tooltips = {}
        self._mode_base_tips = {}
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
            # Nebeneinander wie die KV- und LA-Knoepfe, padx=1 als
            # Fortsetzung — die Gruppe bleibt damit beim Umbruch zusammen.
            btn = HoverButton(mode_frame, t, text=mode_label,
                              color=t.border, width=28, height=24,
                              command=lambda k=mode_key: self._apply_nut_mode(k))
            btn.pack(side="left", padx=1)
            self._mode_base_tips[mode_key] = mode_tip
            self._mode_tooltips[mode_key] = ToolTip(btn, mode_tip, t)
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

        port_label = tk.Label(net_row, text="Port:", font=FONT_BODY_B, bg=t.bg, fg=t.fg)
        # AP-QL-49: erstes Element seiner Zeile — kein Einzug, damit
        # Port: links buendig unter KV: steht.
        port_label.pack(side="left")
        ToolTip(port_label, "OpenAI-compatible API port (default: 8080)", t)
        self._port_var = tk.StringVar(value=str(self.cfg.get("port", 8080)))
        port_entry = tk.Entry(net_row, textvariable=self._port_var, width=5, font=FONT_BODY,
                              bg=t.entry_bg, fg=t.fg, relief="flat", bd=2,
                              insertbackground=t.fg)
        port_entry.pack(side="left", padx=(4, 0))

        # Context size field — passed as -c <ctx> when starting llama-server.
        # Leave empty to use llama-server / model default (256K for Gemma 4 etc.).
        ctx_label = tk.Label(net_row, text="  Ctx:", font=FONT_BODY_B, bg=t.bg, fg=t.fg)
        ctx_label.pack(side="left", padx=(8, 0))
        ToolTip(ctx_label, "Context size (-c). Leave empty to use llama-server / model default.\n"
                           "Set e.g. 8192 for bounded accuracy tests on models with huge native\n"
                           "context (Gemma 4 native = 256K → enormous KV VRAM reservation).", t)
        self._ctx_var = tk.StringVar(value=str(self.cfg.get("ctx_size", "")))
        ctx_entry = tk.Entry(net_row, textvariable=self._ctx_var, width=7, font=FONT_BODY,
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
        timeout_label = tk.Label(row, text="Timeout:", font=FONT_BODY_B,
                                 bg=t.bg, fg=t.fg)
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

        # AP-QL-26 (AP-QL-10b): ausdrueckliche LAN-Freigabe. Aus bindet
        # llama-server an 127.0.0.1 (nur diese Maschine), an bindet es an alle
        # Schnittstellen, damit Handys und andere Rechner im Heimnetz das
        # Modell erreichen. Der Zustand spiegelt cfg["server_host"]; eine dort
        # eingetragene Adapter-IP bleibt erhalten und wird statt 0.0.0.0
        # verwendet (siehe _effective_server_host / _save_current_config).
        self._lan_var = tk.BooleanVar(value=server_host_is_lan(self.cfg))
        # AP-QL-29 (v0.74): die vier Checkboxen der Settings-Leiste gehen
        # FONT_SMALL -> FONT_BODY, die Groesse, die die Beschriftungen
        # daneben (Port:, Ctx:, Timeout:) schon nutzen. Nur "No Thinking"
        # anzuheben liesse LAN direkt daneben unpassend stehen. Der
        # Helfer behaelt seine Vorgabe — cb_rec im Dialog bleibt SMALL.
        cb_lan = themed_checkbutton(toggle_row, t, text="LAN",
                                    variable=self._lan_var, font=FONT_BODY)
        # AP-QL-49: erstes Element seiner Zeile — siehe port_label.
        cb_lan.pack(side="left")
        ToolTip(cb_lan,
                "Aus: der Server bindet an 127.0.0.1 — nur von dieser\n"
                "Maschine erreichbar (sichere Vorgabe).\n\n"
                "An: der Server bindet an alle Schnittstellen (0.0.0.0) und\n"
                "ist von jedem Geraet im Netz erreichbar. Der Launcher\n"
                "schreibt die passende URL in die Startzeile.\n\n"
                "Achtung: llama.cpp erlaubt standardmaessig alle CORS-Quellen —\n"
                "wer den Port erreicht, kann das Modell benutzen, solange kein\n"
                "API-Schluessel gesetzt ist (Feld rechts daneben).", t)

        # AP-QL-26 (AP-QL-10c): optionaler API-Schluessel (--api-key). Leer
        # behaelt das bisherige schluessellose Verhalten. Auf dem Schirm
        # maskiert, weil das ganze Fenster in Screenshots landet.
        key_label = tk.Label(net_row, text="  Key:", font=FONT_BODY_B,
                             bg=t.bg, fg=t.fg)
        key_label.pack(side="left", padx=(6, 0))
        # AP-QL-33: der ECHTE Wert haengt an KEINEM Widget. Das Eingabefeld
        # zeigt eine eigene Anzeigevariable. Unfokussiert steht dort eine
        # Redigierung fester Laenge — show="•" allein maskiert zwar den Wert,
        # verraet aber ueber die Anzahl der Punkte die LAENGE des Schluessels.
        # Eine feste Redigierung verbirgt beides und ist abfotografierbar,
        # ohne etwas preiszugeben.
        self._api_key_var = tk.StringVar(value=load_api_key())
        self._api_key_display = tk.StringVar()
        key_entry = tk.Entry(net_row, textvariable=self._api_key_display, width=40,
                             font=FONT_BODY, bg=t.entry_bg, fg=t.fg,
                             relief="flat", bd=2, insertbackground=t.fg)
        key_entry.pack(side="left", padx=(4, 0))
        self._api_key_entry = key_entry
        key_entry.bind("<FocusIn>", self._on_api_key_focus_in)
        key_entry.bind("<FocusOut>", self._on_api_key_focus_out)
        key_entry.bind("<Return>", lambda e: self.focus_set())
        key_entry.bind("<Escape>", lambda e: self.focus_set())
        self._redact_api_key_field()
        _key_tip = ("Optionaler API-Schluessel fuer llama-server (--api-key).\n\n"
                    "Leer (Vorgabe): kein Schluessel — jeder Client, der den\n"
                    "Port erreicht, kann das Modell benutzen.\n\n"
                    "Gesetzt: jeder Client muss\n"
                    "  Authorization: Bearer <key>\n"
                    "senden. Denselben Wert in OpenWebUI / Continue usw.\n"
                    "eintragen. Dringend empfohlen zusammen mit LAN.\n\n"
                    "Der Launcher sendet ihn bei eigenen Probe-Anfragen\n"
                    "automatisch mit. Er ist hier maskiert und wird nie ins\n"
                    "Server-Log geschrieben.\n\n"
                    f"Liegt getrennt in {KEY_FILE.name} (nicht in der\n"
                    "Hauptkonfiguration) — aber im Klartext: diese Datei aus\n"
                    "Backups und Screenshots heraushalten.")
        ToolTip(key_label, _key_tip, t)
        ToolTip(key_entry, _key_tip, t)

        self._no_think_var = tk.BooleanVar(value=self.cfg.get("no_thinking", False))
        cb_think = themed_checkbutton(toggle_row, t, text="No Thinking",
                                       variable=self._no_think_var, font=FONT_BODY)
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
            toggle_row, text="", font=FONT_SMALL,
            bg=t.bg, fg=t.yellow)
        self._thinking_warn_label.pack(side="left", padx=(4, 0))
        self._thinking_warn_tooltip = ToolTip(
            self._thinking_warn_label, "", t)
        # Trigger the refresh on checkbox toggle. Model-selection changes
        # call _refresh_thinking_warning() directly from _select_cell
        # (there's no StringVar tracking the active model to trace on).
        self._no_think_var.trace_add(
            "write", lambda *_: self._refresh_thinking_warning())

        self._bench_var = tk.BooleanVar(value=self.cfg.get("benchmark", False))
        cb_bench = themed_checkbutton(row, t, text="Benchmark",
                                       variable=self._bench_var, font=FONT_BODY)
        cb_bench.pack(side="left", padx=(8, 0))
        ToolTip(cb_bench, "When checked, double-click runs llama-bench\ninstead of starting the server", t)

        self._bench_all_var = tk.BooleanVar(value=False)
        cb_bench_all = themed_checkbutton(row, t, text="Bench All",
                                           variable=self._bench_all_var, font=FONT_BODY)
        cb_bench_all.pack(side="left", padx=(2, 0))
        ToolTip(cb_bench_all, "Activate: selects ALL KV + LA for benchmarking.\n"
                              "Click individual KV/LA buttons to deselect.\n"
                              "Then press Run to start.", t)
        self._bench_all_var.trace_add("write", self._on_bench_all_toggled)

        # AP-QL-25 (AP-QL-13): Run / Stop / Status haengen am
        # nicht scrollenden Kopfzeilen-Wirt, nicht mehr an `row`.
        # canvas_bg folgt dem Kopfzeilen-Hintergrund — der Standardwert
        # (theme.bg) zeichnete dort oben einen sichtbar falschen Fleck.
        # AP-QL-39: Run/Stop haengen in der MITTLEREN Rasterspalte, damit sie
        # mittig sitzen; der Status kommt in Spalte 2 (siehe unten).
        host = self._header_btn_group
        self._run_btn = HoverButton(host, t, text="Run", color=t.accent,
                                     width=50, height=24,
                                     canvas_bg=t.bg_header,
                                     command=self._bench_run)
        self._run_btn.pack(side="left", padx=(0, 1))
        ToolTip(self._run_btn,
                "Start the selected model on the selected GPU.\n"
                "If 'Benchmark' or 'Bench All' is checked, runs a benchmark\n"
                "with the selected KV × LA configs instead.", t)

        self._stop_btn = HoverButton(host, t, text="Stop", color=t.accent,
                                      width=50, height=24,
                                      canvas_bg=t.bg_header,
                                      command=self._bench_stop)
        self._stop_btn.pack(side="left", padx=1)
        self._stop_btn.configure_btn(state="disabled")
        # AP-QL-78 (Fund F): der Tooltip war auf eine Zeile geschrumpft und
        # hatte zwei Aussagen verloren, die beide weiterhin GELTEN — der
        # Escape-Hinweis (gebunden) und dass ein Benchmark erst nach dem
        # laufenden Durchgang endet (siehe _bench_stop).
        ToolTip(self._stop_btn,
                "Stop whatever is running — benchmark or server.\n"
                "Escape does the same.\n"
                "A running benchmark stops after the current run finishes.", t)

        # AP-QL-111: die Unload-Knoepfe werden hier NICHT gebaut. `self.gpus`
        # ist zur Bauzeit der Kopfleiste noch leer -- der Scan laeuft erst
        # spaeter (`detect_all_gpus()` im Startlauf). Ein Bau an dieser Stelle
        # ergaebe null Knoepfe, und zwar lautlos. Gefunden, weil die Suite die
        # echten Widgets misst statt dem Code zu glauben.
        self._unload_btns = {}

        # AP-QL-39 (Windows-v0.67): Spalte 2 — rechts der zentrierten
        # Knoepfe, linksbuendig, damit der Text nach aussen waechst und
        # Run/Stop nie verschiebt.
        self._status_label = tk.Label(self._header_action_host, text="● Idle",
                                       font=FONT_BODY_B,
                                       bg=t.bg_header, fg=t.fg_dim,
                                       cursor="arrow")
        self._status_label.grid(row=0, column=2, sticky="w", padx=(16, 0))
        self._status_tooltip = ToolTip(self._status_label, "", t)
        self._status_label.bind("<Button-1>", self._on_status_click)

        # AP-QL-27: ab hier uebernimmt der Fluss-Umbruch die Settings-Leiste.
        self._install_settings_flow(row, net_row, toggle_row)

    # ─── AP-QL-27: Fluss-Layout der Settings-Leiste ───────────────────────

    # Ab diesem linken Abstand gilt ein Element als Gruppenanfang, vor dem
    # umgebrochen werden darf. Der Bestandscode kodiert die Gruppen bereits:
    # Gruppenanfaenge nutzen padx=(8, 0) bzw. (6, 0), Fortsetzungen (4, 0)
    # oder padx=1. So bleibt "Port:" mit seinem Eingabefeld zusammen und die
    # KV-Knopfreihe wird nicht mittendrin zerschnitten.
    _FLOW_BREAK_PAD = 6
    _FLOW_LINE_GAP = 4       # senkrechter Abstand zwischen den Zeilen

    @staticmethod
    def _flow_pad(value) -> tuple:
        """padx aus pack_info() in (links, rechts) uebersetzen.

        pack_info() liefert je nach Tk-Fassung einen int, eine Zeichenkette
        ("8" oder "8 0") oder ein Tupel. Alles drei muss hier ankommen.

        Args:
            value: Der padx-Wert, wie pack_info() ihn liefert.

        Returns:
            (links, rechts) als int-Tupel.
        """
        if isinstance(value, (tuple, list)):
            parts = list(value)
        else:
            parts = str(value).split()
        try:
            nums = [int(float(x)) for x in parts]
        except (TypeError, ValueError):
            return (0, 0)
        if not nums:
            return (0, 0)
        if len(nums) == 1:
            return (nums[0], nums[0])
        return (nums[0], nums[1])

    def _install_settings_flow(self, *rows: tk.Frame) -> None:
        """Die gepackten Kinder der Leiste auf Fluss-Layout umstellen.

        Die Abstaende werden VOR dem pack_forget aus pack_info() gelesen —
        danach sind sie weg. Reihenfolge und Abstaende bleiben damit exakt
        die des Bestandscodes, nur die Platzierung wechselt von pack auf
        place, weil pack nicht umbrechen kann.

        Args:
            row: Der Rahmen, dessen Kinder umgestellt werden.
        """
        # AP-QL-37: mehrere Container, jeder mit eigener Elementliste.
        self._flow_rows = []
        if not hasattr(self, "_flow_right_widgets"):
            self._flow_right_widgets = set()
        for row in rows:
            # AP-QL-42: seitenbewusst. winfo_children() liefert die
            # ERZEUGUNGS-Reihenfolge; bei side="right" ist die visuelle
            # Reihenfolge aber genau umgekehrt (das zuerst erzeugte Widget
            # sitzt ganz rechts). Ohne diese Trennung haette der Fluss die
            # rechte Gruppe der Fussleiste gespiegelt — About stuende dann
            # vor Save Results.
            left_items, right_items = [], []
            for w in row.winfo_children():
                try:
                    info = w.pack_info()
                except tk.TclError:
                    continue      # nicht per pack verwaltet
                left, right = self._flow_pad(info.get("padx", 0))
                entry = (w, left, right)
                if str(info.get("side", "left")) == "right":
                    right_items.append(entry)
                    # AP-QL-44: gemerkt, damit _reflow_one sie an den rechten
                    # Rand setzen kann statt sie nur einzureihen.
                    self._flow_right_widgets.add(str(w))
                else:
                    left_items.append(entry)
                w.pack_forget()
            items = left_items + list(reversed(right_items))
            self._flow_rows.append((row, items))
            row.bind("<Configure>", self._on_flow_configure)
        self._flow_last_w = -1
        # Auf <Configure> von row reagieren. Der Umbruch aendert nur die
        # HOEHE, nie die Breite — die Breitenwaechter-Abfrage in
        # _reflow_settings_bar verhindert deshalb die Rueckkopplung.
        self.after(0, self._reflow_settings_bar)

    def _reinstall_slot_flow(self) -> None:
        """Den Fluss der Slot-Leiste erneuern (AP-QL-43).

        Ersetzt einen vorhandenen Eintrag fuer _slot_bar in _flow_rows, statt
        anzuhaengen — sonst wuechse die Liste bei jedem Enginewechsel und
        enthielte lauter tote Widgets.
        """
        bar = getattr(self, "_slot_bar", None)
        if bar is None:
            return
        rows = [(r, it) for r, it in getattr(self, "_flow_rows", [])
                if r is not bar]
        items = []
        for w in bar.winfo_children():
            try:
                info = w.pack_info()
            except tk.TclError:
                continue
            left, right = self._flow_pad(info.get("padx", 0))
            items.append((w, left, right))
            w.pack_forget()
        bar.bind("<Configure>", self._on_flow_configure)
        self._flow_rows = rows + [(bar, items)]
        self._flow_last_w = -1
        self.after(0, lambda: self._reflow_settings_bar(force=True))

    def _install_footer_flow(self, bar) -> None:
        """Die Fussleiste zusaetzlich in den Fluss aufnehmen (AP-QL-42).

        _install_settings_flow setzt _flow_rows neu; die Fussleiste wird
        spaeter gebaut und muss deshalb ANGEHAENGT werden, sonst verloere die
        Settings-Leiste ihren Umbruch.

        Args:
            bar: Der Rahmen der Fussleiste.
        """
        existing = list(getattr(self, "_flow_rows", []))
        self._install_settings_flow(bar)
        self._flow_rows = existing + self._flow_rows
        self._flow_last_w = -1

    def _on_flow_configure(self, event=None) -> None:
        """<Configure>-Haken: nur bei echter Breitenaenderung neu umbrechen."""
        self._reflow_settings_bar()

    def _reflow_settings_bar(self, force: bool = False) -> None:
        """Die Elemente aller Fluss-Container neu umbrechen.

        Args:
            force: True bricht auch dann um, wenn die Breite gleich blieb
                (nach Schriftgroessenwechsel noetig — die Elemente aendern
                dann ihre Wunschbreite, das Fenster aber nicht).
        """
        rows = getattr(self, "_flow_rows", None)
        if not rows:
            return
        # Die Breite wird EINMAL am ersten Container gemessen: beide sind mit
        # fill="x" im selben Elternwidget, haben also dieselbe. Je Container
        # zu messen waere nicht nur ueberfluessig, sondern falsch — ein noch
        # nicht gelayouteter zweiter Container haette Breite 1 und wuerde
        # jedes seiner Elemente in eine eigene Zeile zwingen.
        try:
            if not rows[0][0].winfo_exists():
                return
            avail = rows[0][0].winfo_width()
        except tk.TclError:
            return
        if avail <= 1:
            self.after(50, self._reflow_settings_bar)
            return
        if not force and avail == getattr(self, "_flow_last_w", -1):
            return
        self._flow_last_w = avail

        for row, flow_items in rows:
            self._reflow_one(row, flow_items, avail)

    def _reflow_one(self, row: tk.Frame, flow_items: list, avail: int) -> None:
        """Einen einzelnen Fluss-Container umbrechen.

        Args:
            row: Der Container.
            flow_items: Liste (Widget, linker Abstand, rechter Abstand).
            avail: Verfuegbare Breite in Pixeln.
        """
        # Erst Gruppen bilden, DANN umbrechen. Die Umbruchentscheidung muss
        # die Gesamtbreite der Gruppe kennen: entschied man sie allein am
        # Gruppenanfang, passte z. B. "Port:" gerade noch in die Zeile, sein
        # Eingabefeld aber nicht mehr — und weil das Feld nicht umbrechen
        # darf, ragte es aus dem Fenster. Eine Gruppe ist ein Element mit
        # Gruppenabstand plus alle folgenden Fortsetzungen.
        right_ids = getattr(self, "_flow_right_widgets", set())
        groups: List[list] = []
        prev_right = None
        for item in flow_items:
            pad_l = item[1]
            is_right = str(item[0]) in right_ids
            # AP-QL-44: auch bei einem SEITENWECHSEL trennen. Die Knoepfe der
            # Fussleiste nutzen padx=2 und liegen damit unter der
            # Umbruchschwelle — ohne diese Bedingung verschmolz die ganze
            # Zeile zu EINER Gruppe, deren erstes Widget links steht, und die
            # rechte Gruppe existierte nie.
            #
            # AP-QL-46: rechtsseitige Elemente bilden NIE eine gemeinsame
            # Gruppe. Die Konvention "kleiner Abstand = gehoert zusammen"
            # beschreibt die Settings-Leiste, wo eine Beschriftung an ihrem
            # Eingabefeld haengt. Rechts gepackt wird in dieser App
            # ausschliesslich die Fussleiste, und deren Knoepfe sind
            # voneinander unabhaengig. Zusammengefasst ergaben sie EINE
            # unteilbare Gruppe von ~626 px, die bei schmalem Fenster nicht
            # umbrechen KONNTE und deshalb hinausragte — gemessen bis 608 px
            # bei 600 und bei 400 px Breite, beide Male identisch.
            if (groups and pad_l < self._FLOW_BREAK_PAD
                    and is_right == prev_right and not is_right):
                groups[-1].append(item)
            else:
                groups.append([item])
            prev_right = is_right

        # AP-QL-44: rechtsbuendige Gruppen von den linken trennen. Sie sollen
        # am rechten Rand kleben (Owner-Wunsch), nicht bloss hinten in der
        # Reihe stehen — sonst wandern Save Results / Autoload / Paths / About
        # bei jeder Fensterbreite woandershin.
        left_groups, right_groups = [], []
        for g in groups:
            (right_groups if str(g[0][0]) in right_ids else left_groups).append(g)

        def measure(group):
            """(Liste der Masse, Gesamtbreite) einer Gruppe."""
            out, total = [], 0
            for w, pad_l, pad_r in group:
                try:
                    rw, rh = w.winfo_reqwidth(), w.winfo_reqheight()
                except tk.TclError:
                    continue
                out.append((w, pad_l, rw, rh))
                total += pad_l + rw + pad_r
            return out, total

        # AP-QL-46: die Schwelle faellt EINMAL, vor jeder Platzierung. Passt
        # der rechte Block nicht in die Breite, bringt Rechtsbuendigkeit
        # nichts mehr — er wuerde nur rechts hinausragen. Dann gibt es
        # unterhalb dieser Schwelle gar keine Rechtsbuendigkeit: die Gruppen
        # wandern in den linken Fluss und brechen dort um wie alles andere.
        # Vorher entschied dieselbe Frage erst NACH dem linken Durchlauf und
        # brauchte dafuer eine zweite, wortgleiche Platzierungsschleife.
        # Sichtbar bleiben ist wichtiger als buendig.
        right_blocks = [measure(g) for g in right_groups]
        right_total = sum(bw for _m, bw in right_blocks)
        if right_groups and right_total > avail:
            left_groups += right_groups
            right_groups, right_blocks, right_total = [], [], 0

        x = y = line_h = 0
        for group in left_groups:
            measured, group_w = measure(group)
            if not measured:
                continue
            # Nie als erstes Element einer Zeile umbrechen — eine Gruppe, die
            # allein schon breiter als das Fenster ist, ragt dann eben heraus,
            # statt eine leere Zeile zu erzeugen.
            if x > 0 and x + group_w > avail:
                x = 0
                y += line_h + self._FLOW_LINE_GAP
                line_h = 0
            for w, pad_l, rw, rh in measured:
                try:
                    w.place(x=x + pad_l, y=y)
                except tk.TclError:
                    continue
                x += pad_l + rw
                line_h = max(line_h, rh)
            x += group[-1][2]

        if right_groups:
            # Passt der rechte Block nicht mehr neben das Linke, bekommt er
            # eine eigene Zeile — bleibt aber rechtsbuendig. Dass er in die
            # Breite ueberhaupt passt, ist oben schon entschieden.
            if x > 0 and x + right_total > avail:
                y += line_h + self._FLOW_LINE_GAP
                line_h = 0
            rx = max(0, avail - right_total)
            for measured, _bw in right_blocks:
                for w, pad_l, rw, rh in measured:
                    try:
                        w.place(x=rx + pad_l, y=y)
                    except tk.TclError:
                        continue
                    rx += pad_l + rw
                    line_h = max(line_h, rh)

        try:
            row.configure(height=y + line_h)
        except tk.TclError:
            pass
    def _on_status_click(self, event=None):
        """Open browser when server is running."""
        if self.server_process:
            port = self._port_var.get() or "8080"
            import webbrowser
            # AP-QL-32: nicht hart localhost — bei einer Bindung an eine
            # bestimmte Adapter-IP ist der Server dort NICHT erreichbar.
            webbrowser.open(self._server_open_url(port))

    def _server_bind_url(self, port: str) -> str:
        """Die URL, wie sie der Bindung entspricht — zum ANZEIGEN (AP-QL-32).

        Zeigt genau das, was der Nutzer gewaehlt hat: 127.0.0.1 bei privat,
        0.0.0.0 bei LAN-Freigabe, oder eine von Hand gesetzte Adapter-IP.
        Bevorzugt die Bindung des laufenden Prozesses; erst wenn keiner
        laeuft, die aktuelle Einstellung.

        Args:
            port: Der Port als Zeichenkette.

        Returns:
            Die vollstaendige URL.
        """
        host = self._running_host or self._effective_server_host()
        return f"http://{host}:{port}"

    def _server_open_url(self, port: str) -> str:
        """Die URL, die im Browser tatsaechlich funktioniert (AP-QL-32).

        0.0.0.0 ist eine Bindungsangabe, kein Ziel — ein Browser kann sie
        nicht aufrufen. Bei LAN-Freigabe wird deshalb die echte Adresse des
        Rechners genommen, und wenn die nicht zu ermitteln ist, Loopback.

        Args:
            port: Der Port als Zeichenkette.

        Returns:
            Eine aufrufbare URL.
        """
        host = self._running_host or self._effective_server_host()
        if host == SERVER_HOST_ALL:
            host = detect_lan_ip() or SERVER_HOST_LOOPBACK
        return f"http://{host}:{port}"

    def _update_status_loading(self, filename: str, port: str):
        """Set status label to Loading state (model not yet ready)."""
        t = self.theme
        self._status_state = "loading"          # AP-QL-73
        short = filename[:40] + "…" if len(filename) > 40 else filename
        self._status_label.config(
            text=f"⏳ Loading — {short} @ {self._server_bind_url(port)}",
            fg=t.yellow, cursor="arrow")
        self._status_tooltip.update_text("")

    def _update_status_running(self, filename: str, port: str):
        """Set status label to Running state with browser tooltip."""
        t = self.theme
        self._status_state = "running"          # AP-QL-73
        self._status_label.config(
            text=f"● Running — {filename} @ {self._server_bind_url(port)}",
            fg=t.green, cursor="hand2")
        self._status_tooltip.update_text(
            f"Click to open in browser\nhttp://localhost:{port}")

    def _update_status_idle(self, text="● Idle"):
        """Reset status label to idle state."""
        t = self.theme
        self._status_state = "idle"             # AP-QL-73
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

    def _current_server_family(self) -> str:
        """Return the family of the currently configured llama-server.

        AP-QL-01 (via AP-QL-16). Recomputed from cfg on every call (cheap
        string check) so the result can never go stale after a Quick-Switch.
        """
        return detect_server_family(self.cfg.get("llama_server_path", ""))

    def _select_kv(self, name: str):
        # AP-QL-01: refuse KV configs the active server family does not
        # support. This method is also reachable programmatically (qnut mode
        # apply, profile load), so the guard cannot live in the button alone.
        if not is_kv_allowed(name, self._current_server_family()):
            self._log(f"KV config '{name}' not supported by the active "
                      f"server ({self._current_server_family()}) — ignored.", "warn")
            return
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
            # AP-QL-02: LA validity depends on the KV preset — re-gate
            # immediately after every KV change.
            #
            # AP-QL-75 (Fund A2) holt diesen Aufruf zurueck. Er ist zugleich
            # Fund F5 der Suiten-Kampagne: ohne ihn blieb ein stehendes,
            # jetzt wirkungsloses LA einfach ausgewaehlt, statt sichtbar auf
            # off zurueckzufallen.
            self._apply_la_gating()
            # qnut v0.50: a manual KV click might break the active mode
            # binding (Q/N/C). Clear it if the new selection no longer
            # matches the active anchor.
            self._clear_active_mode_if_user_drifted()

    def _select_la(self, mode: int):
        # AP-QL-02: refuse LA modes without a real effect for the current
        # KV+family tuple. In Bench-All mode the union over all allowed KV
        # presets applies, because the matrix pairs them per tuple later.
        fam = self._current_server_family()
        if self._bench_all_var.get():
            _ok = mode in family_la_union(fam)
        else:
            _ok = is_la_allowed(mode, self._kv_var.get(), fam)
        if not _ok:
            # AP-QL-78 (Fund F): den ZUTREFFENDEN Grund nennen. Im
            # Bench-All-Modus lief die Ablehnung ueber die Familien-Union,
            # der Text nannte trotzdem den aktuellen KV-Preset — also einen
            # Grund, der mit der Ablehnung nichts zu tun hatte.
            if self._bench_all_var.get():
                self._log(f"LA mode '{la_label(mode)}' not supported by the "
                          f"active server ({fam}) — ignored.", "warn")
            else:
                self._log(f"LA mode '{la_label(mode)}' has no valid effect for "
                          f"{self._kv_var.get()!r} on this {fam} server — "
                          f"ignored.", "warn")
            return
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
            # Activate: enable Benchmark, pre-select the valid KV and LA.
            #
            # AP-QL-75 (Fund A2): stand hier bis 03.08. auf
            # `set(KV_CACHE_OPTIONS.keys())` und `set(self._la_buttons)` —
            # also pauschal ALLES. Der Bestaetigungsdialog rechnete mit
            # dieser ungefilterten Liste und kuendigte auf einem
            # Mainline-Build z. B. 84 Laeufe an, von denen der Worker
            # hinterher 78 aussortierte.
            self._bench_var.set(True)
            # AP-QL-01: pre-select only the KV configs the active server
            # family supports (mainline: no turbo types).
            self._bench_kv_set = set(allowed_kv_names(self._current_server_family()))
            # AP-QL-02: pre-select only LA modes valid for at least one
            # allowed KV preset of the active family (mainline: only 0).
            self._bench_la_set = (set(self._la_buttons.keys())
                                  & family_la_union(self._current_server_family()))
            self._refresh_kv_button_visuals()
            # AP-QL-02: color only the actually pre-selected LA modes —
            # the set is family-filtered and may not contain every button.
            for m, btn in self._la_buttons.items():
                btn.configure_btn(
                    color=ACCENT_TURBO if m in self._bench_la_set else t.border)
        else:
            # Deactivate: revert to single-select visuals
            self._bench_kv_set.clear()
            self._bench_la_set.clear()
            self._refresh_kv_button_visuals()
            cur_la = self._la_var.get()
            for m, btn in self._la_buttons.items():
                btn.configure_btn(color=ACCENT_TURBO if m == cur_la else t.border)
        # AP-QL-02: toggle changes the allowed LA set (family union vs.
        # per-KV) — re-gate the buttons for the new mode.
        self._apply_la_gating()

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
        # AP-QL-26: ohne Bearer-Kopfzeile antwortet ein Server mit
        # Schluessel mit 401 — das Warten liefe in den Timeout.
        _hdrs = api_auth_headers(self._api_key_value())
        while time.monotonic() < deadline:
            if self._nut_cancel_event.is_set():
                return False
            try:
                with urllib.request.urlopen(
                        urllib.request.Request(url, headers=_hdrs),
                        timeout=2) as resp:
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
        # AP-QL-26: ohne Kopfzeile deutet ein 401 als Exception das
        # Warten faelschlich als "Server ist unten" — der naechste
        # Lauf startete dann gegen einen noch belegten Port.
        _hdrs = api_auth_headers(self._api_key_value())
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                        urllib.request.Request(url, headers=_hdrs),
                        timeout=1) as resp:
                    _ = resp.status  # laeuft noch
            except urllib.error.HTTPError:
                # AP-QL-66 (F7): an HTTP error response (401/404/...) is an
                # ANSWER — the server is alive. Treating it as "down" risked
                # a port collision on qnut restarts whenever the configured
                # key was empty or stale (AP-QL-10c consumer contract).
                pass
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

        # Pick the GPU the user has highlighted in the model card. No
        # server is running at this point (checked above), so
        # _running_gpu_key is always None here — use the selected row.
        gpu_key = self._selected_gpu_key()

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

        # AP-QL-68 (F1): rechtsbuendig wie alle Dialoge. Cancel zuerst
        # gepackt = ganz rechts, Start links davon — sichtbare Reihenfolge
        # [Start][Cancel] unveraendert.
        cancel_btn = HoverButton(btn_row, t, text="Cancel",
                                 color=t.border, width=110, height=28)
        cancel_btn.pack(side="right")

        start_btn = HoverButton(btn_row, t, text="Start",
                                color=ACCENT_TURBO, width=110, height=28)
        start_btn.pack(side="right", padx=(0, 6))

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
        self._nut_dialog_close_btn.pack(side="right")   # AP-QL-68 (F1)

    def _nut_worker(self, model, gpu_key: str):
        """Background thread: run the full qnut v0.50 three-phase workflow.

        Phase 0 — Baseline: f16 + LA=off, collect responses (no judgement).
        Phase 1 — KV sweep: the family's remaining KVs at LA=off, scored
                  against the baseline.
        Phase 2 — LA sweep: for each KV with score >= 2 from phase 1, test
                  the LA modes that actually have a distinct effect for that
                  KV (AP-QL-05 — no longer a fixed 1/5/7 for every preset).

        Then derives the three (KV, LA) anchors quality / neutral / context
        and persists them as a profile under the current (model, server)
        combination in cfg['nut_profiles'].
        """
        # AP-QL-05 (via AP-QL-16): bind the family ONCE to the executable
        # this worker captured. Without this qnut swept NUT_KV_ORDER
        # wholesale and would have measured turbo tuples on a mainline
        # server, which silently falls back to f16: the profile would carry
        # turbo labels over f16 numbers.
        worker_family = detect_server_family(
            self.cfg.get("llama_server_path", ""))

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
            """Stable display key for the dialog row dict.

            AP-QL-05: la_label() instead of the raw int — phase 2 can now
            carry Auto (LA_MODE_AUTO = -1), which would otherwise render as
            "LA=-1". All three key builders must stay in sync.
            """
            return f"{kv}|LA={la_label(la)}"

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
                api_key=self._api_key_value(),   # AP-QL-26
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
                    baseline_kv, baseline_la, worker_family))

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
                # AP-QL-05: only the presets the captured family really
                # supports — not all of NUT_KV_ORDER.
                phase1_kvs = [kv for kv in nut_kv_order(worker_family)
                              if kv != baseline_kv]
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
                candidates = [
                    kv for kv in nut_kv_order(worker_family)
                    if scores.get((kv, 0), 0) >= 2 and kv != baseline_kv
                ]
                # Always include the baseline KV in the LA sweep too — we
                # want to know whether f16 with a non-zero LA still
                # matches f16 with LA=off.
                candidates = [baseline_kv] + candidates

                # Pre-populate the phase-2 rows now that we know which
                # tuples will actually run. This replaces the placeholder
                # row with the real list.
                # AP-QL-05: per-KV LA modes instead of candidates × (1,5,7).
                # The old cross product produced e.g. LA=1 on q8_0-K presets
                # (mode 1 needs a Turbo-K) and never tested Auto at all.
                phase2_tuples = [(kv, la) for kv in candidates
                                 for la in nut_la_extras(kv, worker_family)]
                self._main_thread_call(
                    lambda t=list(phase2_tuples):
                        self._nut_prepopulate_phase2_rows(t))

                total_phase2 = len(phase2_tuples)
                done = 0
                # AP-QL-05: iterate the planned tuples themselves — the old
                # nested candidates × la_extras loop is exactly the cross
                # product this package removes. la_label() keeps Auto (-1)
                # readable; a raw f"LA={la}" would print "LA=-1".
                for kv, la in phase2_tuples:
                    if self._nut_cancel_event.is_set():
                        cancelled = True
                        break
                    done += 1
                    _set_overall(
                        f"Phase 2/2: LA sweep [{done}/{total_phase2}]  "
                        f"{kv}, LA={la_label(la)}")
                    score, _resps, results = _run_one_tuple(
                        kv, la, is_baseline=False)
                    if score is not None:
                        scores[(kv, la)] = score
                        all_results[(kv, la)] = results

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
        target_gpu_key = self._running_gpu_key or self._selected_gpu_key()

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

        AP-QL-50: die Punktelabel UEBER den Knoepfen sind entfallen —
        stattdessen wird hier je Modus der Tooltip fortgeschrieben
        (Basis-Erklaerung + Anker + "Treue zur f16-Grundlinie: x/3").
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
            # AP-QL-50: der Tooltip traegt die Punktzahl, seit die eigene
            # Etage ueber dem Knopf entfallen ist. Basis-Tipp + eine Zeile.
            tip = self._mode_tooltips.get(mode_key)
            base_tip = self._mode_base_tips.get(mode_key, "")

            def set_tip(extra: str) -> None:
                if tip is not None:
                    tip.update_text(base_tip + extra)

            if not has_profile:
                btn.configure_btn(color=t.border, state="disabled")
                set_tip("\n\nNoch keine Probe-Ergebnisse fuer dieses Modell "
                        "+\ndiesen Server — erst ▶ Probe laufen lassen.")
                continue

            anchor = self._current_nut_profile.get(mode_key)
            if not isinstance(anchor, dict) or not anchor.get("kv"):
                btn.configure_btn(color=t.border, state="disabled")
                set_tip("\n\nDie Probe hat fuer diesen Modus keinen Anker "
                        "gefunden.")
                continue

            is_active = (self._active_mode == mode_key)
            btn.configure_btn(
                color=ACCENT_TURBO if is_active else t.border,
                state="normal",
            )

            # Look up the score for this anchor's (kv, la) tuple. The
            # quality anchor (f16, LA=0) is always 3/3 by definition
            # even if it's not explicitly in the scores dict.
            kv = anchor["kv"]
            la = int(anchor["la"])
            score = scores.get(f"{kv}|LA{la}")
            if score is None and mode_key == "quality":
                # f16 baseline is always 3/3 by definition
                score = 3
            set_tip(f"\n\nAnker: {kv}, LA={la}\n"
                    f"Treue zur f16-Grundlinie: "
                    f"{'?' if score is None else score}/3")

    # ─── Familien-Gating (AP-QL-75, Fund A2) ──────────────────────────────
    #
    # Woertlich aus der archivierten Windows-Linie zurueckgeholt. Beim
    # Zusammenlegen der beiden Linien waren beide Methoden ersatzlos
    # entfallen (_apply_kv_family_gating 3 Verwendungen -> 0,
    # _apply_la_gating 5 -> 0). Folge im Betrieb: nach einem Quick-Switch
    # auf einen Mainline-Build sahen alle turbo-Knoepfe klickbar aus, taten
    # beim Klick aber nichts (stille Ablehnung in _select_kv), und Bench All
    # kuendigte eine Matrix an, von der der Worker hinterher den Grossteil
    # aussortierte.

    def _apply_kv_family_gating(self):
        """Enable/disable KV buttons for the active server family.

        Called from _do_initial_scan, i.e. on startup, after every
        Quick-Switch click and after every Paths dialog save. Disables
        the buttons of unsupported KV configs, prunes them from a
        pending Bench-All selection, and visibly resets an invalid
        current selection to the safe f16 fallback (with a log line).
        Idempotent: repeated engine switches always converge to the
        same button states and the same fallback.
        """
        family = self._current_server_family()
        allowed = set(allowed_kv_names(family))

        for name, btn in self._kv_buttons.items():
            btn.configure_btn(state="normal" if name in allowed else "disabled")

        # Prune an active Bench-All toggle selection of now-invalid entries.
        if self._bench_kv_set - allowed:
            dropped = sorted(self._bench_kv_set - allowed)
            self._bench_kv_set &= allowed
            self._log(f"Bench-All selection pruned for {family} server: "
                      f"removed {', '.join(dropped)}", "warn")

        cur = self._kv_var.get()
        if cur not in allowed:
            self._log(f"KV config '{cur}' is not supported by the active "
                      f"{family} server — falling back to {SAFE_KV_FALLBACK}.",
                      "warn")
            self._kv_var.set(SAFE_KV_FALLBACK)
            self._update_bars_for_kv(SAFE_KV_FALLBACK)
            self._clear_active_mode_if_user_drifted()
        self._refresh_kv_button_visuals()
        # AP-QL-02: LA validity depends on family AND KV preset — re-gate
        # after every engine change (and after the KV fallback above).
        self._apply_la_gating()

    def _apply_la_gating(self):
        """Enable/disable LA buttons for the current family + KV preset.

        Same contract as _apply_kv_family_gating: idempotent, resets an
        invalid selection visibly to 0 (off) with a log line, prunes a
        pending Bench-All LA selection. In Bench-All toggle mode the
        buttons follow the family union (per-tuple validity is enforced
        in the worker); in normal mode they follow the current KV preset.
        """
        t = self.theme
        family = self._current_server_family()
        if self._bench_all_var.get():
            allowed = set(self._la_buttons.keys()) & family_la_union(family)
        else:
            allowed = set(allowed_la_modes(self._kv_var.get(), family))

        for mode, btn in self._la_buttons.items():
            btn.configure_btn(state="normal" if mode in allowed else "disabled")

        if self._bench_la_set - allowed:
            dropped = sorted(self._bench_la_set - allowed)
            self._bench_la_set &= allowed
            self._log(f"Bench-All LA selection pruned for {family} server: "
                      f"removed {', '.join(la_label(m) for m in dropped)}",
                      "warn")

        cur = self._la_var.get()
        if not self._bench_all_var.get() and cur not in allowed:
            self._log(f"LA mode '{la_label(cur)}' is not valid for "
                      f"{self._kv_var.get()} on the active {family} server "
                      f"— falling back to off (0).", "warn")
            self._la_var.set(0)
            cur = 0
        # Refresh single-select visuals (Bench-All visuals are handled by
        # the toggle logic itself).
        if not self._bench_all_var.get():
            for m, btn in self._la_buttons.items():
                btn.configure_btn(color=ACCENT_TURBO if m == cur else t.border)

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

    def _nut_prepopulate_phase1_rows(self, baseline_kv: str, baseline_la: int,
                                     family: str):
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
        # AP-QL-05: nut_kv_order() statt ganz NUT_KV_ORDER — auf einem
        # Mainline-Server listete der Dialog sonst Turbo-Zeilen, die nie laufen.
        for kv in nut_kv_order(family):
            row_label = f"{kv}|LA={la_label(baseline_la)}"
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
            row_label = f"{kv}|LA={la_label(la)}"
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
            self._refresh_unload_buttons()   # AP-QL-111
            self._bench_thread = threading.Thread(
                target=self._run_benchmark_all,
                args=(model, gpu_key, kv_list, la_list, is_cpu),
                daemon=True)
            self._bench_thread.start()
        elif self._bench_var.get():
            # Single benchmark mode
            self._run_bench_for_selection()

    def _bench_stop(self):
        """Stop-Knopf: stoppt, was gerade laeuft — Benchmark oder Server.

        AP-QL-25 (Portierung von AP-QL-13): frueher brach dieser Knopf nur
        einen Benchmark ab und sonst nichts. Lief ein Server, sass er grau da,
        und der einzige Weg zum Stoppen war die nicht auffindbare Escape-Taste.
        Ein Knopf, eine Bedeutung: stoppe das, was gerade laeuft. Der Benchmark
        hat Vorrang, weil er waehrend eines Matrixlaufs den Serverlebenszyklus
        selbst besitzt.
        """
        if self._bench_thread and self._bench_thread.is_alive():
            self._bench_stop_event.set()
            self._log("Stopping benchmark after current run...", "warn")
            self._stop_btn.configure_btn(state="disabled")
            self._refresh_unload_buttons()   # AP-QL-111
        elif self.server_process:
            self._stop_server()

    def _refresh_action_btns(self):
        """Serverzustand auf beiden Aktionsknoepfen abbilden (AP-QL-25).

        Wird nach jedem Serverstart und -stopp aufgerufen. Stop ist nur
        aktiv und rot, solange es ueberhaupt etwas zu stoppen gibt; Run wird
        gruen, solange ein Server laeuft, damit der Zustand am Knopf selbst
        ablesbar ist und nicht nur am Statustext daneben.

        Benchmarklaeufe verwalten beide Knoepfe selbst (sie schalten auch Run
        ab), deshalb ruehrt diese Methode einen aktiven Benchmark nicht an.
        """
        if self._bench_thread and self._bench_thread.is_alive():
            return
        t = self.theme
        try:
            if self.server_process:
                self._run_btn.configure_btn(color=t.green)
                self._stop_btn.configure_btn(state="normal", color=t.red)
            else:
                self._run_btn.configure_btn(color=t.accent)
                self._stop_btn.configure_btn(state="disabled", color=t.accent)
        except (tk.TclError, AttributeError):
            pass      # Widgets bereits zerstoert (Theme-/Font-Neuaufbau)

    def _update_bars_for_kv(self, kv_name: str):
        """Redraw all VRAM bars and fit labels for the newly selected KV config."""
        t = self.theme
        for fn, card_data in self._model_cards.items():
            for rd in card_data["gpu_rows"]:
                size_gb  = rd["size_gb"]
                total_gb = rd["total_gb"]
                eff_gb   = kv_effective_gb(size_gb, kv_name)
                kv_fits  = eff_gb <= total_gb * 0.97
                # AP-QL-22 (AP-QL-19): Ruhezustand, nicht blind card_bg — sonst
                # wuerde jeder KV-Klick die gruene Toenung der laufenden Zeile
                # ueberzeichnen (der Balken ist ein Canvas und wird hier neu
                # gemalt).
                bg = (t.row_running_bg if self._is_running_row(fn, rd["key"])
                      else rd["card_bg"])

                rd["bar"].set_value(eff_gb, total_gb, canvas_bg=bg)
                rd["fit_text_lbl"].config(text=f"{eff_gb:.1f}/{total_gb:.0f} GB")
                rd["fit_icon_lbl"].config(
                    text=" ✓" if kv_fits else " ✗",
                    fg=t.green if kv_fits else t.red)
                rd["bar_tip"].update_text(
                    self._bar_tooltip_text(size_gb, eff_gb, total_gb))

    def _build_unload_buttons(self) -> None:
        """(Re)create one Unload button per detected GPU in the header.

        ⚙ Owner, 2026-09-17: "fuer jede GPU einen, aber nur die tatsaechliche
        Anzahl an GPUs an Unload Buttons anzeigen."

        Called after the GPU scan, not during header construction: `self.gpus`
        is still empty then, so building there yields zero buttons without any
        error. Idempotent -- an earlier set is destroyed first, so a rescan
        cannot leave doubles behind.

        With exactly one card the label carries no index: a number that
        distinguishes nothing only costs width, and the header has run past
        the window edge before (AP-QL-46).
        """
        import tkinter.font as _tkfont   # lokal, wie sonst in dieser Datei
        t = self.theme
        host = getattr(self, "_header_btn_group", None)
        if host is None:
            return
        for btn in (getattr(self, "_unload_btns", None) or {}).values():
            try:
                btn.destroy()
            except Exception:
                pass
        self._unload_btns = {}
        # ⚠ Nur Karten, auf denen ueberhaupt ein Server laufen kann. Auf
        # dieser Maschine meldet die Erkennung DREI Geraete -- RTX 5090,
        # RTX 4090 und eine Intel UHD 770 mit 0 MB. Der Launcher startet
        # ausschliesslich ueber CUDA_VISIBLE_DEVICES, also kann dort nie
        # etwas von uns liegen: ein "Unload 2" waere ein Knopf, der nichts
        # tun kann. Dieselbe Haltung wie AP-QL-105, das Knoepfe weglaesst,
        # die es nicht einloesen koennte.
        gpus = [g for g in (self.gpus or []) if g.vendor == "nvidia"]
        for g in gpus:
            label = "Unload" if len(gpus) == 1 else f"Unload {g.index}"
            # Gemessen, nicht geraten: HoverButton zeichnet auf fester
            # Canvas-Breite und kuerzt still -- AP-QL-105 hat daran drei
            # abgeschnittene Beschriftungen gekostet.
            breite = max(50, _tkfont.Font(font=FONT_BODY_B).measure(label) + 14)
            btn = HoverButton(host, t, text=label, color=t.accent,
                              width=breite, height=24,
                              canvas_bg=t.bg_header,
                              command=lambda i=g.index: self._unload_gpu(i))
            btn.pack(side="left", padx=1)
            btn.configure_btn(state="disabled")
            ToolTip(btn,
                    f"Free the VRAM this launcher holds on {g.name} "
                    f"(GPU {g.index}).\n"
                    "Stops only servers THIS launcher started there — a\n"
                    "foreign process on the same card is never touched.\n"
                    "Stop ends the primary server; this ends everything of\n"
                    "ours on one card, Brain and Vision alike.", t)
            self._unload_btns[g.index] = btn
        self._refresh_unload_buttons()

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
        # AP-QL-88: Rollenzuweisung rechts im Kopf — beide Rollen auf einen
        # Blick, ohne die Kartenbreite anzufassen (die ist tragend, siehe
        # CLAUDE.md zur Slot-Leiste). Rechts gepackt heisst: zuerst gepackt =
        # ganz aussen, deshalb Vision vor Brain, damit Brain links steht.
        # AP-QL-90: der Vision-Server hat genau eine Bedienung — an/aus.
        # Zuerst gepackt heisst ganz rechts, also steht er hinter den beiden
        # Rollenknoepfen: erst waehlen, dann starten, in Leserichtung.
        self._vision_button = HoverButton(
            hdr_frame, t, text="", width=170, height=24,
            command=self._toggle_vision_server)
        self._vision_button.pack(side="right", padx=(6, 0))
        ToolTip(self._vision_button,
                "Start or stop the model assigned to the Vision role.\n\n"
                "It runs BESIDE the main server on its own port and its own\n"
                "card — the main server keeps the conversation, this one\n"
                "answers about images. Port and card are configurable via\n"
                "vision_port / vision_gpu_key in the config.", t)
        for _role in (ROLE_VISION, ROLE_BRAIN):
            _btn = HoverButton(hdr_frame, t, text="", width=200, height=24,
                               command=lambda r=_role: self._open_role_menu(r))
            _btn.pack(side="right", padx=(6, 0))
            self._role_buttons[_role] = _btn
            ToolTip(_btn,
                    f"Pick the model that acts as the {ROLE_TITLES[_role]}.\n\n"
                    "The assignment is explicit and stored in the config, so\n"
                    "--autostart and external callers stop depending on\n"
                    "'whichever model was clicked last'.", t)
        self._models_path_label = tk.Label(hdr_frame, text="", font=FONT_DIM,
                                            bg=t.bg, fg=t.fg_dim)
        self._models_path_label.pack(side="left", padx=(8, 0))

        tk.Frame(container, height=1, bg=t.border).pack(fill="x", pady=(4, 0))

        scroll_frame = tk.Frame(container, bg=t.bg)
        scroll_frame.pack(fill="both", expand=True)

        self._canvas = tk.Canvas(scroll_frame, bg=t.bg, highlightthickness=0, bd=0)
        # ttk.Scrollbar picks up the themed style (see _configure_theme:
        # "Vertical.TScrollbar"), whereas tk.Scrollbar renders as the
        # legacy Motif widget on Linux — ignoring the dark theme and
        # sticking out as a grey 3D block. Windows also benefits: the
        # ttk widget integrates cleaner with the overall look.
        self._scrollbar = ttk.Scrollbar(scroll_frame, orient="vertical",
                                         command=self._canvas.yview,
                                         style="Vertical.TScrollbar")
        self._scrollbar.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)
        self._canvas.configure(yscrollcommand=self._scrollbar.set)

        self._inner_frame = tk.Frame(self._canvas, bg=t.bg)
        self._canvas_window = self._canvas.create_window((0, 0), window=self._inner_frame,
                                                          anchor="nw")
        self._inner_frame.bind("<Configure>",
                               lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>", self._on_canvas_resize)
        # Cross-platform mousewheel binding. Windows + macOS fire
        # <MouseWheel> with e.delta in ±120 increments. X11/Wayland on
        # Linux fire <Button-4> (up) and <Button-5> (down) instead —
        # <MouseWheel> never arrives. We bind all three and route to the
        # same yview_scroll call so the model list scrolls on every
        # platform.
        # AP-QL-77 (Delta-Audit 03.08., Fund E): Bereichspruefung wieder da.
        #
        # `bind_all` faengt das Rad FENSTERWEIT. Ohne die Pruefung scrollte
        # jedes Rad-Ereignis die Modellliste mit — wer im Server-Log
        # zurueckscrollte, verschob gleichzeitig die Karten darueber. Die
        # archivierte Windows-Linie wanderte dafuer die Widget-Kette hoch;
        # dieser Teil ging beim Zusammenlegen verloren, waehrend die
        # Linux-Erweiterung (Button-4/5) hinzukam. Beides gehoert zusammen.
        def _im_kartenbereich(widget) -> bool:
            """True, wenn das Widget unter dem Zeiger zu unserem Canvas gehoert."""
            w = widget
            while w is not None:
                if w is self._canvas:
                    return True
                try:
                    w = w.master
                except AttributeError:
                    return False
            return False

        def _on_mousewheel(event):
            if not _im_kartenbereich(getattr(event, "widget", None)):
                return
            # Windows/macOS path: delta is ±120 per notch.
            if event.delta:
                step = -1 if event.delta > 0 else 1
            # Linux path: num is 4 (up) or 5 (down), delta is 0.
            elif getattr(event, "num", 0) == 4:
                step = -1
            elif getattr(event, "num", 0) == 5:
                step = 1
            else:
                return
            self._canvas.yview_scroll(step, "units")
        self._canvas.bind_all("<MouseWheel>", _on_mousewheel)   # Win + macOS
        self._canvas.bind_all("<Button-4>",   _on_mousewheel)   # Linux X11 up
        self._canvas.bind_all("<Button-5>",   _on_mousewheel)   # Linux X11 down

    def _on_canvas_resize(self, event):
        self._canvas.itemconfig(self._canvas_window, width=event.width)
        # AP-QL-51: passt bei dieser Breite eine andere Spaltenzahl, werden
        # die Karten neu angeordnet. Nur beim WECHSEL — <Configure> feuert
        # waehrend eines Maus-Resize im Millisekundentakt.
        # AP-QL-52 (Fable-Kontrollrunde): die Signatur ist (cols, capped),
        # nicht cols allein. `capped` haengt stetig von der Breite ab — wer
        # nur auf den Spaltenwechsel triggert, laesst beim Ueberschreiten
        # der Deckelgrenze die Karten ueber die Hoechstbreite hinauswachsen
        # bzw. fest geclippt stehen, obwohl die Spaltenzahl gleich bleibt.
        self._model_area_last_w = event.width
        cols = self._model_cols_for(event.width)
        min_w = self._model_card_min_width()
        capped = event.width > 1 and (event.width / cols) > (
            min_w * MODEL_CARD_MAX_FACTOR)
        sig = (cols, capped)
        if sig != getattr(self, "_model_layout_sig", None):
            self._model_cols = cols
            # AP-QL-124: inventory-only libraries also need responsive columns.
            self._model_layout_sig = sig
            if (getattr(self, "_model_cards", None)
                    or getattr(self, "_inventory_cards", None)):
                self._relayout_model_cards()

    def _relayout_model_cards(self) -> None:
        """Karten neu anordnen, ohne Auswahl und Tönung zu verlieren.

        _rebuild_model_cards() ist auf den Rescan zugeschnitten: es setzt die
        Auswahl zurueck und waehlt am Ende Karte 0. Beim blossen Aendern der
        Fensterbreite waere das ein Ruecksprung mitten in die Bedienung —
        und die Zeilentoenung des laufenden Modells (AP-QL-19/22) waere weg.
        """
        sel_m, sel_r = self._sel_model, self._sel_row
        name = (self.models[sel_m].filename
                if 0 <= sel_m < len(self.models) else None)
        self._rebuild_model_cards()
        if name:
            for i, mdl in enumerate(self.models):
                if mdl.filename != name:
                    continue
                # after_idle ist FIFO — diese Wiederherstellung laeuft nach
                # dem _select_cell(0, 0), das _rebuild_model_cards plant.
                self.after_idle(
                    lambda i=i, r=sel_r: self._select_cell(
                        i, min(r, max(0, self._rows_of(i) - 1))))
                break
        if hasattr(self, "_update_running_indicator"):
            self.after_idle(self._update_running_indicator)

    def _model_card_min_width(self) -> int:
        """Mindestbreite einer Modellkarte in Pixeln (AP-QL-51).

        Bemessen am laengsten vorhandenen Modellnamen plus Groessenangabe,
        gemessen an der TATSAECHLICH gerenderten Schrift — nicht an einer
        Zeichenzahl. Nur so wandert die Breite mit dem Schriftzoom mit
        (Ctrl+Plus/Minus) und stimmt auch dann, wenn eine andere Mono-Familie
        aufgeloest wurde als erwartet.

        Returns:
            Mindestbreite in Pixeln; ein konservativer Rueckfallwert, wenn
            noch keine Modelle bekannt sind oder Tk nicht messen kann.
        """
        try:
            import tkinter.font as tkfont
            f_name = tkfont.Font(font=FONT_BODY_B)
            f_size = tkfont.Font(font=FONT_BODY)
            names = [m.filename for m in (self.models or [])]
            if not names:
                return 360
            widest = max(f_name.measure(n) for n in names)
            # Die Groessenangabe steht rechts daneben ("20.8 GB").
            widest += f_size.measure("999.9 GB")
            return int(widest + MODEL_CARD_PAD_PX)
        except Exception:
            return 360

    def _model_area_width(self) -> int:
        """Breite des Kartenbereichs in Pixeln, 0 wenn noch nicht gelayoutet.

        AP-QL-52: Die letzte <Configure>-Breite gewinnt gegen winfo_width().
        Der Rebuild laeuft IM Configure-Handler — winfo_width() kann dort
        noch den alten Stand tragen, und dann rechnet die capped-Entscheidung
        mit einer Breite, die gleich nicht mehr gilt.
        """
        hint = getattr(self, "_model_area_last_w", 0)
        if hint > 1:
            return int(hint)
        try:
            return int(self._canvas.winfo_width())
        except (AttributeError, tk.TclError):
            return 0

    def _current_model_cols(self) -> int:
        """Die aktuell gueltige Spaltenzahl (AP-QL-51).

        Einziger Zugriffspunkt — sowohl der Kartenaufbau als auch der
        Tastaturlauf fragen hier. Liefe die Navigation weiter gegen die
        Konstante, spraenge der Fokus nach einem Spaltenwechsel in die
        falsche Karte (das war der Fehler, den AP-QL-30 behoben hat).
        """
        cols = getattr(self, "_model_cols", None)
        if not cols:
            cols = self._model_cols_for(self._model_area_width())
            self._model_cols = cols
        return max(1, int(cols))

    def _model_cols_for(self, avail: int) -> int:
        """Wie viele Kartenspalten bei dieser Breite (AP-QL-51).

        Args:
            avail: Verfuegbare Breite des Kartenbereichs in Pixeln.

        Returns:
            Spaltenzahl >= 1, so gewaehlt, dass keine Spalte schmaler als die
            Mindestbreite und — soweit der Deckel es zulaesst — keine breiter
            als das Doppelte davon wird.
        """
        if avail <= 1:
            # `or` statt getattr-Vorgabe: _model_cols kann ausdruecklich auf
            # None stehen ("noch nicht berechnet"), und dann liefert getattr
            # nicht die Vorgabe, sondern None — der Aufrufer bekaeme keinen
            # gueltigen Wert zurueck.
            return getattr(self, "_model_cols", None) or MODEL_CARD_COLS
        min_w = self._model_card_min_width()
        max_w = min_w * MODEL_CARD_MAX_FACTOR
        # So viele passen hoechstens hinein, ohne unter die Mindestbreite zu
        # fallen …
        by_min = max(1, avail // min_w)
        # … und so viele braucht es mindestens, damit keine breiter als die
        # Hoechstbreite wird (Aufrunden).
        by_max = max(1, -(-avail // max_w))
        return max(1, min(by_min, MODEL_CARD_COLS_MAX, max(by_max, 1)))

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

    def _measure_label_group_width(self, labels, size: int,
                                   bold: bool = False,
                                   pad_chars: float = 2.0) -> int:
        """Einheitliche Breite fuer eine Gruppe von Knopfbeschriftungen.

        Bemessen am breitesten Label der Gruppe ueber ``tkfont.measure()``,
        also an der wirklich gerenderten Schrift statt an einer geschaetzten
        Zeichenzahl.

        AP-QL-59 — ``size`` ist PFLICHT, und das ist der eigentliche Inhalt
        dieses Pakets. Vorher stand die Groesse hier fest auf 10 verdrahtet,
        also der BASISgroesse aus ``_FONT_BASE``. Das stimmt bei Zoomstufe
        100 % und sonst nie: bei 250 % zeichnet Tk in 25 px, gemessen wurde
        weiter in 10 px. Der Fehler steckte damit in JEDEM Aufrufer, der ihn
        nicht bemerkt hat — gemessen am 02.08.2026 in allen vieren
        (Fusszeile 280 px zu schmal bei 250 %, Slot-Leiste noch mehr, KV und
        LA bereits ab 100 %).

        Ein Vorgabewert waere hier genau die Falle, in die vier Aufrufer
        nacheinander getappt sind. Ohne Vorgabe MUSS jeder Aufrufer sagen,
        in welcher Groesse sein Knopf zeichnet — und wer das tut, kann es
        nicht mehr falsch machen.

        Args:
            labels: Die Beschriftungen der Gruppe.
            size: Die Schriftgroesse, in der der Knopf tatsaechlich
                zeichnet — ``FONT_SMALL[1]`` bzw. ``FONT_SMALL_B[1]``,
                NICHT der Basiswert aus ``_FONT_BASE``.
            bold: Fett messen — muss zur Schrift des Knopfes passen.
            pad_chars: Waagerechter Innenrand, gemessen in Zeichenbreiten
                statt in festen Pixeln, damit er mit der Schrift mitwaechst.
                2.0 (Vorgabe) ist ein Zeichen Luft je Seite und ergibt bei
                100 % exakt die 20 px, mit denen die Fusszeile immer schon
                gebaut wurde; 1.0 ist die engere Variante der KV/LA-Reihen.

        Returns:
            Die Breite in Pixeln fuer jeden Knopf der Gruppe.
        """
        import tkinter.font as tkfont
        f = tkfont.Font(family=_MONO, size=size,
                        weight="bold" if bold else "normal")
        if not labels:
            return 80
        return max(f.measure(l) for l in labels) + round(pad_chars * f.measure("0"))

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
        # Previous hard-coded widths (130/160, 80/100, 168/200) are gone —
        # tkfont.measure() automatically handles Linux's wider mono fonts
        # (DejaVu Sans Mono etc.) and Tk's HiDPI scaling factor.
        _BTN_H = 28

        # AP-QL-44: EINE Breite fuer alle Knoepfe dieser Zeile, bemessen am
        # laengsten Text (Owner-Wunsch). Vorher hatte jede Gruppe ihre eigene
        # Breite — "Save Results..." war deutlich breiter als "Paths", und die
        # Zeile wirkte unruhig. Ein gemeinsames Mass macht sie zu einer Reihe.
        FOOTER_LABELS = ["Rescan Models", "Update Binaries", "About", "Paths",
                         "Autoload", "💾  Save Results..."]
        # AP-QL-59: gemessen in der GERENDERTEN Groesse. Vorher wurde
        # hier stets in 10 px gemessen — bei 250 % fehlten der Zeile
        # dadurch 280 px Breite. pad_chars=2.0 ist ein Zeichen Luft je
        # Seite und ergibt bei 100 % exakt die 20 px von vorher, die
        # Zeile sieht dort also unveraendert aus.
        footer_w = self._measure_label_group_width(
            FOOTER_LABELS, size=FONT_SMALL[1], bold=False)
        group_a_w = footer_w
        HoverButton(bar, t, text="Rescan Models", color=t.accent_soft,
                    width=group_a_w, height=_BTN_H, font=FONT_SMALL,
                    command=self._rescan_models).pack(side="left", padx=2)
        HoverButton(bar, t, text="Update Binaries", color=t.accent_soft,
                    width=group_a_w, height=_BTN_H, font=FONT_SMALL,
                    command=self._update_binaries).pack(side="left", padx=2)

        # Group B: engine slot buttons (middle, dynamic).
        # Sized and spacing handled in _refresh_slot_buttons. Extra left
        # margin separates the slot group from Update Binaries.
        # AP-QL-43: die Engine-Slots liegen als EIGENE Zeile UEBER den
        # uebrigen Knoepfen — Owner-Wunsch. Sie sind der Schalter, den man im
        # Betrieb am haeufigsten braucht, und sollen nicht mit Rescan/Paths/
        # About um denselben Platz konkurrieren.
        #
        # Kind von `self`, nicht mehr von `bar`: nur so laesst sich mit
        # side="bottom" eine eigene Zeile darueber erzeugen. Die Reihenfolge
        # ist dabei entscheidend — bei side="bottom" landet das ZUERST
        # gepackte ganz unten. `bar` ist oben schon gepackt, diese Zeile
        # kommt also darueber zu liegen.
        self._slot_bar = tk.Frame(self, bg=t.bg)
        self._slot_bar.pack(side="bottom", fill="x", padx=16, pady=(0, 4))
        self._footer_bar = bar      # AP-QL-42: Fluss folgt am Ende
        self._slot_buttons: list = []

        # Group C: settings (right, packed right-to-left so About lands
        # at the far right). Save Results is standalone with its own width.
        group_c_w = footer_w      # AP-QL-44
        HoverButton(bar, t, text="About", color=t.accent_soft,
                    width=group_c_w, height=_BTN_H, font=FONT_SMALL,
                    command=self._show_about).pack(side="right", padx=2)
        HoverButton(bar, t, text="Paths", color=t.accent_soft,
                    width=group_c_w, height=_BTN_H, font=FONT_SMALL,
                    command=self._show_paths_dialog).pack(side="right", padx=2)

        # v0.56 — Autoload toggle. Sits just left of Paths in the footer.
        # Three visual states, all rendered with the same label "Autoload"
        # so the button's width stays constant across transitions — state
        # is encoded in color + corner_glyph only, never in text length:
        #   locked  = install_verified==False  -> grey, disabled, "🔒" glyph
        #   OFF     = verified, autoload==False -> theme.accent_soft, no glyph
        #   ON      = autoload==True            -> green, "✓" glyph
        # _refresh_autoload_btn() applies the initial state and is also
        # called whenever install_verified flips (first "listening") or
        # the user clicks the button.
        al_w = footer_w           # AP-QL-44
        self._autoload_btn = HoverButton(
            bar, t, text="Autoload", color=t.border,
            width=al_w, height=_BTN_H, font=FONT_SMALL,
            command=self._on_autoload_click)
        self._autoload_btn.pack(side="right", padx=2)
        self._autoload_tooltip = ToolTip(self._autoload_btn, "", t)
        self._refresh_autoload_btn()

        save_w = footer_w         # AP-QL-44
        # AP-QL-44: gleiche Farbe wie die uebrigen Knoepfe der Zeile. Der
        # graue t.border war doppelt gemoppelt — der Knopf ist ohnehin
        # deaktiviert, solange es nichts zu speichern gibt, und zeichnet
        # dann bg_secondary. Aktiv ist er jetzt blau wie die anderen.
        self._save_bench_btn = HoverButton(bar, t, text="💾  Save Results...",
                                            color=t.accent_soft, width=save_w,
                                            height=_BTN_H, font=FONT_SMALL,
                                            command=self._save_pending_bench)
        self._save_bench_btn.configure_btn(state="disabled")
        self._save_bench_btn.pack(side="right", padx=(2, 12))

        # AP-QL-42: die Fussleiste bricht ebenfalls um. Im Screenshot des
        # Owners war der fuenfte Engine-Slot rechts abgeschnitten und
        # Save Results / Autoload / Paths / About gar nicht mehr sichtbar —
        # 318 px Dateiknoepfe + 991 px Slot-Leiste + ~450 px rechte Gruppe
        # passen in kein vernuenftiges Fenster. Jetzt rutscht die rechte
        # Gruppe in eine zweite Zeile, statt zu verschwinden.
        self._install_footer_flow(bar)

    # ─── Engine Quick-Switch Slot Buttons ────────────────────────────

    def _refresh_slot_buttons(self):
        """Rebuild the quick-switch slot buttons from self.cfg['server_slots'].

        Called after config changes (Paths dialog save) and after a slot click
        (to update the active highlight). Only non-empty slots get a button.
        The button whose path matches the currently active llama_server_path
        is highlighted with the accent color; all others use theme.accent_soft.
        """
        t = self.theme
        # Tear down existing buttons
        for btn in getattr(self, "_slot_buttons", []):
            try:
                btn.destroy()
            except Exception:
                pass
        self._slot_buttons = []

        # AP-QL-93: render from a de-duplicated view. The stored list can
        # carry duplicates from an older version, a hand edit, or the other
        # platform's config file — the footer must still show one button
        # per build, or a click lights two of them at once.
        slots, _ = dedupe_slot_paths(self.cfg.get("server_slots") or [])
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
        # AP-QL-59: wie die Fusszeile — und hier fiel es am staerksten
        # aus, weil die Slot-Namen die laengsten Beschriftungen der App
        # sind ("spiritbuun_0eb1e82b").
        btn_w = self._measure_label_group_width(
            slot_labels, size=FONT_SMALL[1],
            bold=False) if slot_labels else 90

        for slot_path in slots:
            if not slot_path or not slot_path.strip():
                continue
            label = slot_label_from_path(slot_path)
            if not label:
                continue

            # Highlight active slot
            slot_norm = os.path.normcase(os.path.normpath(slot_path))
            is_active = (slot_norm == active_norm) and bool(active_norm)
            # Aktiver Slot gruen — dieselbe Farbe wie der Autoload-Button
            # im ON-Zustand. "Gruen = aktiv" ist damit ueberall dieselbe
            # Aussage (Autoload, Auswahlrahmen, aktive Engine).
            color = t.green if is_active else t.accent_soft

            # Missing-file indicator: still show the button, but in muted color
            # so the user knows the slot is broken. Click will log an error.
            if not os.path.isfile(slot_path):
                color = t.border  # muted/disabled look

            btn = HoverButton(self._slot_bar, t, text=label, color=color,
                              width=btn_w, height=_BTN_H, font=FONT_SMALL,
                              command=lambda p=slot_path: self._on_slot_click(p))
            # AP-QL-43: padx=6 erreicht die Umbruchschwelle, jeder Slot ist
            # damit ein eigener Gruppenanfang und darf umbrechen.
            btn.pack(side="left", padx=6)
            ToolTip(btn, slot_path, theme=t)
            self._slot_buttons.append(btn)

        # AP-QL-43: Die Knoepfe entstehen bei JEDEM Wechsel neu, der Fluss
        # muss also neu eingehaengt werden — sonst zeigte _flow_rows auf
        # zerstoerte Widgets und die neue Leiste braeche nie um.
        self._reinstall_slot_flow()

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

    # ─── AP-QL-88: Model roles ────────────────────────────────────────

    def _model_by_filename(self, filename: str):
        """Resolve a stored filename against the current scan, or None.

        A role survives in the config as a bare filename; the library it
        refers to may have moved, shrunk, or be on another machine. Returning
        None for a name that is no longer there is deliberate: a stale
        assignment then behaves like an unassigned one instead of pointing at
        a path that does not exist.
        """
        name = (filename or "").strip()
        if not name:
            return None
        return next((m for m in self.models if m.filename == name), None)

    def _role_button_text(self, role: str) -> str:
        """Label for one role button: glyph, title, and the short model name.

        The name is shortened rather than wrapped because the button sits in
        a header row whose width the model grid does not control. The full
        name stays reachable through the tooltip.
        """
        name = (self.cfg.get(role, "") or "").strip()
        if not name:
            shown = "\u2014"
        else:
            shown = name[:-5] if name.lower().endswith(".gguf") else name
            if len(shown) > 20:
                shown = shown[:19] + "\u2026"
        return f"{ROLE_GLYPHS[role]} {ROLE_TITLES[role]}: {shown}"

    def _refresh_role_widgets(self) -> None:
        """Re-label the role buttons and the badges on the model cards."""
        # AP-QL-122: measure the displayed labels at the current font scale.
        # Fixed 200px canvases clipped even the deliberately shortened names.
        labels = {role: self._role_button_text(role)
                  for role in self._role_buttons}
        width = self._measure_label_group_width(
            list(labels.values()), size=FONT_SMALL_B[1], bold=True)
        for role, btn in self._role_buttons.items():
            assigned = (self.cfg.get(role, "") or "").strip()
            btn.configure_btn(
                text=labels[role], width=width,
                color=self.theme.accent if assigned else self.theme.button_bg)
        for fn, card in self._model_cards.items():
            badge = card.get("role_badge")
            if badge is None:
                continue
            glyphs = "".join(ROLE_GLYPHS[r] for r in ROLE_KEYS
                             if (self.cfg.get(r, "") or "").strip() == fn)
            badge.config(text=(glyphs + " ") if glyphs else "")

    def _open_role_menu(self, role: str) -> None:
        """Offer every scanned model for one role, plus 'none'.

        A menu instead of a grid of buttons: the library is 19 models here and
        can grow, while a role holds exactly one of them. tk.Menu is already
        themed application-wide (see _apply_theme), so this inherits the look
        without a second styling path.
        """
        current = (self.cfg.get(role, "") or "").strip()
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(
            label=("\u2022 " if not current else "   ") + ROLE_NONE_LABEL,
            command=lambda: self._assign_role(role, ""))
        if self.models:
            menu.add_separator()
        for model in self.models:
            mark = "\u2022 " if model.filename == current else "   "
            menu.add_command(
                label=mark + model.filename,
                command=lambda fn=model.filename: self._assign_role(role, fn))
        btn = self._role_buttons.get(role)
        if btn is None:
            return
        try:
            menu.tk_popup(btn.winfo_rootx(),
                          btn.winfo_rooty() + btn.winfo_height())
        finally:
            menu.grab_release()

    def _assign_role(self, role: str, filename: str) -> None:
        """Store one role assignment and say what it means.

        The Vision check is a WARNING, never a refusal. Whether a build can
        see is decided by llama-server and the projector next to the file —
        the launcher only reports what it can observe. Refusing here would be
        the same class of mistake AP-QL-87 removed: a guess overruling the
        user.
        """
        if (self.cfg.get(role, "") or "") == filename:
            return
        self.cfg[role] = filename
        save_config(self.cfg)
        self._refresh_role_widgets()
        self._refresh_vision_widgets()
        title = ROLE_TITLES[role]
        if not filename:
            self._log(f"{title} role cleared.", "info")
            return
        self._log(f"{title} role: {filename}", "info")
        model = next((m for m in self.models if m.filename == filename), None)
        if model is None:
            return
        if role == ROLE_VISION and not find_mmproj(model.path):
            self._log(
                f"  Warning: no mmproj*.gguf next to {filename} — this build "
                f"will start text-only and cannot process images.", "warn")
        if role == ROLE_BRAIN and not model.is_reasoning:
            self._log(
                f"  Note: {filename} is not detected as a thinking model.",
                "info")

    def _refresh_models_header(self) -> None:
        """Keep runnable GGUF counts distinct from external model inventory."""
        extra = len(getattr(self, "installed_models", []))
        suffix = f"; {extra} other" if extra else ""
        self._models_header.config(
            text=f"Models ({len(self.models)} GGUF, "
                 f"{sum(m.size_gb for m in self.models):.0f} GB{suffix})")

    def _open_inventory_folder(self, path: str) -> None:
        """Open a model's existing canonical directory without launching it."""
        folder = os.path.realpath(path)
        if not os.path.isdir(folder):
            self._log(f"Model folder no longer exists: {folder}", "warn")
            return
        try:
            if IS_WINDOWS:
                os.startfile(folder)
            else:
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.Popen([opener, folder])
        except OSError as exc:
            self._log(f"Cannot open model folder: {exc}", "error")

    def _build_installed_model_cards(self, columns: int) -> int:
        """Add compact inventory columns above the runnable GGUF cards.

        Returns the number of grid rows reserved by this group. Entries never
        become model/GPU rows, role candidates, or server/benchmark targets.
        """
        self._inventory_cards = {}
        entries = getattr(self, "installed_models", [])
        if not entries:
            return 0
        t = self.theme
        group = tk.Frame(self._inner_frame, bg=t.bg)
        group.grid(row=0, column=0, columnspan=columns, sticky="ew", pady=(0, 8))
        tk.Label(group, text=f"Other models ({len(entries)}) · non-GGUF",
                 font=FONT_BODY_B, bg=t.bg, fg=t.fg_secondary,
                 anchor="w").grid(row=0, column=0, columnspan=columns,
                                  sticky="ew", padx=10, pady=(4, 2))
        for column in range(columns):
            group.columnconfigure(column, weight=1, uniform="inventorycol")
        button_width = self._measure_label_group_width(
            ["Open folder"], size=FONT_SMALL_B[1], bold=True)
        for index, entry in enumerate(entries):
            row, column = divmod(index, columns)
            card = tk.Frame(group, bg=t.bg_secondary)
            card.grid(row=row + 1, column=column, sticky="nsew", padx=2, pady=1)
            open_button = HoverButton(
                card, t, text="Open folder", width=button_width, height=26,
                command=lambda path=entry.path: self._open_inventory_folder(path))
            open_button.pack(side="right", anchor="n", padx=8, pady=6)
            details = tk.Frame(card, bg=t.bg_secondary)
            details.pack(side="left", fill="x", expand=True, padx=8, pady=4)
            summary = (f"{entry.kind} · {entry.format_name} · "
                       f"{entry.size_bytes / (1024 ** 3):.1f} GB")
            texts = [(entry.name, FONT_BODY_B, t.fg),
                     (summary, FONT_BODY, t.fg_secondary)]
            if entry.status != "Files present":
                texts.append((entry.status, FONT_BODY, t.red))
            labels = []
            for text, font, color in texts:
                label = tk.Label(details, text=text, font=font, fg=color,
                                 bg=t.bg_secondary, anchor="w", justify="left",
                                 wraplength=1)
                label.pack(fill="x", anchor="w")
                labels.append(label)
            # Wrapping adapts to the existing canvas instead of making the
            # GGUF grid wider when an inventory path or model name is long.
            def resize(event: tk.Event, widgets: tuple = tuple(labels)) -> None:
                for label in widgets:
                    label.configure(wraplength=max(1, event.width - 4))
            details.bind("<Configure>", resize)
            # AP-QL-124: keep detail on hover instead of repeating it in every
            # card. Diagnostic states remain visible without hovering.
            tooltip = (f"{entry.name}\n{summary}\n{entry.status}\n"
                       f"Runtime: {entry.runtime}\n{entry.path}")
            tooltips = [ToolTip(widget, tooltip, t)
                        for widget in (card, details, *labels, open_button)]
            self._inventory_cards[entry.path] = {
                "card": card, "labels": labels, "open_button": open_button,
                "tooltips": tooltips}
        return 1

    def _rebuild_model_cards(self):
        t = self.theme
        HOVER_BG = "#252830" if self.is_dark else "#e8f0fe"
        SEL_BORDER = t.accent

        for w in self._inner_frame.winfo_children():
            w.destroy()
        self._model_cards.clear()
        self._sel_model = -1
        self._sel_row = 0
        inventory_rows = self._build_installed_model_cards(self._current_model_cols())

        paths = [p for p in (self.cfg.get("llm_models_paths") or []) if p]
        label_text = f"({' • '.join(paths)})" if paths else "(no directory configured)"
        self._models_path_label.config(text=label_text)

        # AP-QL-24 (v0.70): die Karten liegen jetzt im grid (zwei Spalten)
        # statt im pack. tkinter verweigert die Mischung beider Manager in
        # einem Elternwidget, also muss auch der Leerzustand gridden.
        if not self.models:
            for column in range(MODEL_CARD_COLS_MAX + 1):
                self._inner_frame.columnconfigure(
                    column, weight=1 if column == 0 else 0, minsize=0, uniform="")
            tk.Label(self._inner_frame, text="No GGUF models found.",
                     font=FONT_BODY, bg=t.bg, fg=t.fg_dim).grid(
                         row=inventory_rows, column=0,
                         columnspan=self._current_model_cols(), pady=20)
            self._refresh_role_widgets()
            self._refresh_vision_widgets()
            return

        # AP-QL-24 (v0.70): mehrere Spalten, damit eine volle Modellliste ohne
        # Scrollen passt. uniform= haelt alle Spalten exakt gleich breit,
        # egal wie lang der laengste Dateiname ist — ohne das wuerde die
        # Spalte mit den breiteren Karten mehr Platz beanspruchen.
        #
        # AP-QL-51: die Spaltenzahl kommt aus der Breite, nicht aus einer
        # Konstante. Und wenn der Deckel MODEL_CARD_COLS_MAX greift, wuerden
        # die Karten ueber die Hoechstbreite hinaus gedehnt — dann bekommen
        # sie eine feste Breite und eine mitwachsende Fuellspalte rechts
        # schluckt den Rest, statt Leerraum in die Karten zu legen.
        cols = self._current_model_cols()
        avail = self._model_area_width()
        max_w = self._model_card_min_width() * MODEL_CARD_MAX_FACTOR
        capped = avail > 1 and (avail / cols) > max_w
        # AP-QL-52 (Fable-Kontrollrunde): ERST alle je erreichbaren Spalten
        # zuruecksetzen. columnconfigure ist persistent am wiederverwendeten
        # _inner_frame — nur die Kinder werden zerstoert. Ohne den Reset
        # behielt Spalte 3 nach einem 4-Spalten-Layout ihr weight/uniform
        # (Karten zu schmal, rechtes Drittel leer), und ein einmal gesetztes
        # minsize=max_w klebte im uncapped-Zweig fuer immer (Karten 841 px
        # bei 1424 px Fenster — rechte Kante 264 px abgeschnitten).
        for _c in range(MODEL_CARD_COLS_MAX + 1):
            self._inner_frame.columnconfigure(_c, weight=0, minsize=0,
                                              uniform="")
        for _c in range(cols):
            if capped:
                self._inner_frame.columnconfigure(
                    _c, weight=0, minsize=max_w, uniform="modelcol")
            else:
                self._inner_frame.columnconfigure(_c, weight=1,
                                                  uniform="modelcol")
        # Fuellspalte: nur wenn gedeckelt, sonst wuerde sie den Karten Platz
        # wegnehmen. Sie traegt nichts und hat kein uniform.
        self._inner_frame.columnconfigure(
            cols, weight=1 if capped else 0, minsize=0)
        # Signatur nachziehen — _on_canvas_resize vergleicht dagegen.
        self._model_layout_sig = (cols, capped)

        for i, model in enumerate(self.models):
            grid_row, grid_col = divmod(i, cols)
            # Streifung nach GRID-ZEILE, nicht nach Listenindex: ein
            # Wechsel auf i ergaebe bei mehr als einer Spalte ein Schachbrett.
            bg = t.bg_secondary if grid_row % 2 == 0 else t.bg
            card = tk.Frame(self._inner_frame, bg=bg)
            # AP-QL-51: der Abstand sitzt zwischen den Spalten, nicht an den
            # Aussenkanten — bei mehr als zwei Spalten braucht die mittlere
            # ihn auf beiden Seiten.
            if cols == 1:
                pad = (0, 0)
            elif grid_col == 0:
                pad = (0, 3)
            elif grid_col == cols - 1:
                pad = (3, 0)
            else:
                pad = (3, 3)
            card.grid(row=grid_row + inventory_rows, column=grid_col, sticky="nsew",
                      padx=pad, pady=1)

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
            # AP-QL-88: Rollenabzeichen. Immer angelegt, meist leer — so
            # kann _refresh_role_widgets() es ohne Neuaufbau der Karten
            # umschreiben, und ein Rollenwechsel kostet keine Auswahl.
            role_badge = tk.Label(hdr, text="", font=FONT_BODY_B,
                                  bg=bg, fg=t.accent)
            role_badge.pack(side="left")
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
                    self._gpu_source_color(gpu.vendor),
                    model.size_gb, vram_gb, fits)
                gpu_rows.append(row_data)

            if self._cpu_ram_gb > 0:
                cpu_fits = model.size_gb <= self._cpu_ram_gb * 0.7
                row_data = self._create_gpu_row(
                    card, bg, HOVER_BG, model, i, len(gpu_rows), "CPU",
                    "CPU RAM", self._gpu_source_color("cpu"),
                    model.size_gb, self._cpu_ram_gb, cpu_fits)
                gpu_rows.append(row_data)

            tk.Frame(card, bg=bg, height=4).pack(fill="x")

            self._model_cards[model.filename] = {
                "card": card, "bg": bg, "index": i,
                "gpu_rows": gpu_rows,
                "role_badge": role_badge,
            }

        # AP-QL-88: Karten sind neu, also Abzeichen und Knopfbeschriftungen
        # nachziehen. Steht hier und nicht im Aufrufer, damit KEIN Pfad die
        # Karten neu bauen kann, ohne dass die Rollen mitkommen.
        self._refresh_role_widgets()
        self._refresh_vision_widgets()

        if self.models and not inventory_rows:
            # Defer the initial selection until after the first idle pass
            # so the geometry manager has a chance to lay out _inner_frame
            # and the <Configure> binding has set the canvas scrollregion.
            # Calling _select_cell(0, 0) inline here used to leave a large
            # empty band above the first card on first display because
            # _inner_frame.winfo_height() was still 1 at that moment.
            self.after_idle(lambda: self._select_cell(0, 0))
        # With inventory above the grid, the first GGUF may be off-screen.
        # Wait for an explicit click/Down key instead of letting Return start
        # an invisible default selection. Restored selections stay unchanged.

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

        # AP-QL-78 (Fund F): der Balken folgt jetzt der Schriftskala. Vorher
        # standen 200x18 px fest, waehrend FONT_BRAILLE bis 250 % waechst —
        # ab etwa 140 % wurden die Braille-Zeichen abgeschnitten. Die alte
        # Windows-Linie hatte dafuer scaled_px(); diese Linie misst sonst
        # ueber tkfont, hier genuegt der Skalenfaktor selbst.
        _bar_scale = max(1.0, _FONT_SCALE_CURRENT)
        bar = BrailleBar(row, t, width=int(200 * _bar_scale),
                         height=int(18 * _bar_scale), chars=16,
                         canvas_bg=card_bg)
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
            # AP-QL-22 (AP-QL-19): auf den RUHEZUSTAND zuruecksetzen, nicht
            # blind auf card_bg. Sonst loescht ein Hover-drueber-und-weg die
            # gruene Toenung der laufenden Zeile — _enter hatte den Guard
            # schon, _leave nicht.
            resting = (t.row_running_bg
                       if self._is_running_row(model.filename, gpu_key)
                       else card_bg)
            for child in row.winfo_children():
                try: child.configure(bg=resting)
                except: pass
            row.configure(bg=resting)
            cur_eff = kv_effective_gb(size_gb, self._kv_var.get())
            bar.set_value(cur_eff, total_gb, canvas_bg=resting)

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
        # AP-QL-22 (AP-QL-19): Auswahlrahmen gruen (war t.accent/blau) —
        # dieselbe Farbe wie der Autoload-Button. Auswahl bleibt trotzdem von
        # der laufenden Zeile unterscheidbar: die Auswahl ist ein RAHMEN, das
        # laufende Modell eine getoente FLAECHE.
        rd["frame"].config(highlightbackground=t.green)

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

    def _selected_gpu_key(self) -> str:
        """Return the key ("GPU n" / "CPU") of the highlighted GPU row.

        Falls back to "GPU 0" when no model / row is selected yet.
        """
        if 0 <= self._sel_model < len(self.models):
            card_data = self._model_cards.get(
                self.models[self._sel_model].filename)
            if card_data:
                rows = card_data.get("gpu_rows") or []
                if 0 <= self._sel_row < len(rows):
                    return rows[self._sel_row]["key"]
        return "GPU 0"

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

    # ─── AP-QL-30 (v0.73): Tastaturlauf folgt dem KARTEN-RASTER ──────────
    #
    # Bis AP-QL-24 lagen die Karten einspaltig, da waren "naechste in der
    # Liste" und "die Karte darunter" dasselbe, und Hoch/Runter liefen die
    # Liste linear ab (Links/Rechts stiegen durch die GPU-Zeilen, was
    # Hoch/Runter ohnehin schon taten). Mit MODEL_CARD_COLS = 2 fallen die
    # beiden Bedeutungen auseinander: das naechste Modell sitzt RECHTS,
    # nicht darunter. Deshalb jetzt:
    #
    #   Hoch/Runter   laufen die GPU-Zeilen einer Karte ab und wechseln dann
    #                 in die Karte darueber/darunter IN DERSELBEN SPALTE
    #                 (Index -/+ COLS).
    #   Links/Rechts  springen zur Nachbarkarte derselben Rasterzeile
    #                 (Index -/+ 1) und behalten die GPU-Zeile, soweit
    #                 moeglich.
    #
    # Jeder Schritt ist reine Indexrechnung — die Rasterposition von Modell
    # i ist divmod(i, _current_model_cols()), es muss also nichts bei tkinter
    # nach Geometrie gefragt werden.

    def _rows_of(self, model_idx: int) -> int:
        """Anzahl der GPU-Zeilen einer Karte, oder 0 wenn unbekannt.

        Args:
            model_idx: Index in self.models.

        Returns:
            Zeilenzahl der Karte, 0 bei ungueltigem Index.
        """
        if model_idx < 0 or model_idx >= len(self.models):
            return 0
        card = self._model_cards.get(self.models[model_idx].filename)
        return len(card["gpu_rows"]) if card else 0

    def _goto_card(self, target_idx: int, row_idx: int) -> None:
        """row_idx der Karte target_idx auswaehlen, auf deren Zeilenzahl begrenzt.

        Karten koennen sich in der Zeilenzahl unterscheiden (eine GPU, die
        zwischen zwei Scans verschwunden ist), deshalb wird begrenzt statt
        angenommen.

        Args:
            target_idx: Index der Zielkarte.
            row_idx: Gewuenschte GPU-Zeile.
        """
        rows = self._rows_of(target_idx)
        if rows <= 0:
            return
        self._select_cell(target_idx, max(0, min(row_idx, rows - 1)),
                          scroll="into")

    def _on_key_up(self, event):
        """Eine GPU-Zeile hoch; ueber der obersten in die Karte darueber in
        DERSELBEN Spalte, dort auf die letzte Zeile."""
        m, r = self._sel_model, self._sel_row
        if m < 0:
            return
        if r > 0:
            self._select_cell(m, r - 1, scroll="into")
            return
        above = m - self._current_model_cols()
        if above >= 0:
            self._goto_card(above, self._rows_of(above) - 1)

    def _on_key_down(self, event):
        """Eine GPU-Zeile runter; unter der letzten in die Karte darunter in
        DERSELBEN Spalte, dort auf die erste Zeile."""
        m, r = self._sel_model, self._sel_row
        if m < 0:
            if getattr(self, "installed_models", None) and self._rows_of(0):
                self._goto_card(0, 0)
            return
        if r < self._rows_of(m) - 1:
            self._select_cell(m, r + 1, scroll="into")
            return
        below = m + self._current_model_cols()
        if below < len(self.models):
            self._goto_card(below, 0)

    def _on_key_left(self, event):
        """Zur Karte links davon, gleiche Rasterzeile, gleiche GPU-Zeile."""
        m = self._sel_model
        if m < 0:
            return
        # Bewusst nicht einfach m-1: das spraenge von der linken Karte einer
        # Zeile zur rechten der Zeile darueber und liest sich als Sprung.
        if m % self._current_model_cols() > 0:
            self._goto_card(m - 1, self._sel_row)

    def _on_key_right(self, event):
        """Zur Karte rechts davon, gleiche Rasterzeile, gleiche GPU-Zeile."""
        m = self._sel_model
        if m < 0:
            return
        cols = self._current_model_cols()
        if (m % cols < cols - 1
                and m + 1 < len(self.models)):
            self._goto_card(m + 1, self._sel_row)

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
        # AP-QL-72 (Fund A1), woertlich aus der archivierten Windows-Linie.
        # AP-QL-01: same central validation as _start_server — a stale or
        # profile-injected turbo config must never reach llama-bench on a
        # mainline/unknown build.
        if not is_kv_allowed(kv_name, self._current_server_family()):
            self._log(f"KV config '{kv_name}' not supported by the active "
                      f"{self._current_server_family()} server — benchmarking "
                      f"{SAFE_KV_FALLBACK} instead.", "warn")
            kv_name = SAFE_KV_FALLBACK
            self._kv_var.set(kv_name)
            self._refresh_kv_button_visuals()
        # AP-QL-74 (Delta-Audit 03.08., Fund B): woertlich aus der
        # archivierten Windows-Linie zurueckgeholt.
        # AP-QL-02: capture the LA mode on the main thread and hand it to
        # the worker explicitly — it was silently dropped before.
        la_mode = self._la_var.get()
        if is_cpu:
            kv_name = "f16 (default)"
            la_mode = 0
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
               f"KV-Cache: {kv_name}\n"
               f"Layer-Adaptive: {la_label(la_mode)}\n\n"
               f"Runs f16 baseline (LA=off) + selected KV config.\n"
               f"Estimated time: ~90s")
        if not messagebox.askyesno("Benchmark", msg):
            return

        self._log(f"Benchmark: {model.filename} on {target_label} "
                  f"({kv_name}, LA={la_label(la_mode)})", "info")
        self._status_label.config(text="● Benchmarking...", fg=self.theme.yellow)

        thread = threading.Thread(target=self._run_benchmark,
                                   args=(model, gpu_key, kv_name, la_mode, is_cpu),
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
        bench_exe = os.path.join(server_dir, exe_name("llama-bench"))

        if not os.path.isfile(bench_exe):
            # AP-QL-76 (Fund D): den echten Dateinamen nennen. Der Pfad wird
            # direkt darueber ueber exe_name() gebildet — auf Linux suchte
            # der Benutzer bisher eine Datei, die es dort nie gab.
            self.after(0, self._log,
                       f"{exe_name('llama-bench')} not found in {server_dir}",
                       "error")
            self.after(0, self._bench_all_finished, None)
            return

        # AP-QL-01 (via AP-QL-16): Familie EINMAL an die Exe binden, die
        # dieser Worker erfasst hat — nicht an die Live-Config, die ein
        # Quick-Switch mitten im Lauf aendern kann.
        worker_family = detect_server_family(server_exe)
        dropped = [k for k in kv_list if not is_kv_allowed(k, worker_family)]
        if dropped:
            self.after(0, self._log,
                       f"Bench All: skipping KV configs not supported by the "
                       f"captured {worker_family} llama-bench: "
                       f"{', '.join(dropped)}", "warn")
            kv_list = [k for k in kv_list if is_kv_allowed(k, worker_family)]
        if not kv_list:
            self.after(0, self._log,
                       "Bench All: no valid KV configs for the captured "
                       "server family.", "warn")
            self.after(0, self._bench_all_finished, None)
            return

        ngl = "0" if is_cpu else "99"
        results = {}
        step = 0
        depth_list = list(BENCH_DEPTH_LIST)

        # AP-QL-05 (via AP-QL-16): kein kartesisches Produkt mehr. Der Plan
        # entsteht vorab aus der Gueltigkeitsmatrix, damit `total` exakt die
        # Tupel zaehlt, die wirklich laufen. Die alte Zaehlung multiplizierte
        # kv × la × depth und uebersprang Ungueltiges dann innen — der
        # Fortschritt zeigte Luecken und erreichte sein eigenes Ziel nie.
        plan = [(la, [k for k in kv_list
                      if is_la_allowed(la, k, worker_family)])
                for la in la_list]
        plan = [(la, kvs) for la, kvs in plan if kvs]

        dropped_la = [la_label(la) for la in la_list
                      if la not in {p_la for p_la, _ in plan}]
        if dropped_la:
            self.after(0, self._log,
                       f"Bench All: skipping LA modes without any valid KV "
                       f"config on this {worker_family} server: "
                       f"{', '.join(dropped_la)}", "warn")
        if not plan:
            self.after(0, self._log,
                       "Bench All: no valid KV/LA tuples for the captured "
                       "server family.", "warn")
            self.after(0, self._bench_all_finished, None)
            return

        total = sum(len(kvs) for _, kvs in plan) * len(depth_list)

        # Resolve GPU display name
        if is_cpu:
            gpu_display = "CPU"
        else:
            gpu_display = gpu_key
            for g in self.gpus:
                if f"GPU {g.index}" == gpu_key:
                    gpu_display = g.name.replace("NVIDIA GeForce ", "")
                    break

        for la_mode, la_kv_list in plan:
            if self._bench_stop_event.is_set():
                self.after(0, self._log, "Benchmark stopped by user.", "warn")
                break

            la_tag = f"LA={la_label(la_mode)}"
            self.after(0, self._log, f"\n  ═══ {la_tag} — {gpu_display} ═══", "info")

            env = os.environ.copy()
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            if not is_cpu:
                env["CUDA_VISIBLE_DEVICES"] = gpu_key.split(" ")[1]
            # AP-QL-02 (via AP-QL-16): off setzt "0" explizit, Auto entfernt
            # den Schluessel auch bei geerbtem Wert. Das alte 'del bei 0' hat
            # off in Auto verwandelt.
            apply_la_env(env, la_mode)

            for depth in depth_list:
                if self._bench_stop_event.is_set():
                    break

                depth_tag = f"d={depth}" if depth > 0 else "d=0"
                self.after(0, self._log,
                           f"  ─── {depth_tag} ───", "info")

                for kv_name in la_kv_list:
                    if self._bench_stop_event.is_set():
                        break
                    # AP-QL-01/02/05 (via AP-QL-16, defensiv): `plan` ist
                    # bereits aus der Gueltigkeitsmatrix der erfassten Familie
                    # gebaut, keiner der Guards darf feuern. Falls doch, wird
                    # das Tupel verworfen OHNE einen Schritt zu verbrauchen —
                    # `total` zaehlt nur geplante Tupel, und ein stiller Skip
                    # darf die Fortschrittsanzeige nicht entkoppeln.
                    if not is_kv_allowed(kv_name, worker_family):
                        self.after(0, self._log,
                                   f"  ▸ {kv_name} dropped — not supported by "
                                   f"the captured {worker_family} server "
                                   f"(unexpected: not in plan).", "error")
                        continue
                    if not is_la_allowed(la_mode, kv_name, worker_family):
                        self.after(0, self._log,
                                   f"  ▸ {kv_name} ({la_tag}) dropped — LA mode "
                                   f"has no valid effect for this tuple "
                                   f"(unexpected: not in plan).", "error")
                        continue
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

    def _run_benchmark(self, model: ModelInfo, gpu_key: str, kv_name: str,
                       la_mode: int, is_cpu: bool):
        """Background thread: run llama-bench twice (f16 baseline + selected KV).

        AP-QL-74 (Fund B): ``la_mode`` war beim Zusammenlegen der Linien aus
        der Signatur verschwunden. Ohne ihn wurde ``apply_la_env`` hier nie
        gerufen — und weil ``env`` eine Kopie von ``os.environ`` ist, blieb
        TURBO_LAYER_ADAPTIVE schlicht ungesetzt. Fuer den Fork bedeutet eine
        FEHLENDE Variable aber Auto, nicht off: wer LA=off waehlte, mass in
        Wahrheit Auto, und in der Ergebnisdatei stand ein nackter KV-Name,
        dem man das nicht mehr ansah.
        """
        server_exe = self.cfg.get("llama_server_path", "")
        server_dir = os.path.dirname(server_exe)
        bench_exe = os.path.join(server_dir, exe_name("llama-bench"))

        if not os.path.isfile(bench_exe):
            # AP-QL-76 (Fund D): den echten Dateinamen nennen. Der Pfad wird
            # direkt darueber ueber exe_name() gebildet — auf Linux suchte
            # der Benutzer bisher eine Datei, die es dort nie gab.
            self.after(0, self._log,
                       f"{exe_name('llama-bench')} not found in {server_dir}",
                       "error")
            self.after(0, self._bench_finished, None)
            return

        # AP-QL-72 (Fund A1), woertlich aus der archivierten Windows-Linie.
        # AP-QL-01 (Rework): bind the family to the executable THIS worker
        # actually captured. Validating against the live config would race
        # with a Quick-Switch performed after the UI check — a turbo config
        # could then reach a mainline llama-bench.exe.
        worker_family = detect_server_family(server_exe)
        if not is_kv_allowed(kv_name, worker_family):
            self.after(0, self._log,
                       f"KV config '{kv_name}' not supported by the captured "
                       f"{worker_family} llama-bench — benchmarking "
                       f"{SAFE_KV_FALLBACK} instead.", "warn")
            kv_name = SAFE_KV_FALLBACK
        # AP-QL-74 (Fund B) / AP-QL-02: validate the LA mode against the same
        # captured family and the (possibly fallen-back) KV preset.
        if not is_la_allowed(la_mode, kv_name, worker_family):
            self.after(0, self._log,
                       f"LA mode '{la_label(la_mode)}' is not valid for "
                       f"{kv_name} on the captured {worker_family} "
                       f"llama-bench — using LA=off (0) instead.", "warn")
            la_mode = 0

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

        # AP-QL-74 (Fund B) / AP-QL-02: baseline always runs with explicit
        # LA=0 in its own env copy — a missing variable would mean Auto on
        # TheTom, und dann waere die Grundlinie keine Grundlinie mehr.
        env_baseline = env.copy()
        apply_la_env(env_baseline, 0)
        self.after(0, self._log,
                   f"  ▸ f16 Baseline (LA=off, {gpu_display})...", "info")
        baseline = self._exec_bench(bench_exe, model.path, ngl, None, None,
                                    env_baseline)
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
        # AP-QL-74 (Fund B) / AP-QL-02: log line and stored result label must
        # state the actual LA mode; LA=0 keeps the legacy plain label, damit
        # aeltere Eintraege in TurboQuant_Benchmark_results.md vergleichbar
        # bleiben.
        kv_result_label = (kv_name if la_mode == 0
                           else f"{kv_name} [LA={la_label(la_mode)}]")
        if ctk or ctv:
            env_selected = env.copy()
            apply_la_env(env_selected, la_mode)
            self.after(0, self._log, f"  ▸ {kv_result_label}...", "info")
            turbo = self._exec_bench(bench_exe, model.path, ngl, ctk, ctv,
                                     env_selected)
            if turbo:
                results[kv_result_label] = turbo
                pp_key = next((k for k in turbo if k.startswith("pp")), None)
                tg_key = next((k for k in turbo if k.startswith("tg")), None)
                pp = turbo.get(pp_key, "?") if pp_key else "?"
                tg = turbo.get(tg_key, "?") if tg_key else "?"
                self.after(0, self._log, f"    {kv_result_label}: {pp_key or 'pp'}={pp} t/s, {tg_key or 'tg'}={tg} t/s", "good")

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
        # AP-QL-96: reading a Tk variable is a Tcl call, so it belongs on the
        # main thread. Both benchmark workers reach this through _exec_bench
        # (:8600, :8707, :8730) and were calling .get() from their own thread
        # — the one place in this file where a worker still touched Tk
        # directly. Found by an AST audit that walks every thread entry point
        # transitively and knows this file's two marshalling idioms, after()
        # and _main_thread_call(); see dev/_scratch/AP-QL-96/.
        #
        # Marshalled rather than passed down as a parameter: _exec_bench has
        # three call sites and a long signature already, and a setting that
        # is read once per run does not need to travel through all of them.
        # The guard is required — _main_thread_call blocks on an Event that
        # only the main loop can set, so calling it FROM the main thread
        # would deadlock. :11225 is such a caller.
        if threading.current_thread() is not threading.main_thread():
            return self._main_thread_call(self._get_bench_timeout)
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
        """Append benchmark results to the per-OS Benchmark_results markdown file."""
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

    # ── AP-QL-33/40: Anzeige des API-Schluessels ─────────────────────────
    # Feste Laenge, damit die Redigierung nichts ueber den Schluessel verraet
    # — weder seinen Inhalt noch seine Laenge. Echte Schluessel sind 50-60
    # Zeichen lang; show="•" haette diese Laenge ueber die Punktzahl verraten.
    #
    # AP-QL-40: Das "sk-"-Praefix ist eine bewusste Owner-Entscheidung
    # (2026-08-01). Es stimmt nicht zwingend — llama-servers --api-key nimmt
    # jede Zeichenkette, und ein mit Ziffern beginnender Schluessel wird hier
    # trotzdem als "sk-..." dargestellt. Der Owner wurde darauf hingewiesen
    # und hat es ausdruecklich so gewollt: das Feld liest sich damit sofort
    # als "hier steht ein Schluessel". Wer den Wert braucht, klickt hinein.
    # NICHT ohne Ruecksprache wieder aendern.
    API_KEY_REDACTED = "sk-••••••••"

    def _redact_api_key_field(self) -> None:
        """Das Feld auf die Redigierung setzen, wenn ein Schluessel da ist."""
        entry = getattr(self, "_api_key_entry", None)
        if entry is None:
            return
        try:
            if (self._api_key_var.get() or "").strip():
                self._api_key_display.set(self.API_KEY_REDACTED)
                entry.configure(show="")        # Platzhalter lesbar zeigen
            else:
                self._api_key_display.set("")
                entry.configure(show="")
        except tk.TclError:
            pass

    def _on_api_key_focus_in(self, event=None) -> None:
        """Beim Hineinklicken den echten Wert zum Bearbeiten anbieten.

        Waehrend der Eingabe maskiert show="•" die Zeichen wie gewohnt — die
        Laenge ist dann zwangslaeufig sichtbar, aber nur solange das Feld den
        Fokus hat und jemand aktiv davorsitzt.
        """
        try:
            self._api_key_display.set(self._api_key_var.get())
            self._api_key_entry.configure(show="•")
            self._api_key_entry.icursor("end")
        except tk.TclError:
            pass

    def _on_api_key_focus_out(self, event=None) -> None:
        """Das Getippte uebernehmen und das Feld wieder redigieren."""
        try:
            typed = (self._api_key_display.get() or "").strip()
            # Die Redigierung selbst darf nie als Schluessel durchgehen.
            if typed != self.API_KEY_REDACTED:
                self._api_key_var.set(typed)
            self._redact_api_key_field()
        except tk.TclError:
            pass

    def _api_key_value(self) -> str:
        """Den konfigurierten API-Schluessel liefern, sonst "" (AP-QL-26).

        Liest das GUI-Feld, wenn es existiert (der Nutzer kann einen
        Schluessel getippt, aber noch nicht gespeichert haben) und faellt
        sonst auf die Datei zurueck — auch Arbeitsthreads rufen das hier auf,
        und tk-Variablen sind von dort aus lesbar.

        Returns:
            Der Schluessel, oder "" wenn keiner gesetzt ist.
        """
        # AP-QL-33: haengt das Feld noch im Fokus, steht der frisch getippte
        # Wert nur in der Anzeigevariable — _api_key_var wird erst beim
        # Verlassen uebernommen. Ohne diesen Zweig startete ein "tippen und
        # sofort Run" den Server ohne den gerade eingegebenen Schluessel.
        disp = getattr(self, "_api_key_display", None)
        entry = getattr(self, "_api_key_entry", None)
        if disp is not None and entry is not None:
            try:
                if entry.focus_get() is entry:
                    typed = (disp.get() or "").strip()
                    if typed and typed != self.API_KEY_REDACTED:
                        return typed
            except Exception:
                pass
        var = getattr(self, "_api_key_var", None)
        if var is not None:
            try:
                return var.get().strip()
            except Exception:
                pass
        return load_api_key()   # AP-QL-10d: key_Linux.json, nicht die Config

    def _effective_server_host(self) -> str:
        """Die Adresse liefern, an die llama-server binden soll (AP-QL-26).

        Folgt der LAN-Checkbox. Bei freigegebenem LAN gewinnt eine in
        cfg["server_host"] eingetragene Adapter-IP gegen die 0.0.0.0-Wildcard,
        damit eine bewusste Bindung an eine einzelne Schnittstelle nicht
        stillschweigend aufgeweitet wird.

        Returns:
            Die Bind-Adresse als Zeichenkette.
        """
        var = getattr(self, "_lan_var", None)
        if var is None or not var.get():
            return SERVER_HOST_LOOPBACK
        configured = str(self.cfg.get("server_host") or "").strip()
        if configured and configured != SERVER_HOST_LOOPBACK:
            return configured
        return SERVER_HOST_ALL

    def _gpu_source_color(self, vendor: str) -> str:
        """Die Beschriftungsfarbe einer GPU-Zeile (AP-QL-41).

        Alle Quellen gruen, je Quelle nur ganz leicht im Farbton verschoben.
        Unbekannte Hersteller bekommen den Bezugspunkt, nicht `fg_dim` —
        eine graue Zeile haette sonst wie "nicht verfuegbar" ausgesehen.

        Args:
            vendor: Herstellerkennung aus GPUInfo, z. B. "nvidia".

        Returns:
            Farbe als "#rrggbb".
        """
        shift = GPU_SOURCE_HUE_SHIFT.get((vendor or "").lower(), 0)
        return shift_hue(self.theme.green, shift)

    def _is_running_row(self, filename: str, gpu_key: str) -> bool:
        """True, wenn genau diese (Modell, GPU)-Zeile gerade laeuft."""
        return filename == self.running_model and gpu_key == self._running_gpu_key

    def _paint_row(self, rd: dict, bg: str) -> None:
        """Faerbt eine GPU-Zeile samt aller Kindwidgets.

        AP-QL-22 (Portierung von AP-QL-19). Der BrailleBar ist ein Canvas und
        muss ueber set_value() neu gezeichnet werden — ein blosses
        configure(bg=...) laesst seine gezeichneten Balken auf dem alten
        Hintergrund stehen.

        Args:
            rd: Zeilen-Dict aus card_data["gpu_rows"].
            bg: Zielfarbe als Tk-Farbstring.
        """
        try:
            rd["frame"].configure(bg=bg)
            for child in rd["frame"].winfo_children():
                try:
                    child.configure(bg=bg)
                except tk.TclError:
                    pass          # Widgets ohne bg-Option (z. B. ttk)
            eff = kv_effective_gb(rd["size_gb"], self._kv_var.get())
            rd["bar"].set_value(eff, rd["total_gb"], canvas_bg=bg)
        except tk.TclError:
            pass                  # Zeile bereits zerstoert (Rebuild laeuft)

    def _update_running_indicator(self):
        """Punkt UND Zeilentoenung fuer das laufende Modell setzen.

        AP-QL-22 (Portierung von AP-QL-19): vorher gab es nur den gruenen
        Punkt. Die Toenung macht auf einen Blick sichtbar, welches Modell
        wirklich geladen ist — auch wenn die Auswahl gerade woanders steht.
        """
        t = self.theme
        for filename, card_data in self._model_cards.items():
            for rd in card_data["gpu_rows"]:
                if self._is_running_row(filename, rd["key"]):
                    rd["indicator"].config(text="● ", fg=t.green)
                    self._paint_row(rd, t.row_running_bg)
                else:
                    rd["indicator"].config(text="  ")
                    self._paint_row(rd, rd["card_bg"])

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION: Server Management
    # ═══════════════════════════════════════════════════════════════════════

    def _on_run_model(self, filename: str, gpu_key: str = "GPU 0",
                      owner_token: Optional[str] = None):
        """Start a model on specified GPU. Stops any running server first."""
        if self.running_model == filename and self._running_gpu_key == gpu_key:
            self._stop_server()
            return

        if self.server_process:
            if not self._stop_server(silent=True):
                self._log("Model switch cancelled — the running server "
                          "could not be stopped.", "error")
                return
            self.after(500, lambda: self._start_server(
                filename, gpu_key, owner_token=owner_token))
        else:
            self._start_server(filename, gpu_key, owner_token=owner_token)

    def _start_server(self, filename: str, gpu_key: str = "GPU 0",
                      owner_token: Optional[str] = None):
        # A RELEASE can arrive while a newly opened launcher is still scanning
        # its models.  The capability was revoked before this queued callback
        # ran, so starting now would resurrect an ownerless GPU allocation.
        if self._model_lease_is_revoked(owner_token):
            self._log("Autostart cancelled — its external owner has gone.",
                      "info")
            return
        t = self.theme
        card_data = self._model_cards.get(filename)
        if not card_data:
            return

        model = next((m for m in self.models if m.filename == filename), None)
        if not model:
            return

        server_exe = self.cfg.get("llama_server_path", "")
        if not os.path.isfile(server_exe):
            self._log(f"{exe_name('llama-server')} not found: {server_exe}", "error")
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
        # Cross-platform: works identically on Linux and Windows since
        # llama.cpp uses the same --mmproj flag on both platforms.
        # ────────────────────────────────────────────────────────────────
        # AP-QL-88: the rule itself now lives in find_mmproj() so the Vision
        # role check and this command line cannot disagree about what counts
        # as a projector. A scan failure still yields None and a text-only
        # start — a missing projector must never stop a server.
        mmproj_path = find_mmproj(model.path)
        if mmproj_path:
            cmd.extend(["--mmproj", mmproj_path])
            image_token_args = _vision_image_token_args(model, mmproj_path)
            cmd.extend(image_token_args)
            self._log(
                f"  Vision encoder auto-attached: "
                f"{os.path.basename(mmproj_path)}",
                "info",
            )
            if image_token_args:
                self._log(
                    f"  Qwen-VL grounding floor: "
                    f"{_QWEN_VL_MIN_IMAGE_TOKENS} image tokens",
                    "info",
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
        # AP-QL-72 (Delta-Audit 03.08., Fund A1): woertlich aus der
        # archivierten Windows-Linie zurueckgeholt. Beim Zusammenlegen der
        # Linien war diese letzte Instanz vor dem Kommandozeilenbau
        # verlorengegangen — SAFE_KV_FALLBACK stand danach nur noch als
        # unbenutzte Konstante im Modul (12 Verwendungen -> 1).
        #
        # AP-QL-01: final start-time validation — UI gating alone is not
        # sufficient (config files, profiles and code paths can set the
        # KV var). An invalid config never reaches the command line.
        if not is_kv_allowed(kv_name, self._current_server_family()):
            self._log(f"KV config '{kv_name}' not supported by the active "
                      f"{self._current_server_family()} server — starting "
                      f"with {SAFE_KV_FALLBACK} instead.", "warn")
            kv_name = SAFE_KV_FALLBACK
            self._kv_var.set(kv_name)
            self._refresh_kv_button_visuals()
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
        # AP-QL-85: Multi-Token Prediction, an das MODELL gebunden.
        #
        # Bis hierher setzte QLauncher --spec-type nie — der produktiv
        # gestartete Server lief also ohne MTP, waehrend die Startzeile im
        # Handover es von Hand mitgab. Das kostete gemessen den Faktor 1,76
        # (103,0 gegen 58,5 tok/s, Qwen3.8-27B-Q6_K auf der RTX 5090,
        # 2026-08-21).
        #
        # has_nextn kommt aus dem GGUF selbst (_detect_model_traits). Damit
        # kann das Flag nicht global gesetzt werden, und genau darauf kam es
        # an: auf einem Modell ohne NextN-Koepfe bricht llama-server mit
        # Exit 1 ab ("context type MTP requested but model doesn't contain
        # MTP layers"). Die Tiefe bleibt beim Server-Default 3 — gemessen das
        # Optimum (68,3 tok/s gegen 62,9 bei 1 und 61,1 bei 5).
        if self.cfg.get("mtp_mode", "auto") == "auto" and model.has_nextn:
            cmd.extend(["--spec-type", "draft-mtp"])
            self._log(
                f"MTP on: '{filename}' carries NextN heads — "
                f"--spec-type draft-mtp set (measured 1.76x on Qwen3.8-27B).",
                "info")

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
            # Keep the persisted/UI source in sync with the actual command.
            # Otherwise the post-Popen config save parses the original bad
            # text and turns a successful spawn into a cleanup exception.
            self._port_var.set(port_raw)

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
            # AP-QL-105: name the occupant instead of guessing at it out
            # loud, and offer a free port beside the kill. Measured on
            # 2026-09-16: the process on 8080 was our own llama-server,
            # started by ReachyMini's deploy copy of this launcher.
            choice = self._ask_port_conflict(port_int)
            action = choice["action"]
            if action == "cancel":
                self._log(
                    f"Server start cancelled — port {port_int} is in "
                    f"use.",
                    "warn")
                return
            if action == "switch":
                port_int = int(choice["port"])
                port_raw = str(port_int)
                # Keep the UI and the persisted config on the port we
                # actually start on, exactly as the invalid-port path does.
                self._port_var.set(port_raw)
                self._log(
                    f"Starting on port {port_int} instead — the other "
                    f"process keeps running.",
                    "info")
            elif action == "stop":
                if not self._free_port_by_stopping(port_int,
                                                   choice["report"]):
                    self._log(
                        f"Server start cancelled — port {port_int} "
                        f"could not be freed.",
                        "error")
                    return
            else:
                self._log(
                    f"Starting on port {port_int} despite apparent "
                    f"conflict — user confirmed override.",
                    "warn")

        port = port_raw
        # AP-QL-26 (AP-QL-10a/b): die Bind-Adresse folgt jetzt der
        # LAN-Checkbox statt des frueheren fest verdrahteten 0.0.0.0.
        # llama.cpps Vorgaben (alle CORS-Quellen, kein Schluessel) bedeuten,
        # dass eine LAN-Bindung das Modell jedem aushaendigt, der den Port
        # erreicht — deshalb sagt der Launcher das ausdruecklich und nennt
        # die URL, die tatsaechlich funktioniert.
        host = self._effective_server_host()
        self._running_host = host          # AP-QL-32
        cmd.extend(["--host", host, "--port", port])

        # AP-QL-26 (AP-QL-10c): optionaler Schluessel. Nie geloggt,
        # siehe mask_command_for_log.
        api_key = self._api_key_value()
        if api_key:
            cmd.extend(["--api-key", api_key])

        # AP-QL-65 (F2): the full LAN security messaging of the archived
        # Windows line, ported verbatim — and English throughout (owner
        # rule 2026-08-03; the interim line here was German and carried
        # neither the CORS hint nor the key-required confirmation).
        if host != SERVER_HOST_LOOPBACK:
            lan_ip = detect_lan_ip()
            where = f"http://{lan_ip}:{port}" if lan_ip else f"port {port}"
            self._log(f"  LAN released — reachable from your network at "
                      f"{where}", "warn")
            if api_key:
                self._log(f"  API key required — clients must send "
                          f"'Authorization: Bearer <key>'.", "info")
            else:
                self._log(f"  No API key set — anyone who can reach it can "
                          f"use the model (llama.cpp allows all CORS "
                          f"origins). Set a key in the Key field.", "warn")
        elif api_key:
            self._log(f"  API key required — clients must send "
                      f"'Authorization: Bearer <key>'.", "info")

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

        # AP-QL-02: validate the (kv, la) tuple immediately before the
        # environment is built, then apply the exact semantics via the
        # central helper. Off (0) must SET "0" — a missing variable means
        # Auto on TheTom; Auto (-1) must REMOVE even an inherited value.
        #
        # AP-QL-72 (Fund A1): die Validierung ist woertlich aus der
        # archivierten Windows-Linie zurueckgeholt. Ohne sie wurde ein
        # unzulaessiges TURBO_LAYER_ADAPTIVE gesetzt, der Server startete
        # normal — und gemessen wurde etwas anderes als eingestellt. Der
        # Aufruf von _apply_la_gating() der Vorlage entfaellt hier: die
        # Methode gibt es in dieser Linie (noch) nicht (Fund A2); die
        # Knopffarbe zieht _select_la nach, sobald A2 sie zurueckbringt.
        la_mode = self._la_var.get()
        if not is_la_allowed(la_mode, kv_name, self._current_server_family()):
            self._log(f"LA mode '{la_label(la_mode)}' is not valid for "
                      f"{kv_name} on the active "
                      f"{self._current_server_family()} server — starting "
                      f"with LA=off (0) instead.", "warn")
            la_mode = 0
            self._la_var.set(0)
        apply_la_env(env, la_mode)

        la_info = f", LA={la_label(la_mode)}"
        # AP-QL-65 (F2): name the actual bind host, not just the port — a
        # LAN release must be visible in the start line itself.
        self._log(f"Starting {filename} on {gpu_display} ({kv_name}, {host}:{port}{la_info})", "info")
        self._log(f"  Server: {server_exe}", "info")
        # AP-QL-26 (AP-QL-10c): maskiert — der Schluessel darf nie im Log stehen.
        self._log(f"$ {' '.join(mask_command_for_log(cmd))}", "info")

        proc = None
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
            # RELEASE runs on an IPC thread.  Hold the lease-state lock from
            # the final revocation check through Popen and immediate adoption:
            # it can therefore observe either "not started and revoked" or a
            # concrete owned slot, never the false-empty gap between them.
            lease_lock = (self._model_lease_state_lock
                          if owner_token is not None else None)
            if lease_lock is not None:
                lease_lock.acquire()
            try:
                if (owner_token is not None
                        and owner_token in self._revoked_model_leases):
                    self._running_host = None
                    self._log("Server start cancelled — its external owner "
                              "has gone.", "info")
                    return
                proc = subprocess.Popen(cmd, **kw)
                # Adopt the process and capability before releasing the state
                # lock. Every later UI/config/thread action may raise safely.
                inst = self.slot(SLOT_PRIMARY)
                inst.process = proc
                inst.owner_token = owner_token
                inst.model = filename
                inst.gpu_key = gpu_key
                inst.port = port
                inst.host = host
            finally:
                if lease_lock is not None:
                    lease_lock.release()
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

            self._log_reader_thread = threading.Thread(
                target=self._read_server_output, args=(SLOT_PRIMARY,),
                daemon=True)
            self._log_reader_thread.start()
            # AP-QL-113: bei JEDEM Start nachsehen, welche Karte es wurde.
            threading.Thread(target=self._confirm_gpu_identity,
                             args=(SLOT_PRIMARY,), daemon=True).start()
            self._refresh_action_btns()   # AP-QL-25: Stop stoppt jetzt den Server

        except Exception as e:
            self._log(f"Failed to start server: {e}", "error")
            if (proc is not None
                    and not self._cleanup_failed_server_start(
                        SLOT_PRIMARY, proc, owner_token, "Server")):
                # The process resisted every escalation. Keep its exact slot
                # and lease visible so RELEASE cannot report false success.
                return
            self.server_process = None
            self.running_model = None
            self._running_gpu_key = None
            self._running_kv = None
            self._running_la = None
            self._running_host = None   # AP-QL-32
            self.slot().owner_token = None
            # AP-QL-78: after a pre-Popen failure or a confirmed cleanup the
            # controls return to idle. AP-QL-91 deliberately does not do this
            # for an unkillable post-Popen process: that process stays tracked.
            self._update_status_idle()
            self._refresh_action_btns()

    # ─── AP-QL-90: the vision slot ────────────────────────────────────────

    def vision_gpu_key(self) -> str:
        """Which card the vision server should use.

        A configured value wins. Without one, the default is "not the card the
        primary server is on" — the two run side by side, and putting both on
        the same GPU is the one arrangement that reliably runs it out of
        memory. Measured on this machine 2026-08-21: the brain holds ~27 GB of
        the 5090's 32.6 GB, so a second model there has nowhere to go, while
        the 4090 sits at 4.3 GB of 24.5 GB.
        """
        configured = str(self.cfg.get("vision_gpu_key") or "").strip()
        if configured:
            return configured
        belegt = self.slot().gpu_key or ""
        for gpu in self.gpus:
            key = f"GPU {gpu.index}"
            if key != belegt:
                return key
        return belegt or "GPU 0"

    def _start_vision_server(self,
                             owner_token: Optional[str] = None) -> bool:
        """Start the model assigned to the Vision role on its own port.

        Returns:
            True when a process was spawned. False on every refusal, each of
            which is logged with the reason — a silent no-op here would look
            exactly like a server that started and died.
        """
        if self._model_lease_is_revoked(owner_token):
            self._log("Vision autostart cancelled — its external owner has "
                      "gone.", "info")
            return False
        if self.slot(SLOT_VISION).is_alive():
            self._log("Vision: already running.", "info")
            return False

        name = (self.cfg.get(ROLE_VISION, "") or "").strip()
        if not name:
            self._log("Vision: no model assigned to the Vision role — "
                      "pick one in the model header.", "warn")
            return False
        model = self._model_by_filename(name)
        if model is None:
            self._log(f"Vision: the assigned model '{name}' is not in the "
                      f"current scan.", "warn")
            return False

        server_exe = self.cfg.get("llama_server_path", "")
        if not os.path.isfile(server_exe):
            self._log(f"Vision: {exe_name('llama-server')} not found: "
                      f"{server_exe}", "error")
            return False

        port = str(self.cfg.get("vision_port", 8081))
        if str(self.cfg.get("port", 8080)) == port:
            self._log(f"Vision: port {port} is the same as the main server's "
                      f"— give the vision server its own port.", "error")
            return False

        gpu_key = self.vision_gpu_key()
        is_cpu = (gpu_key == "CPU")
        cmd = [server_exe, "-m", model.path,
               "-ngl", "0" if is_cpu else "99",
               "-c", str(self.cfg.get("vision_ctx", 8192))]

        mmproj_path = find_mmproj(model.path)
        if mmproj_path:
            cmd.extend(["--mmproj", mmproj_path])
        else:
            # Not fatal: the server starts, it just cannot see. Saying so here
            # is the difference between "no images" and "why is it ignoring
            # my picture".
            self._log(f"Vision: no mmproj*.gguf next to {name} — this server "
                      f"will start text-only.", "warn")

        deckel = int(self.cfg.get("vision_image_max_tokens", 128) or 0)
        image_token_args = _vision_image_token_args(
            model, mmproj_path, configured_max=deckel)
        cmd.extend(image_token_args)
        if ("--image-min-tokens" in image_token_args
                and 0 < deckel < _QWEN_VL_MIN_IMAGE_TOKENS):
            self._log(
                f"Vision: image-token cap {deckel} raised to "
                f"{_QWEN_VL_MIN_IMAGE_TOKENS} for Qwen-VL grounding.",
                "info",
            )

        # Dieselbe Bindungsregel wie beim Hauptserver — der LAN-Haken
        # gilt fuer beide Server, sonst waere einer freigegeben und der
        # andere nicht, ohne dass es irgendwo steht.
        host = self._effective_server_host()
        cmd.extend(["--host", host, "--port", port])

        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if not is_cpu:
            env["CUDA_VISIBLE_DEVICES"] = gpu_key.split(" ")[1]

        proc = None
        inst = self.slot(SLOT_VISION)
        try:
            kw = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                      text=True, bufsize=1, env=env,
                      errors="replace")
            if sys.platform == "win32":
                kw["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP
                                       | 0x08000000)
            lease_lock = (self._model_lease_state_lock
                          if owner_token is not None else None)
            if lease_lock is not None:
                lease_lock.acquire()
            try:
                if (owner_token is not None
                        and owner_token in self._revoked_model_leases):
                    self._log("Vision autostart cancelled — its external "
                              "owner has gone.", "info")
                    return False
                proc = subprocess.Popen(cmd, **kw)
                inst.process = proc
                inst.owner_token = owner_token
                inst.role = ROLE_VISION
                inst.model = name
                inst.gpu_key = gpu_key
                inst.port = int(port)
                inst.host = host
            finally:
                if lease_lock is not None:
                    lease_lock.release()
            inst.log_thread = threading.Thread(
                target=self._read_server_output,
                args=(SLOT_VISION,), daemon=True)
            inst.log_thread.start()
            # AP-QL-113: auch der Vision-Server sagt, welche Karte es wurde.
            # Er laeuft oft auf der ANDEREN als der Brain-Server -- gerade
            # dort ist eine Verwechslung teuer.
            threading.Thread(target=self._confirm_gpu_identity,
                             args=(SLOT_VISION,), daemon=True).start()
            self._log(f"Vision: starting {name} on {gpu_key}, port {port}"
                      + (f", image cap {deckel}" if deckel > 0 else ""),
                      "good")
            self._refresh_vision_widgets()
            return True
        except Exception as e:
            self._log(f"Vision: failed to start: {e}", "error")
            if (proc is not None
                    and not self._cleanup_failed_server_start(
                        SLOT_VISION, proc, owner_token, "Vision")):
                self._refresh_vision_widgets()
                return False
            self._servers.pop(SLOT_VISION, None)
            self._refresh_vision_widgets()
            return False

    # ─── AP-QL-111 (2026-09-17): Unload je GPU ──────────────────────────
    # ⚙ Owner: "eigentlich brauchen wir neben Run und Stop auch noch einen
    # unload Button" und, entscheidend: "wenn wir mehrere GPUs haben, dann
    # sollten wir auch die entsprechende Anzahl an Unload Buttons haben (für
    # jede GPU einen, aber nur die tatsächliche Anzahl an GPUs anzeigen)".
    #
    # WAS ES TUT, und warum es kein zweites Stop ist: Stop beendet, was im
    # PRIMAERSLOT laeuft. Seit AP-QL-89 kann der Launcher aber Brain und
    # Vision gleichzeitig fahren, jeden auf einer eigenen GPU. Wer den VRAM
    # einer bestimmten Karte zurueckhaben will, hatte dafuer keinen Griff.
    # "Unload 1" beendet alles, was DIESER Launcher auf GPU 1 gestartet hat,
    # und laesst die andere Karte in Ruhe.
    #
    # ⚠ Nur eigene Prozesse. Das ist dieselbe Eigentuemerregel wie bei
    # AP-QL-91: der Launcher beendet, was er verursacht hat, nie ein
    # vorgefundenes Modell und nie einen fremden Prozess auf derselben Karte.
    # Ein Nachbarprozess, der die Karte mitbenutzt, bleibt unberuehrt -- der
    # Knopf verspricht "meins von dieser Karte", nicht "diese Karte leer".
    #
    # Der Server-Endpunkt /models/unload waere der elegantere Weg, existiert
    # aber nur im Router-Modus (--models-dir), den dieser Launcher nicht
    # startet. Geprueft an b11015 und am thetom-Build, 2026-09-17.

    def _gpu_slots(self, gpu_index: int) -> List[str]:
        """Return this launcher's live slots running on one GPU.

        Args:
            gpu_index: Zero-based CUDA index, as in the ``"GPU 1"`` key.

        Returns:
            Slot names, primary first, for servers this launcher started on
            that GPU. Empty when nothing of ours runs there. A CPU server
            never matches: there is no VRAM to give back.
        """
        wanted = f"GPU {gpu_index}"
        treffer: List[str] = []
        for name in (SLOT_PRIMARY, SLOT_VISION):
            inst = self.slot(name)
            if inst.is_alive() and inst.gpu_key == wanted:
                treffer.append(name)
        return treffer

    def _unload_gpu(self, gpu_index: int) -> bool:
        """Stop everything this launcher runs on one GPU.

        The proof is the process, not the signal: each stop path already
        confirms the process is gone before clearing its slot, and this
        method reports failure rather than claiming the memory is back.

        Args:
            gpu_index: Zero-based CUDA index.

        Returns:
            True when every matching slot was confirmed stopped.
        """
        slots = self._gpu_slots(gpu_index)
        if not slots:
            self._log(f"Unload GPU {gpu_index}: nothing of ours is running "
                      f"there.", "info")
            return True
        ok = True
        for name in slots:
            if name == SLOT_VISION:
                ok = bool(self._stop_vision_server()) and ok
            else:
                ok = bool(self._stop_server()) and ok
        if ok:
            self._log(f"Unload GPU {gpu_index}: released "
                      f"{len(slots)} server(s).", "info")
        else:
            self._log(f"Unload GPU {gpu_index}: at least one server did not "
                      f"confirm its stop; VRAM may still be held.", "error")
        self._refresh_unload_buttons()
        return ok

    def _refresh_unload_buttons(self) -> None:
        """Enable each GPU's Unload button only while it has something to do.

        A button that cannot act must not look as if it could -- the same
        reasoning as AP-QL-105's port dialog, which omits a stop button it
        could not honour.
        """
        for gpu_index, btn in getattr(self, "_unload_btns", {}).items():
            try:
                btn.configure_btn(
                    state="normal" if self._gpu_slots(gpu_index)
                    else "disabled")
            except Exception:
                pass

    def _stop_vision_server(self, silent: bool = False) -> Optional[bool]:
        """Stop the vision server and clear its slot only after process exit."""
        inst = self.slot(SLOT_VISION)
        if inst.process is None:
            return None
        if not silent:
            self._log(f"Vision: stopping {inst.model or 'server'}...", "warn")
        proc = inst.process
        self._terminate_process(
            proc, None if silent else (lambda m: self._log(m, "warn")))
        try:
            still_alive = proc.poll() is None
        except Exception:
            still_alive = True
        if still_alive:
            if not silent:
                self._log("Vision: stop not confirmed; process remains "
                          "tracked.", "error")
            self._refresh_vision_widgets()
            return False
        self._servers.pop(SLOT_VISION, None)
        if not silent:
            self._log("Vision: stopped.", "info")
        self._refresh_vision_widgets()
        return True

    def _toggle_vision_server(self) -> None:
        """The one control: start when idle, stop when running."""
        if self.slot(SLOT_VISION).is_alive():
            self._stop_vision_server()
        else:
            self._start_vision_server()

    def _vision_status_text(self) -> str:
        """One line describing the vision slot, for the button and the log."""
        inst = self.slot(SLOT_VISION)
        if not inst.is_alive():
            name = (self.cfg.get(ROLE_VISION, "") or "").strip()
            return (f"{ROLE_GLYPHS[ROLE_VISION]} Start Vision"
                    if name else f"{ROLE_GLYPHS[ROLE_VISION]} Vision: no role")
        kurz = (inst.model or "")[:-5] if (inst.model or "").lower().endswith(
            ".gguf") else (inst.model or "")
        if len(kurz) > 16:
            kurz = kurz[:15] + "\u2026"
        return f"{ROLE_GLYPHS[ROLE_VISION]} Stop Vision :{inst.port}"

    def _refresh_vision_widgets(self) -> None:
        """Re-label the vision control to match the slot."""
        btn = getattr(self, "_vision_button", None)
        if btn is None:
            return
        laeuft = self.slot(SLOT_VISION).is_alive()
        try:
            btn.configure_btn(
                text=self._vision_status_text(),
                color=self.theme.red if laeuft else self.theme.button_bg)
        except Exception:
            pass

    def _own_server_pids(self) -> List[int]:
        """Return the pids of the servers this launcher instance started."""
        pids: List[int] = []
        for inst in getattr(self, "_servers", {}).values():
            proc = getattr(inst, "process", None)
            if proc is None:
                continue
            try:
                if proc.poll() is None:
                    pids.append(proc.pid)
            except Exception:
                continue
        return pids

    def _port_conflict_report(self, port: int) -> Dict[str, object]:
        """Collect everything the port-conflict dialog needs to say.

        Separated from the widgets so the decision — what may be
        offered — can be tested without opening a window.

        Args:
            port: The port the user tried to start on.

        Returns:
            The occupant dict from _port_occupant(), extended by
            ``headline``, ``rows`` (label/value pairs), ``note``,
            ``can_stop`` and ``alt_port``.
        """
        occ = dict(_port_occupant(port))
        pid = occ.get("pid")
        can_stop = bool(occ.get("stoppable"))
        note = ""

        if pid is not None and pid == os.getpid():
            can_stop = False
            note = ("This is the QLauncher window you are looking at. Use "
                    "Stop instead of killing it.")
        elif pid is not None and pid in self._own_server_pids():
            can_stop = False
            note = ("This is a server started by this window. Use Stop in "
                    "the header — it also frees the VRAM.")
        elif occ.get("kind") == OCCUPANT_HIDDEN:
            note = (f"Stopping it needs root: "
                    f"`sudo ss -ltnp 'sport = :{port}'` names it, and the "
                    f"owning service has to be stopped its own way.")
        elif occ.get("kind") == OCCUPANT_QLAUNCHER:
            note = ("Killing a launcher would orphan the servers it owns. "
                    "Close it in its own window.")
        elif occ.get("kind") == OCCUPANT_LAUNCHER_CHILD:
            note = ("Another QLauncher owns this server — stopping it here "
                    "takes the model away from whoever is using it.")

        rows: List[tuple] = []
        if pid is not None:
            rows.append(("Process", f"{occ.get('name') or 'unknown'} "
                                    f"(PID {pid})"))
        if occ.get("model"):
            rows.append(("Model", str(occ["model"])))
        if occ.get("detail"):
            rows.append(("Origin", str(occ["detail"])))
        if occ.get("user"):
            rows.append(("Owner", str(occ["user"])))

        if occ.get("kind") == OCCUPANT_NONE:
            headline = (f"Port {port} looked busy, but nothing is listening "
                        f"on it now.")
        elif pid is None:
            headline = f"Port {port} is in use by a process we cannot see."
        else:
            headline = f"Port {port} is in use by {occ.get('name') or 'a process'}."

        occ.update({
            "headline": headline,
            "rows": rows,
            "note": note,
            "can_stop": can_stop,
            "alt_port": _next_free_port(
                port + 1, avoid=[self._ipc_port] if self._ipc_port else None),
        })
        return occ

    def _ask_port_conflict(self, port: int) -> Dict[str, object]:
        """Show who holds the port and let the owner decide what to do.

        Replaces the v0.57 yes/no box, which could only guess out loud.
        Measured on 2026-09-16: the occupant was ReachyMini's own
        llama-server, so "kill it" was the wrong default — the dialog
        therefore offers a free port with equal weight.

        Args:
            port: The port the user tried to start on.

        Returns:
            ``{"action": "stop" | "switch" | "anyway" | "cancel",
               "port": int, "report": dict}``.
        """
        report = self._port_conflict_report(port)
        t = self.theme
        result: Dict[str, object] = {"action": "cancel", "port": port,
                                     "report": report}

        dlg = tk.Toplevel(self)
        dlg.title("Port Already in Use")
        dlg.configure(bg=t.bg)
        dlg.transient(self)
        dlg.grab_set()

        tk.Label(dlg, text=str(report["headline"]), font=FONT_TITLE,
                 bg=t.bg, fg=t.fg, wraplength=560, justify="left").pack(
                     padx=20, pady=(14, 8), anchor="w")
        tk.Frame(dlg, height=1, bg=t.border).pack(fill="x", padx=16,
                                                  pady=(0, 10))

        for label, value in report["rows"]:            # type: ignore[union-attr]
            row = tk.Frame(dlg, bg=t.bg)
            row.pack(fill="x", padx=20, pady=1)
            tk.Label(row, text=label, font=FONT_BODY_B, bg=t.bg,
                     fg=t.fg_secondary, width=9, anchor="w").pack(side="left")
            tk.Label(row, text=value, font=FONT_BODY, bg=t.bg, fg=t.fg,
                     wraplength=460, justify="left").pack(side="left")

        if report["note"]:
            tk.Label(dlg, text=str(report["note"]), font=FONT_BODY,
                     bg=t.bg, fg=t.yellow, wraplength=560,
                     justify="left").pack(padx=20, pady=(10, 0), anchor="w")

        alt_port = report["alt_port"]
        if report["kind"] == OCCUPANT_NONE:
            # The probe and this dialog are moments apart. A socket in
            # TIME_WAIT, or an occupant that just exited, lands here —
            # saying "the other process" would name something that is no
            # longer there. Seen on a real run while testing.
            guidance = ("It was busy when the check ran a moment ago. "
                        "Starting normally should work now.")
        else:
            guidance = ("Starting llama-server on a busy port fails with a "
                        "bind error a few seconds in.")
            if alt_port:
                guidance += (f" Port {alt_port} is free — starting there "
                             f"leaves the other process running.")
        tk.Label(dlg, text=guidance, font=FONT_BODY, bg=t.bg, fg=t.fg_dim,
                 wraplength=560, justify="left").pack(padx=20, pady=(10, 4),
                                                      anchor="w")

        btn_frame = tk.Frame(dlg, bg=t.bg)
        btn_frame.pack(fill="x", padx=20, pady=(12, 14))

        def _choose(action: str, chosen: int) -> None:
            result["action"] = action
            result["port"] = chosen
            dlg.destroy()

        def _fits(label: str, minimum: int = 90) -> int:
            """Width a HoverButton needs for a label, measured not guessed.

            The canvas does not shrink its text: "Start on 11435" was
            clipped to "tart on 1143" at a hand-picked 130 px, and the
            width that fits depends on the port's digit count and on
            which mono font the platform picked.
            """
            try:
                import tkinter.font as _tkfont
                return max(minimum,
                           _tkfont.Font(font=FONT_SMALL_B).measure(label) + 24)
            except Exception:
                return max(minimum, 16 + 9 * len(label))

        # AP-QL-68: the acting buttons sit bottom right. With side="right"
        # the FIRST packed widget ends up rightmost, so Cancel comes first.
        HoverButton(btn_frame, t, text="Cancel", color=t.border,
                    width=_fits("Cancel"), height=28,
                    command=lambda: _choose("cancel", port)).pack(side="right")
        if alt_port:
            HoverButton(btn_frame, t, text=f"Start on {alt_port}",
                        color=t.accent, width=_fits(f"Start on {alt_port}"),
                        height=28,
                        command=lambda: _choose("switch", int(alt_port))).pack(
                            side="right", padx=8)
        if report["can_stop"]:
            # "Stop it and start" was clipped to "top it and star" at
            # width 150 on the first real run — HoverButton draws on a
            # fixed-width canvas and does not shrink its label.
            HoverButton(btn_frame, t, text="Stop and start", color=t.red,
                        width=_fits("Stop and start"), height=28,
                        command=lambda: _choose("stop", port)).pack(
                            side="right", padx=(0, 8))
        # The v0.57 override stays reachable: a port in TIME_WAIT is busy
        # for the probe and free for the real bind moments later.
        HoverButton(btn_frame, t, text="Start anyway", color=t.border,
                    width=_fits("Start anyway"), height=28,
                    command=lambda: _choose("anyway", port)).pack(side="left")

        # The height has to be settled HERE: winfo_reqheight() is
        # evaluated before _center_window() runs its own
        # update_idletasks(), so an unsettled dialog gets sized from a
        # stale request and the button row lands outside the window.
        # Seen on the first real run of this dialog — 620x200 for
        # content that needed 330.
        dlg.update_idletasks()
        self._center_window(dlg, 620, dlg.winfo_reqheight())
        dlg.wait_window()
        return result

    def _free_port_by_stopping(self, port: int,
                               report: Dict[str, object]) -> bool:
        """Stop the process holding a port and prove the port came free.

        The proof is the port, not the signal: a process that ignores
        SIGTERM and one that exits while a sibling keeps the socket look
        the same from the signal's side.

        Args:
            port: The port to free.
            report: The dict from _port_conflict_report().

        Returns:
            True only if the port binds freely afterwards.
        """
        pid = report.get("pid")
        if not isinstance(pid, int):
            self._log(f"Cannot stop the process on port {port} — its pid is "
                      f"not visible.", "error")
            return False
        self._log(f"Stopping PID {pid} ({report.get('name') or 'unknown'}) "
                  f"to free port {port}.", "warn")
        outcome = _kill_pid(pid)
        self._log(f"  {outcome['message']}.",
                  "warn" if outcome["ok"] else "error")
        # Deliberately no early return on a failed signal: the port is
        # the proof, not the signal. A process that ignored SIGTERM may
        # still have closed its socket, and one that died may have left
        # a sibling holding the port. Only the bind decides.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if _port_is_free(port):
                self._log(f"  Port {port} is free again.", "info")
                return True
            time.sleep(0.2)
        self._log(f"  PID {pid} is gone but port {port} is still held — "
                  f"something else is listening on it.", "error")
        return False

    def _terminate_process(self, proc, on_force=None) -> None:
        """End one server process: politely, then firmly, then by force.

        AP-QL-90 lifted this out of _stop_server() so the vision slot cannot
        grow a second, subtly different teardown. The escalation is the part
        worth keeping in one place — a Windows CTRL_BREAK, a POSIX terminate,
        a five second grace, then kill, then a last kill if even that raised.

        Args:
            proc: The Popen to end. A None is accepted and ignored, so
                callers do not each need their own guard.
            on_force: Optional callable(str) for the message when the polite
                path raised and force was needed.
        """
        if proc is None:
            return
        try:
            if sys.platform == "win32":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
        except Exception as e:
            if on_force is not None:
                on_force(f"Force kill: {e}")
            try:
                proc.kill()
            except Exception:
                pass

    def _cleanup_failed_server_start(
            self, slot_name: str, proc, owner_token: Optional[str],
            label: str) -> bool:
        """Clean a process whose post-Popen setup raised, without lying.

        Popen transfers real OS ownership before status widgets, config saves
        and log-reader startup run. If one of those later operations fails,
        the process is first adopted into its slot with the original lease,
        then terminated. A process that resists termination remains tracked;
        consequently LEASE_STATUS cannot become empty while it is alive.
        """
        inst = self.slot(slot_name)
        inst.process = proc
        inst.owner_token = owner_token
        self._terminate_process(
            proc, lambda message: self._log(message, "warn"))
        try:
            still_alive = proc.poll() is None
        except Exception:
            still_alive = True
        if still_alive:
            self._log(f"{label}: failed start cleanup not confirmed; "
                      "process remains tracked.", "error")
            return False
        if self.slot(slot_name).process is proc:
            self._servers[slot_name] = ServerInstance(role=inst.role)
        return True

    def _stop_server(self, silent=False) -> bool:
        t = self.theme
        if self.server_process:
            name = self.running_model or "server"
            if not silent:
                self._log(f"Stopping {name}...", "warn")
            proc = self.server_process
            self._terminate_process(
                proc,
                None if silent else (lambda m: self._log(m, "warn")))
            try:
                still_alive = proc.poll() is None
            except Exception:
                still_alive = True
            if still_alive:
                if not silent:
                    self._log("Server stop not confirmed; process remains "
                              "tracked.", "error")
                return False
            self.server_process = None
            self.running_model = None
            self._running_gpu_key = None
            self._running_kv = None
            self._running_la = None
            self.slot().owner_token = None

            # AP-QL-25: bewusst VOR dem silent-Zweig — auch ein stiller Stopp
            # (Neustart ueber Q/N/C) darf keinen aktiven Stop-Knopf hinterlassen.
            self._refresh_action_btns()   # AP-QL-25: nichts mehr zu stoppen
            if not silent:
                self._log("Server stopped.", "info")
            # Always reset the status pill and running dot — silent stops
            # (Probe, Q/N/C restart, model switch) otherwise leave a stale
            # "Running" state behind once no new server follows.
            self._update_status_idle()
            self._update_running_indicator()
        return True

    # ─── AP-QL-113 (2026-09-17): welche Karte es WIRKLICH wurde ─────────
    # ⚙ Owner: "überprüfe dann bei jedem Start um welche GPU es sich dann
    # exakt handelt."
    #
    # Der Anlass ist gemessen: `CUDA_VISIBLE_DEVICES=0` allein trifft auf
    # dieser Maschine die **4090**, nicht die 5090 -- ohne
    # `CUDA_DEVICE_ORDER=PCI_BUS_ID` gilt die CUDA-Voreinstellung
    # FASTEST_FIRST, und die ordnet anders als nvidia-smi. Der Launcher setzt
    # PCI_BUS_ID seit jeher, aber niemand hat je NACHGESEHEN. Ein
    # Messwerkzeug ohne die Variable hat am 17.09. zwei Wochen lang
    # 4090-Zahlen unter der Ueberschrift "RTX 5090" geliefert.
    #
    # ⚠ Gefragt wird die PID, NICHT der Serverlog. Der erste Entwurf las
    # "Device 0: <name>" aus der Ausgabe -- diese Zeile erscheint bei
    # `--list-devices` und im Smoke-Test, aber beim echten Serverstart steht
    # sie NICHT im Log (an einem realen Start nachgesehen, 26 Zeilen, kein
    # Treffer). Die Pruefung haette nie gefeuert. `nvidia-smi` beantwortet
    # dieselbe Frage unabhaengig von der Ausfuehrlichkeit des Servers.

    def _gpu_of_pid(self, pid: int) -> List[int]:
        """Ask the driver which GPUs a process actually occupies.

        Args:
            pid: The process id to look up.

        Returns:
            nvidia-smi indices holding that process, empty if none does. The
            indices match `GPUInfo.index`, because `detect_all_gpus()` reads
            the same tool in the same order.
        """
        treffer: List[int] = []
        for g in (self.gpus or []):
            if g.vendor != "nvidia":
                continue
            out = _run_cmd(["nvidia-smi", "-i", str(g.index),
                            "--query-compute-apps=pid",
                            "--format=csv,noheader"])
            for zeile in out.splitlines():
                zeile = zeile.strip()
                if zeile.isdigit() and int(zeile) == pid:
                    treffer.append(g.index)
                    break
        return treffer

    def _confirm_gpu_identity(self, slot_name: str = SLOT_PRIMARY) -> None:
        """Report which card this slot's server really landed on.

        Runs in a worker thread and reaches the UI only through the main
        queue -- no worker touches Tk (AP-QL-96/97/98). Silent when the
        process is gone or the driver has nothing to say; a start that
        crashed has its own message already.

        Args:
            slot_name: Which slot to confirm.
        """
        inst = self.slot(slot_name)
        proc = inst.process
        key = inst.gpu_key or ""
        praefix = "" if slot_name == SLOT_PRIMARY else "Vision: "
        if proc is None or key == "CPU":
            return
        # Der Prozess taucht beim Treiber erst auf, wenn er Speicher belegt.
        # Versuche/Pause sind Attribute, damit eine Suite nicht 20 s schlafen
        # muss, um den Fall "Treiber weiss nichts" zu pruefen.
        for _ in range(getattr(self, "_gpu_confirm_tries", 20)):
            if proc.poll() is not None:
                return
            belegt = self._gpu_of_pid(proc.pid)
            if belegt:
                break
            time.sleep(getattr(self, "_gpu_confirm_delay", 1.0))
        else:
            self._queue_main(
                self._log,
                f"{praefix}[GPU] konnte nicht bestaetigt werden — der "
                f"Treiber meldet PID {proc.pid} auf keiner Karte.", "warn")
            return
        namen = {g.index: g.name for g in (self.gpus or [])}
        gemeldet = ", ".join(f"GPU {i} ({namen.get(i, '?')})" for i in belegt)
        try:
            gewaehlt = int(key.split(" ")[1])
        except (IndexError, ValueError):
            self._queue_main(
                self._log, f"{praefix}[GPU] laeuft auf {gemeldet}.", "info")
            return
        if belegt == [gewaehlt]:
            self._queue_main(
                self._log,
                f"{praefix}[GPU] bestaetigt: {gemeldet}", "good")
        else:
            self._queue_main(
                self._log,
                f"{praefix}[GPU] ⚠ FALSCHE KARTE: gewaehlt war {key} "
                f"({namen.get(gewaehlt, '?')}), belegt wird {gemeldet}. "
                f"Pruefe CUDA_DEVICE_ORDER/CUDA_VISIBLE_DEVICES, bevor du "
                f"diesem Lauf eine Zahl entnimmst.", "error")

    def _read_server_output(self, slot_name: str = SLOT_PRIMARY):
        """Background thread reading one server's stdout.

        AP-QL-90: takes the slot it belongs to. Everything that reaches into
        the app's single status line — Loading → Running, install_verified —
        stays bound to the PRIMARY slot on purpose. The vision server has its
        own small status line and must not be able to overwrite the one the
        whole UI already talks about; and "the installation works" is a
        statement about the engine the user actually configured, not about a
        second process the launcher started on the side.
        """
        inst = self.slot(slot_name)
        proc = inst.process
        if not proc or not proc.stdout:
            return
        primaer = (slot_name == SLOT_PRIMARY)
        praefix = "" if primaer else "Vision: "
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
                        if primaer and ("listening" in ll
                                        or "all slots are idle" in ll):
                            fn  = self.running_model or ""
                            prt = getattr(self, "_running_port", "8080")
                            self._queue_main(
                                self._upgrade_status_to_running, fn, prt)
                    elif "warning" in ll:
                        tag = "warn"
                    if "all slots are idle" in ll:
                        tag = "good"
                        if primaer:
                            fn  = self.running_model or ""
                            prt = getattr(self, "_running_port", "8080")
                            self._queue_main(
                                self._upgrade_status_to_running, fn, prt)
                        else:
                            self._queue_main(self._refresh_vision_widgets)
                    # v0.56 — First time we see the server announce it's
                    # actually listening on its port, the installation is
                    # proven to work. This one-way flip unlocks the
                    # "Autoload" button for the user. Runs on the main
                    # thread to keep cfg writes single-threaded.
                    if (primaer and "listening" in ll
                            and not self.cfg.get("install_verified", False)):
                        self._queue_main(self._mark_install_verified)
                    if not primaer and "listening" in ll:
                        self._queue_main(self._refresh_vision_widgets)
                    # AP-QL-97: queue instead of one after() per line —
                    # this single line was the difference between 0/60
                    # and 9/12 segfaults.
                    self._queue_log(praefix + line, tag)
        except Exception:
            pass
        # AP-QL-90: Beim Beenden ist die Ereignisschleife weg, bevor der
        # Leser-Faden merkt, dass sein Rohr zu ist — `after()` wirft dann
        # "main thread is not in main loop" und der Faden stirbt mit einem
        # Rueckverfolgungsbericht auf der Konsole. Das war schon vorher so;
        # mit einem zweiten Server sind es jetzt zwei Faeden, die es treffen
        # kann. Ein Aufraeumruf, der beim Zumachen ins Leere geht, ist genau
        # der Fall, in dem es nichts mehr aufzuraeumen gibt.
        try:
            self._queue_main(self._on_server_exited, slot_name, proc)
        except Exception:
            pass

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

    def _on_server_exited(self, slot_name: str = SLOT_PRIMARY,
                          expected_process=None):
        # A reader belongs to one exact process generation. Its queued exit
        # callback may arrive after that slot has already been reused; never
        # erase the replacement process or its newer ownership capability.
        if (expected_process is not None
                and self.slot(slot_name).process is not expected_process):
            return
        if slot_name != SLOT_PRIMARY:
            inst = self.slot(slot_name)
            if inst.model:
                self._log(f"Vision: server process exited.", "warn")
                self._servers.pop(slot_name, None)
                self._refresh_vision_widgets()
            return
        if self.running_model:
            self._log(f"Server process exited.", "warn")
            self.running_model = None
            self._running_gpu_key = None
            self._running_kv = None
            self._running_la = None
            self._running_host = None   # AP-QL-32
            self.server_process = None
            self.slot().owner_token = None
            # AP-QL-73 (Fund C): ueber den Helfer statt roh. Die rohe
            # config()-Zeile setzte weder _status_state noch cursor und
            # Tooltip zurueck — starb der Server von selbst, blieb der
            # Zeiger auf "hand2" und der Tooltip "Click to open in
            # browser" stehen, obwohl nichts mehr lauschte.
            self._update_status_idle()
            self._update_running_indicator()
            self._refresh_action_btns()   # AP-QL-25

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION: Log
    # ═══════════════════════════════════════════════════════════════════════

    def _log_path_repairs(self) -> None:
        """Die Umzugs-Reparaturen vom Config-Laden melden (AP-QL-31).

        Es wird nichts still repariert: der Nutzer sieht jedes alt → neu
        Paar, damit eine falsche Neuverankerung sichtbar wird statt
        raetselhaft zu bleiben.
        """
        if not self._path_repairs:
            return
        self._log("Workspace scheint verschoben worden zu sein — "
                  "repariere tote Pfade:", "warn")
        for note in self._path_repairs:
            self._log(f"  {note}", "warn")
        # AP-QL-57: es sind zwei Anker (Workspace + eine Ebene darueber), also
        # werden auch beide genannt. Nur den ersten zu melden waere jetzt eine
        # Halbwahrheit — und gerade bei einer Reparatur, die aus der zweiten
        # Wurzel stammt, die irrefuehrendste Angabe von allen.
        anker = "  ·  ".join(str(r) for r in _workspace_search_roots())
        self._log(f"Config aktualisiert ({len(self._path_repairs)} Pfad(e)) — "
                  f"Anker: {anker}", "good")
        self._path_repairs = []

    def _log(self, text: str, tag: str = "info"):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_text.config(state="normal")
        self._log_text.insert("end", f"[{ts}] ", "time")
        self._log_text.insert("end", f"{text}\n", tag)
        self._log_text.see("end")
        self._log_text.config(state="disabled")

    #: AP-QL-97. How often the main thread drains the queue. 120 ms is
    #: below the eye's threshold for a log pane and turns a burst of
    #: hundreds of cross-thread calls into one timer tick.
    LOG_DRAIN_MS = 120
    #: Upper bound per tick, so a very chatty server cannot freeze the UI
    #: inside one drain. The rest follows on the next tick.
    LOG_DRAIN_MAX = 200

    def _start_log_drain(self) -> None:
        """Start the single periodic drain of the server-log queue."""
        if getattr(self, "_log_drain_running", False):
            return
        self._log_drain_running = True
        self._drain_log_queue()

    def _drain_log_queue(self) -> None:
        """Write queued server-log lines. MAIN THREAD ONLY.

        WHY THIS EXISTS — measured, 2026-08-30, on Linux with Tcl 9.0.
        The launcher segfaulted in 9 of 12 runs when MyIDE started a model,
        and in 0 of 60 runs without one. The only difference between those
        two populations is this reader: `_read_server_output` fired one
        `self.after(0, self._log, line, tag)` PER LINE, and llama-server
        prints hundreds of them in a burst while a model loads.

        Three crash signatures appeared across those 9 — Tcl's threaded
        allocator, an 80,587-frame recursion through UpdateStringOfList,
        and others — which is one corruption wearing different faces, not
        three bugs.

        ⚠ `after()` from a worker IS the sanctioned idiom and _tkinter
        marshals it; the defect is the RATE, not the direction. Every call
        registers a fresh Tcl command, hands it across the thread boundary
        and deletes it again. This turns that burst into one timer tick on
        the thread that owns the interpreter, so the reader thread makes no
        Tcl call at all any more.

        The queue is plain `queue.Queue`: stdlib, no dependency, and the
        only thing the reader touches.
        """
        if getattr(self, "_closing", False):
            self._log_drain_running = False
            return
        # AP-QL-98: Auftraege zuerst und vollstaendig — eine geschwaetzige
        # Log-Ausgabe darf ein SHUTDOWN nicht hinter sich warten lassen.
        self._drain_main_queue()
        posten = []
        try:
            for _ in range(self.LOG_DRAIN_MAX):
                posten.append(self._log_queue.get_nowait())
        except queue.Empty:
            pass
        except Exception:
            posten = []
        for text, tag in posten:
            try:
                self._log(text, tag)
            except Exception:
                # A dying interpreter must not take the drain with it.
                break
        try:
            self.after(self.LOG_DRAIN_MS, self._drain_log_queue)
        except Exception:
            self._log_drain_running = False

    def _queue_log(self, text: str, tag: str = "info") -> None:
        """Hand a log line to the main thread. Safe from any thread."""
        try:
            self._log_queue.put_nowait((text, tag))
        except Exception:
            pass

    def _queue_main(self, fn, *args, delay_ms: int = 0) -> bool:
        """Run ``fn(*args)`` on the main thread. Safe from any thread.

        AP-QL-98. Replaces ``self.after(...)`` at every site reached from a
        worker thread.

        WHY THIS EXISTS — measured, 2026-08-31, Linux, at a minimal example
        with no launcher involved:

            python3   3.14.4 / Tcl 8.6.17 -> a worker's after() FIRES
            venv      3.12.13 / Tcl 9.0.3 -> it is accepted without raising
                                             and NEVER FIRES

        The main loop keeps running its own after() chains throughout, so
        nothing looks wrong. What broke, measured through the CLI:

            --release  exit 1 "model lease release was not confirmed"
                       forever; owned_slots stayed ["primary"] because
                       _release_model_lease never ran and the server was
                       never stopped
            --shutdown "close was not confirmed"; the instance could not be
                       closed at all
            AUTOSTART  skipped as "already running", because
                       _on_server_exited never ran and this launcher still
                       believed its dead server was alive

        MyIDE hit all of it because it starts the launcher with its own
        venv interpreter (Tcl 9.0.3); a start through start_qlauncher.sh
        uses python3 (Tcl 8.6.17) and worked. That is why it only ever
        appeared "while MyIDE drives the launcher".

        ⚠ This is the same shape AP-QL-97 introduced for log lines, and it
        is not a duplicate of it: that one carries text and is capped per
        tick, this one carries callables and is drained in full. Commands
        are rare; a burst of log lines must never delay a SHUTDOWN.

        Args:
            fn: Callable to run on the main thread.
            *args: Its positional arguments.
            delay_ms: Kept for the two sites that had a deliberate delay
                (100 ms so an IPC reply can flush, 50 ms so the GUI can
                settle). The drain re-schedules those with ``after()``
                FROM THE MAIN THREAD, where it works — dropping the delay
                would be a silent behaviour change.

        Returns:
            True if the job was queued. False only if the queue itself
            refused it; the ACQUIRE path uses that to retire its pending
            lease instead of leaving it dangling.
        """
        try:
            self._main_queue.put_nowait((fn, args, int(delay_ms)))
            return True
        except Exception:
            return False

    def _drain_main_queue(self) -> None:
        """Run everything workers have queued. MAIN THREAD ONLY.

        Drained in full rather than capped: these are state transitions
        (a server exited, a lease was released, the app was asked to
        quit), not a stream. Called from _drain_log_queue so there is one
        timer, not two — but kept as its own method so AP-QL-97's drain
        keeps its name and its suite keeps its anchors.
        """
        auftraege = []
        try:
            while True:
                auftraege.append(self._main_queue.get_nowait())
        except queue.Empty:
            pass
        except Exception:
            auftraege = []
        for fn, args, delay_ms in auftraege:
            try:
                if delay_ms > 0:
                    self.after(delay_ms, fn, *args)
                else:
                    fn(*args)
            except Exception:
                # One failing job must not take the drain — and with it
                # every later RELEASE and SHUTDOWN — down with it.
                try:
                    self._log("A queued main-thread job failed.", "error")
                except Exception:
                    pass

    def _clear_log(self):
        self._log_text.config(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.config(state="disabled")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION: Scanning & Initialization
    # ═══════════════════════════════════════════════════════════════════════

    def _heal_active_server_path(self) -> None:
        """On startup, if cfg['llama_server_path'] doesn't point to a real
        llama-server binary but a slot does, silently auto-switch. Prevents
        the user getting stuck in a broken-config loop (e.g. active path
        still pointing at start_qlauncher.sh from an earlier mishap)."""
        active = (self.cfg.get("llama_server_path") or "").strip()
        bad_suffixes = (".sh", ".bat", ".py", ".json", ".md", ".txt")
        def _ok(path: str) -> bool:
            if not path or not os.path.isfile(path):
                return False
            n = os.path.basename(path).lower()
            if n.endswith(bad_suffixes):
                return False
            return "llama-server" in n or n == exe_name("llama-server")
        if _ok(active):
            return
        for cand in (self.cfg.get("server_slots") or []):
            cand = (cand or "").strip()
            if _ok(cand):
                self.cfg["llama_server_path"] = cand
                save_config(self.cfg)
                self._log(
                    f"Startup heal: active llama-server path was invalid — "
                    f"auto-switched to: {slot_label_from_path(cand) or cand}",
                    "warn")
                return

    def _do_initial_scan(self):
        self._heal_active_server_path()
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
                # AP-QL-36 (v0.65): kein VRAM in der Kopfzeile.
                parts.append(f"GPU {g.index}: {short_gpu_name(g.name)}")   # AP-QL-78
            if self._cpu_ram_gb > 0:
                parts.append(f"CPU RAM: {self._cpu_ram_gb:.0f} GB")
            self._gpu_label.config(text="  •  ".join(parts))
        else:
            ram_info = f" — CPU RAM: {self._cpu_ram_gb:.0f} GB" if self._cpu_ram_gb > 0 else ""
            self._gpu_label.config(text=f"No GPUs detected — CPU mode only{ram_info}")

        self._build_gpu_buttons()
        self._build_unload_buttons()   # AP-QL-111

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
        # AP-QL-48: keine Build-Angabe — siehe Begruendung in _quick_refresh.
        if dll_total > 0:
            if dll_found == dll_total:
                cuda_parts.append(f"{RUNTIME_LIB_LABEL}: {dll_found}/{dll_total} ✓")
                dll_color = t.fg_secondary
            else:
                cuda_parts.append(f"{RUNTIME_LIB_LABEL}: {dll_found}/{dll_total} ✗ Missing: {', '.join(dll_missing)}")
                dll_color = t.yellow
        else:
            dll_color = t.fg_dim
            cuda_parts.append(f"{RUNTIME_LIB_LABEL}: server path not set")

        # AP-QL-36 (v0.65): fuehrender Trenner — setzt die GPU/RAM-Zeile fort.
        self._cuda_label.config(text="  •  " + "  •  ".join(cuda_parts),
                                foreground=dll_color)

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

        # AP-QL-75 (Fund A2) / AP-QL-01: re-gate the KV preset buttons for
        # the (possibly new) server family. Runs on startup, Quick-Switch
        # and Paths save — all three paths route through _do_initial_scan.
        self._apply_kv_family_gating()

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
        owner_token = self._autostart_owner_token
        self._autostart_owner_token = None
        # AP-QL-92: same one-shot handling as the token above — read, clear,
        # then bind into the callback, so the wish cannot survive into a
        # second scan.
        want_model = self._autostart_model
        want_gpu = self._autostart_gpu
        self._autostart_model = None
        self._autostart_gpu = None
        self.after_idle(
            lambda token=owner_token, m=want_model, g=want_gpu:
                self._run_pending_autoload(token, want_model=m, want_gpu=g))

    def _rescan_models(self):
        paths = [p for p in (self.cfg.get("llm_models_paths") or []) if p]
        recursive = bool(self.cfg.get("llm_models_recursive", True))
        self.models = scan_models(paths, recursive=recursive)
        self.installed_models = scan_non_gguf_models(paths, recursive=recursive)
        self._refresh_models_header()
        self._rebuild_model_cards()
        if not paths:
            self._log("Scan: no model directories configured.", "warn")
        else:
            mode = "recursive" if recursive else "top-level"
            self._log(f"Scan ({mode}): {len(self.models)} GGUF models, "
                      f"{len(self.installed_models)} other installed models in "
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
        cb_rec = themed_checkbutton(rec_row,
                                     t,
                                     text="Include subdirectories (recursive scan)",
                                     variable=recursive_var,
                                     font=FONT_BODY)
        cb_rec.pack(side="left")

        # ── Quick-switch slots ──
        tk.Frame(dlg, height=1, bg=t.border).pack(fill="x", padx=16, pady=(4, 8))
        tk.Label(dlg, text="Quick-Switch Slots:",
                 font=FONT_BODY_B, bg=t.bg, fg=t.fg).pack(padx=20, anchor="w")
        tk.Label(dlg,
                 text=f"Bookmark {exe_name('llama-server')} builds here — they "
                      "appear as one-click buttons in the footer. Click a "
                      "button to activate that engine.",
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
            # v0.55: active llama_server_path comes from cfg (set via slot
            # click), not from a UI field anymore. We still validate it's
            # a real binary and try to auto-rescue from slots if it's broken
            # (e.g. file deleted, +x bit lost after rsync).
            active = (self.cfg.get("llama_server_path") or "").strip()
            slots_clean = [v.get().strip() for v in slot_vars]
            # AP-QL-93: one build, one slot. Cleared here rather than
            # merely hidden, so the stored list stops carrying a state the
            # footer cannot render honestly.
            slots_clean, slot_dupes = dedupe_slot_paths(slots_clean)
            for slot_no, dupe_label, kept_no in slot_dupes:
                self._log(
                    f"Slot {slot_no} pointed at the same build as slot "
                    f"{kept_no} ({dupe_label}) — cleared it. Two slots "
                    "sharing one binary light up together and look like "
                    "two active engines.", "warn")
            def _looks_like_llama_server(path: str) -> bool:
                if not path or not os.path.isfile(path):
                    return False
                name = os.path.basename(path).lower()
                # Reject obviously wrong things regardless of OS.
                bad_suffixes = (".sh", ".bat", ".py", ".json", ".md", ".txt")
                if name.endswith(bad_suffixes):
                    return False
                # Filename sanity: either the POSIX binary name, the
                # Windows variant, or any name containing 'llama-server'
                # (covers symlinks and custom suffixes like -madreag).
                name_ok = ("llama-server" in name
                           or name == exe_name("llama-server"))
                if not name_ok:
                    return False
                # On POSIX, also require the executable bit. A file
                # named llama-server with no +x set is a broken copy
                # (common after a failed rsync or archive extract) and
                # would silently fail at start time.
                if not IS_WINDOWS:
                    if not os.access(path, os.X_OK):
                        return False
                return True
            if not _looks_like_llama_server(active):
                # Try to rescue from slots
                for cand in slots_clean:
                    if _looks_like_llama_server(cand):
                        self._log(
                            f"Active llama-server path invalid ('{os.path.basename(active) or '<empty>'}') — "
                            f"auto-switched to slot: {slot_label_from_path(cand) or cand}",
                            "warn")
                        active = cand
                        break
            # If the user removed the slot whose path was active, log a
            # hint — engine still works, but no slot button will be
            # highlighted until they click one.
            # AP-QL-78 (Delta-Audit 03.08., Fund F): die zwei Faelle wieder
            # getrennt. Schlaegt die Slot-Rettung darueber fehl, fiel der
            # Code bisher in denselben Zweig und sagte "It still works" —
            # ueber eine Engine, deren Datei gar nicht existiert. Diese
            # Halbwahrheit schickt den Benutzer auf die Suche nach einem
            # Aktualisierungsfehler, statt ihm zu sagen, was zu tun ist.
            if active:
                slot_norms = {os.path.normcase(os.path.normpath(s))
                              for s in slots_clean if s}
                active_norm = os.path.normcase(os.path.normpath(active))
                if not os.path.isfile(active):
                    self._log(
                        "Active engine points at a file that does not exist: "
                        f"{active}", "error")
                    self._log(
                        "  Click a Quick-Switch slot button in the footer to "
                        "activate a build that is really there.", "warn")
                elif active_norm not in slot_norms:
                    self._log(
                        "Note: active engine is no longer bookmarked in any "
                        "slot. It still works, but no slot button will be "
                        "highlighted until you click one.", "info")
            self.cfg["llama_server_path"] = active
            # persist slot paths (strip whitespace, keep empty slots)
            self.cfg["server_slots"] = slots_clean
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

        # AP-QL-68 (F1): Aktionsknoepfe unten RECHTS, wie im neuen Picker
        # (AP-QL-62) und wie unter Windows ueblich. Cancel wird zuerst
        # gepackt und liegt damit ganz rechts; Save landet links davon —
        # die sichtbare Reihenfolge [Save][Cancel] bleibt also erhalten.
        HoverButton(btn_frame, t, text="Cancel", color=t.border,
                    width=100, height=28, command=_cancel).pack(side="right")
        HoverButton(btn_frame, t, text="Save", color=t.accent,
                    width=100, height=28, command=_save).pack(side="right", padx=8)

        dlg.update_idletasks()
        self._center_window(dlg, 1024, dlg.winfo_reqheight())

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION: Themed File/Directory Pickers (Tk custom)
    # ═══════════════════════════════════════════════════════════════════════
    # Tk's built-in filedialog is unthemeable — it uses a mix of tk::entry,
    # ttk::combobox, custom IconList canvas, and xmfbox widgets that each
    # ignore a different theming mechanism. Even with full option_add +
    # ttk.Style coverage, some parts still render white. We sidestep the
    # whole mess by shipping our own minimal pickers built from Tk
    # primitives we already style.

    def _themed_picker(self, initial: str, title: str = "Open",
                       pick_files: bool = False,
                       filename_patterns: Optional[List[str]] = None,
                       preselect: str = "") -> str:
        """Detail-view file/directory picker (AP-QL-62).

        Commander-style table with Name / Ext / Size / Date columns,
        click-to-sort headers, a pinned [..] row and directories-first
        grouping. The plain Listbox of the first picker generation was
        unusable for Windows users (owner screenshot, 2026-08-03). All
        labels are English (owner rule, 2026-08-03).

        pick_files=False  -> returns a directory path or ""
        pick_files=True   -> returns a file path or ""
        filename_patterns -> list of fnmatch-style patterns to filter files
                             (e.g. ['llama-server*']). None = show all.
        preselect         -> file name to highlight after the first fill
                             (slot browse: the currently configured binary).

        Landing-zone rule (owner, 2026-08-03): whenever the start folder is
        missing or anything else goes wrong, the picker opens in
        LAUNCHER_DIR — the folder this program was started from. Never the
        user's home directory, never a temp directory.
        """
        import fnmatch
        import tkinter.font as _tkfont
        t = self.theme

        dlg = tk.Toplevel(self)
        dlg.title(title)
        dlg.configure(bg=t.bg)
        dlg.transient(self)
        dlg.grab_set()

        state = {
            "result": "",
            # Landing zone: LAUNCHER_DIR — never the user's home directory
            # and never a temp folder (AP-QL-62, owner rule).
            "current": os.path.abspath(
                initial if initial and os.path.isdir(initial)
                else str(LAUNCHER_DIR)),
            "entries": [],            # cached listing of the current folder
            "error": "",              # OSError text when unreadable
            "sort": ("name", False),  # (column, descending)
            "rows": {},               # tree iid -> (kind, absolute path)
        }

        # ── Top row: path entry + Up button ──────────────────────────────
        top = tk.Frame(dlg, bg=t.bg)
        top.pack(fill="x", padx=10, pady=(10, 4))
        tk.Label(top, text="Path:", font=FONT_BODY_B,
                 bg=t.bg, fg=t.fg).pack(side="left")
        path_var = tk.StringVar(value=state["current"])
        path_entry = tk.Entry(top, textvariable=path_var, font=FONT_BODY,
                               bg=t.entry_bg, fg=t.fg,
                               insertbackground=t.fg,
                               selectbackground=t.accent, selectforeground="#ffffff",
                               relief="flat", bd=2, highlightthickness=1,
                               highlightbackground=t.border,
                               highlightcolor=t.accent)
        path_entry.pack(side="left", fill="x", expand=True, padx=(6, 6))
        HoverButton(top, t, text="↑ Up", color=t.border, width=60, height=24,
                    command=lambda: _go_to(os.path.dirname(state["current"]))).pack(side="left")

        # ── Detail table (ttk.Treeview with sortable columns) ────────────
        # Treeview instead of the old Listbox: real columns and per-column
        # sorting. The app switched ttk to the "clam" theme at startup, so
        # the private style below takes full effect on both platforms. Row
        # height follows the rendered font so Ctrl +/- zoom stays readable.
        style = ttk.Style(dlg)
        body_f = _tkfont.Font(root=dlg, font=FONT_BODY)
        row_h = body_f.metrics("linespace") + 6
        style.configure("Picker.Treeview",
                        background=t.bg_secondary,
                        fieldbackground=t.bg_secondary,
                        foreground=t.fg,
                        borderwidth=0, relief="flat",
                        rowheight=row_h, font=FONT_BODY)
        style.map("Picker.Treeview",
                  background=[("selected", t.accent)],
                  foreground=[("selected", "#ffffff")])
        style.configure("Picker.Treeview.Heading",
                        background=t.bg, foreground=t.fg,
                        borderwidth=0, relief="flat", font=FONT_BODY_B)
        style.map("Picker.Treeview.Heading",
                  background=[("active", t.bg)])

        list_frame = tk.Frame(dlg, bg=t.bg)
        list_frame.pack(fill="both", expand=True, padx=10, pady=4)
        scroll = ttk.Scrollbar(list_frame, orient="vertical",
                                style="Vertical.TScrollbar")
        scroll.pack(side="right", fill="y")
        COLS = ("name", "ext", "size", "date")
        tree = ttk.Treeview(list_frame, columns=COLS, show="headings",
                            style="Picker.Treeview", selectmode="browse",
                            yscrollcommand=scroll.set)
        tree.pack(side="left", fill="both", expand=True)
        scroll.config(command=tree.yview)

        HEAD = {"name": "Name", "ext": "Ext", "size": "Size", "date": "Date"}
        tree.column("name", anchor="w", minwidth=200, width=380, stretch=True)
        tree.column("ext", anchor="w", minwidth=44, width=64, stretch=False)
        tree.column("size", anchor="e", minwidth=80, width=110, stretch=False)
        tree.column("date", anchor="w", minwidth=110, width=140, stretch=False)

        def _headings():
            col, desc = state["sort"]
            for c in COLS:
                arrow = ("  ▼" if desc else "  ▲") if c == col else ""
                tree.heading(c, text=HEAD[c] + arrow,
                             command=lambda c=c: _sort_by(c))

        def _sort_by(col: str):
            cur, desc = state["sort"]
            state["sort"] = (col, not desc if col == cur else False)
            _fill()

        # ── Table content ────────────────────────────────────────────────
        def _scan():
            """Read the current folder into state['entries']."""
            state["entries"] = []
            state["error"] = ""
            cur = state["current"]
            path_var.set(cur)
            try:
                names = os.listdir(cur)
            except OSError as e:
                state["entries"] = None      # unreadable folder
                state["error"] = str(e)
                return
            for name in names:
                if name.startswith("."):
                    continue                 # hide dotfiles/dirs
                full = os.path.join(cur, name)
                try:
                    st = os.stat(full)
                except OSError:
                    continue                 # broken link etc.: skip
                is_dir = os.path.isdir(full)
                if not is_dir:
                    if not pick_files:
                        continue             # folder mode lists folders only
                    if filename_patterns and not any(
                            fnmatch.fnmatch(name, p)
                            for p in filename_patterns):
                        continue
                stem, dot, ext = name.rpartition(".")
                if is_dir or not dot or not stem:
                    stem, ext = name, ""
                state["entries"].append({
                    "name": name, "stem": stem, "ext": ext,
                    "is_dir": is_dir, "size": st.st_size,
                    "mtime": st.st_mtime, "full": full})

        def _fmt_date(mtime: float) -> str:
            try:
                return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            except (OverflowError, OSError, ValueError):
                return ""

        def _fill(select_name: str = ""):
            tree.delete(*tree.get_children())
            state["rows"] = {}
            _headings()
            cur = state["current"]
            parent = os.path.dirname(cur)
            if parent and parent != cur:
                iid = tree.insert("", "end",
                                  values=("[..]", "", "<DIR>", ""))
                state["rows"][iid] = ("up", parent)
            if state["entries"] is None:
                tree.insert("", "end",
                            values=(f"[error: {state['error']}]", "", "", ""))
                return
            col, desc = state["sort"]
            key = {
                "name": lambda e: e["name"].lower(),
                "ext":  lambda e: (e["ext"].lower(), e["name"].lower()),
                "size": lambda e: e["size"],
                "date": lambda e: e["mtime"],
            }[col]
            dirs = sorted((e for e in state["entries"] if e["is_dir"]),
                          key=key, reverse=desc)
            files = sorted((e for e in state["entries"] if not e["is_dir"]),
                           key=key, reverse=desc)
            chosen = None
            for e in dirs + files:
                if e["is_dir"]:
                    vals = ("[" + e["name"] + "]", "", "<DIR>",
                            _fmt_date(e["mtime"]))
                else:
                    vals = (e["stem"], e["ext"], f"{e['size']:,}",
                            _fmt_date(e["mtime"]))
                iid = tree.insert("", "end", values=vals)
                state["rows"][iid] = ("dir" if e["is_dir"] else "file",
                                      e["full"])
                if select_name and e["name"] == select_name:
                    chosen = iid
            if chosen is None:
                kids = tree.get_children()
                chosen = kids[0] if kids else None
            if chosen:
                tree.selection_set(chosen)
                tree.focus(chosen)
                tree.see(chosen)

        # ── Navigation ───────────────────────────────────────────────────
        def _go_to(new_path: str, select_name: str = ""):
            if new_path and os.path.isdir(new_path):
                state["current"] = os.path.abspath(new_path)
                _scan()
                _fill(select_name)

        def _selected():
            sel = tree.selection()
            return state["rows"].get(sel[0]) if sel else None

        def _commit(path: str):
            state["result"] = path
            dlg.destroy()

        def _activate(_evt=None):
            hit = _selected()
            if not hit:
                return
            kind, path = hit
            if kind == "up":
                # Land on the folder we come from, Commander-style.
                _go_to(path, select_name=os.path.basename(state["current"]))
            elif kind == "dir":
                _go_to(path)
            elif kind == "file" and pick_files:
                _commit(path)
            return "break"

        def _on_ok():
            hit = _selected()
            if hit:
                kind, path = hit
                if kind == "file" and pick_files:
                    _commit(path)
                    return
                if kind in ("dir", "up") and not pick_files:
                    # Folder mode: take the highlighted directory; on the
                    # [..] row take the folder we are looking at.
                    _commit(path if kind == "dir" else state["current"])
                    return
                if kind in ("dir", "up") and pick_files:
                    _activate()
                    return
            typed = path_var.get().strip()
            if pick_files:
                if typed and os.path.isfile(typed):
                    _commit(typed)
            else:
                if typed and os.path.isdir(typed):
                    _commit(typed)
                elif os.path.isdir(state["current"]):
                    _commit(state["current"])

        def _on_path_enter(_evt=None):
            typed = path_var.get().strip()
            if os.path.isdir(typed):
                _go_to(typed)

        # ── Button row ───────────────────────────────────────────────────
        btn_row = tk.Frame(dlg, bg=t.bg)
        btn_row.pack(fill="x", padx=10, pady=(4, 10))
        ok_label = "Open" if pick_files else "Select Folder"
        ok_btn = HoverButton(btn_row, t, text=ok_label, color=t.accent,
                              width=130, height=28, command=_on_ok)
        ok_btn.pack(side="right", padx=(6, 0))
        HoverButton(btn_row, t, text="Cancel", color=t.border,
                    width=100, height=28,
                    command=dlg.destroy).pack(side="right")

        tree.bind("<Double-Button-1>", _activate)
        tree.bind("<Return>", _activate)
        path_entry.bind("<Return>", _on_path_enter)
        dlg.bind("<Escape>", lambda _e: dlg.destroy())

        _scan()
        _fill(preselect)
        path_entry.icursor("end")
        tree.focus_set()

        # Size + center — four columns need more width than the old list.
        dlg.update_idletasks()
        w, h = 860, 520
        self._center_window(dlg, w, h)
        dlg.wait_window()
        return state["result"]

    def _picker_start_dir(self, *kandidaten) -> str:
        """Den ersten existierenden Ordner der Liste liefern.

        AP-QL-60. Vorher fielen beide Auswahlfenster auf ``~`` zurueck. Das
        ist auf dieser Maschine schon unpraktisch (der Workspace liegt z. B. unter
        /srv/projects) und wird als ausgelieferte Nuitka-``.exe`` richtig
        falsch: dort ist das Benutzerverzeichnis voellig unbeteiligt, waehrend
        die gesuchten Ordner direkt neben der ``.exe`` liegen.

        Die Reihenfolge geht vom SPEZIFISCHSTEN zum allgemeinsten: was der
        Nutzer schon eingetragen hat, schlaegt den fachlichen Standardort, und
        der schlaegt die Workspace-Wurzel. ``LAUNCHER_DIR`` ist der letzte
        Rueckfall und existiert per Definition immer.

        Args:
            *kandidaten: Pfade in absteigender Vorzugsreihenfolge; leere
                Eintraege und Nicht-Verzeichnisse werden uebergangen.

        Returns:
            Ein existierender Verzeichnispfad, nie eine leere Zeichenkette.
        """
        for kandidat in kandidaten:
            if kandidat and os.path.isdir(str(kandidat)):
                return str(kandidat)
        return str(LAUNCHER_DIR)

    def _browse_dir(self, var: tk.StringVar):
        """Ein Modellverzeichnis auswaehlen (LLM Models Directories)."""
        # Die Bibliothek liegt seit 02.08.2026 NEBEN dem Workspace; der alte
        # Ort darin bleibt als zweiter Kandidat stehen, damit aeltere
        # Installationen weiterhin am richtigen Fleck aufmachen.
        path = self._themed_picker(
            self._picker_start_dir(var.get(),
                                   WORKSPACE_ROOT.parent / MODELS_DIRNAME,
                                   WORKSPACE_ROOT / MODELS_DIRNAME,
                                   WORKSPACE_ROOT),
            title="Select LLM model-folder",
            pick_files=False)
        if path:
            var.set(path)

    def _browse_file(self, var: tk.StringVar):
        """Ein llama-server-Binary fuer einen Quick-Switch-Slot auswaehlen."""
        if IS_WINDOWS:
            patterns = ["*.exe"]
        else:
            patterns = ["llama-server*"]
        # AP-QL-62: slot folders live one level below the builds root, so a
        # configured binary path yields its root as dirname(dirname(...)).
        # The active engine and the other slots therefore know the user's
        # builds root even when it is NOT inside the workspace — that was
        # the old AP-QL-20 promise on Windows, carried over to the new
        # picker.
        def engine_root(p: str) -> str:
            return os.path.dirname(os.path.dirname(p)) if p else ""

        slot_roots = [engine_root(s)
                      for s in (self.cfg.get("llama_server_path", ""),
                                *self.cfg.get("server_slots", []))]
        # Start folder: the folder of the already-configured binary, then
        # the builds root derived from engine/slots, then this platform's
        # build folder inside the workspace. _picker_start_dir lands on
        # LAUNCHER_DIR when nothing of that exists (landing-zone rule).
        path = self._themed_picker(
            self._picker_start_dir(os.path.dirname(var.get()) if var.get() else "",
                                   *slot_roots,
                                   WORKSPACE_ROOT / BUILDS_DIRNAME,
                                   WORKSPACE_ROOT),
            title="Select llama-server file",
            pick_files=True,
            filename_patterns=patterns,
            # AP-QL-62: highlight the configured binary in the table so a
            # re-browse starts on the file the slot currently points at; a
            # fresh slot highlights the canonical server name if present.
            preselect=os.path.basename(var.get()) if var.get()
                      else exe_name("llama-server"))
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

        # AP-QL-68 (F1): OK lag zentriert, weil direkt auf das Toplevel
        # gepackt. Eine eigene Zeile mit fill="x" gibt ihm denselben
        # rechtsbuendigen Sitz wie in allen anderen Dialogen.
        about_btn_row = tk.Frame(dlg, bg=t.bg)
        about_btn_row.pack(fill="x", padx=16, pady=(12, 12))
        HoverButton(about_btn_row, t, text="OK", color=t.accent,
                    width=80, height=26,
                    command=dlg.destroy).pack(side="right")
        dlg.update_idletasks()
        self._center_window(dlg, 500, dlg.winfo_reqheight())

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION: Update Binaries
    # ═══════════════════════════════════════════════════════════════════════

    def _update_binaries(self):
        # Source path is configurable via Settings (update_binaries_source_path).
        src_str = self.cfg.get("update_binaries_source_path", "") or ""
        if not src_str.strip():
            self._log("Update Binaries: source path not configured. Set "
                      "'update_binaries_source_path' in "
                      f"{CONFIG_FILE.name} first.", "warn")
            return

        src = Path(src_str)
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

        # Platform-appropriate binaries + shared libs.
        if IS_WINDOWS:
            files = ["llama-server.exe", "llama-cli.exe", "llama-bench.exe",
                     "llama.dll", "ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll",
                     "ggml.dll", "mtmd.dll"]
        else:
            files = ["llama-server", "llama-cli", "llama-bench",
                     "libllama.so", "libggml-base.so", "libggml-cpu.so",
                     "libggml-cuda.so", "libggml.so", "libmtmd.so"]
        count = 0
        import shutil
        for f in files:
            s = src / f
            if s.exists():
                try:
                    shutil.copy2(s, dst / f)
                    # Ensure exec bit on POSIX for the 3 binaries (not libs).
                    if not IS_WINDOWS and not f.startswith("lib"):
                        try: os.chmod(dst / f, 0o755)
                        except OSError: pass
                    count += 1
                except Exception as e:
                    self._log(f"Error copying {f}: {e}", "error")
        self._log(f"Update: {count}/{len(files)} files copied.", "good")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION: Config & Cleanup
    # ═══════════════════════════════════════════════════════════════════════

    # ─── AP-QL-15: Fenstergeometrie sichern ────────────────────────────

    def _on_configure_geometry(self, event=None):
        """<Configure> auf dem Hauptfenster -> entprelltes Auto-Save.

        <Configure> feuert auch fuer JEDES Kindwidget (und beim Rebuild
        hunderte Male). Nur das Ereignis des Toplevels selbst zaehlt.
        """
        if event is not None and event.widget is not self:
            return
        if not getattr(self, "_geom_ready", False):
            return
        if self._geom_save_after is not None:
            try:
                self.after_cancel(self._geom_save_after)
            except Exception:
                pass
        self._geom_save_after = self.after(GEOM_SAVE_DEBOUNCE_MS,
                                           self._persist_geometry)
        # AP-QL-54: dieselbe Entprellung heilt den "Block" an Run/Stop.
        # BEWUSST nicht in _persist_geometry eingehaengt: das kehrt bei
        # maximiert/minimiert frueh um — Maximieren/Wiederherstellen ist
        # aber genau ein Resize, bei dem der Hover-Rest entsteht.
        if getattr(self, "_hover_resync_after", None) is not None:
            try:
                self.after_cancel(self._hover_resync_after)
            except Exception:
                pass
        self._hover_resync_after = self.after(GEOM_SAVE_DEBOUNCE_MS,
                                              self._resync_header_hover)

    def _resync_header_hover(self) -> None:
        """Run/Stop nach einem Resize einmal sauber neu zeichnen (AP-QL-54).

        Der Fenster-Resize verschiebt die Kopfzeilen-Knoepfe unter dem
        stehenden Mauszeiger; ein verpasstes <Enter>/<Leave> laesst dann
        einen hellen Hover-Rest ("Block") stehen, der erst beim naechsten
        zufaelligen Neuzeichnen verschwindet. Owner-Beobachtung 01.08.
        Spaetestens GEOM_SAVE_DEBOUNCE_MS nach dem Loslassen ist der
        Zustand jetzt garantiert abgeglichen.
        """
        self._hover_resync_after = None
        for btn in (getattr(self, "_run_btn", None),
                    getattr(self, "_stop_btn", None)):
            if btn is not None:
                try:
                    btn.resync_hover()
                except tk.TclError:
                    pass

    def _persist_geometry(self, announce: bool = False) -> bool:
        """Schreibt NUR die vier Geometrie-Keys in die Config.

        Bewusst nicht _save_current_config(): das schreibt die komplette
        Oberflaechenzustands-Config und liest dafuer UI-Variablen aus, die
        waehrend eines Rebuilds gerade im Umbau sein koennen. Ein Resize
        soll ausschliesslich Groesse und Position sichern.

        Gibt True zurueck, wenn wirklich geschrieben wurde.
        """
        self._geom_save_after = None
        try:
            # Maximiert/minimiert nicht speichern: winfo_width() liefert
            # dann die Vollbildmasse, und beim naechsten Start stuende das
            # Fenster in Maximalgroesse da, ohne maximiert zu sein.
            if self.state() != "normal":
                return False
            geom = {
                "window_x": self.winfo_x(),
                "window_y": self.winfo_y(),
                "window_w": self.winfo_width(),
                "window_h": self.winfo_height(),
            }
        except Exception:
            return False

        if all(self.cfg.get(k) == v for k, v in geom.items()):
            if announce:
                self._log("Window layout already saved "
                          f"({geom['window_w']}×{geom['window_h']} at "
                          f"{geom['window_x']},{geom['window_y']}).", "info")
            return False

        self.cfg.update(geom)
        save_config(self.cfg)
        if announce:
            self._log(f"Window layout saved — {geom['window_w']}×"
                      f"{geom['window_h']} at {geom['window_x']},"
                      f"{geom['window_y']}.", "good")
        return True

    def _on_save_geometry_click(self, _evt=None):
        """Klick auf '💾 Layout' — sofort schreiben, ohne Entprellung.

        Der Nutzer hat es ausdruecklich verlangt: ein bereits geplantes
        Auto-Save wird abgeraeumt, damit es nicht kurz danach nochmal
        schreibt.
        """
        if self._geom_save_after is not None:
            try:
                self.after_cancel(self._geom_save_after)
            except Exception:
                pass
            self._geom_save_after = None
        if self.state() != "normal":
            self._log("Window layout not saved — the window is maximized "
                      "or minimized. Restore it first.", "warn")
            return
        self._persist_geometry(announce=True)

    def _save_current_config(self):
        self.cfg["kv_cache"] = self._kv_var.get()
        try:
            self.cfg["port"] = int(self._port_var.get() or 8080)
        except ValueError:
            pass  # keep the last valid port instead of crashing on close
        # AP-QL-26 (AP-QL-10a/d): den Wert nur umschreiben, wenn der
        # freigegeben/privat-Zustand wirklich gekippt ist — sonst wuerde
        # eine von Hand gesetzte Adapter-IP (z. B. "192.168.1.50") bei
        # jedem Speichern auf die 0.0.0.0-Wildcard plattgemacht.
        _lan_now = self._lan_var.get()
        if _lan_now != server_host_is_lan(self.cfg):
            self.cfg["server_host"] = (SERVER_HOST_ALL if _lan_now
                                       else SERVER_HOST_LOOPBACK)
        # Der Schluessel geht in key_Linux.json, nie in die Hauptconfig.
        save_api_key(self._api_key_var.get())
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

        # AP-QL-27: Der Warntext aendert die Wunschbreite dieses Labels, und
        # place bricht nicht von selbst neu um — ohne diesen Anstoss ruecken
        # die folgenden Elemente nicht nach und ueberlappen im Zweifel.
        # force=True, weil sich die FENSTERbreite dabei nicht aendert und die
        # Waechter-Abfrage sonst abwinkt.
        self._reflow_settings_bar(force=True)

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
                color=t.accent_soft, state="normal",
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

    def _register_pending_model_lease(self, owner_token: str) -> None:
        """Record one queued ACQUIRE before its Tk callback can start."""
        if not valid_model_lease_token(owner_token):
            return
        with self._model_lease_state_lock:
            self._pending_model_leases[owner_token] = (
                self._pending_model_leases.get(owner_token, 0) + 1)

    def _finish_pending_model_lease(self, owner_token: Optional[str]) -> None:
        """Retire exactly one queued start after every success/refusal path."""
        if not valid_model_lease_token(owner_token):
            return
        with self._model_lease_state_lock:
            remaining = self._pending_model_leases.get(owner_token, 0) - 1
            if remaining > 0:
                self._pending_model_leases[owner_token] = remaining
            else:
                self._pending_model_leases.pop(owner_token, None)

    def _revoke_model_lease(self, owner_token: str) -> None:
        """Prevent every not-yet-adopted process start for this capability."""
        if not valid_model_lease_token(owner_token):
            return
        with self._model_lease_state_lock:
            self._revoked_model_leases.add(owner_token)

    def _model_lease_is_revoked(self, owner_token: Optional[str]) -> bool:
        if not valid_model_lease_token(owner_token):
            return False
        with self._model_lease_state_lock:
            return owner_token in self._revoked_model_leases

    def _run_pending_autoload(self, owner_token: Optional[str],
                              want_model: Optional[str] = None,
                              want_gpu: Optional[str] = None) -> None:
        """Run an externally owned autoload and always settle its pending bit."""
        try:
            self._trigger_autoload_if_eligible(owner_token=owner_token,
                                               want_model=want_model,
                                               want_gpu=want_gpu)
        finally:
            self._finish_pending_model_lease(owner_token)

    def _trigger_autoload_if_eligible(
            self, owner_token: Optional[str] = None,
            want_model: Optional[str] = None,
            want_gpu: Optional[str] = None):
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
        if self._closing:
            self._log("Autoload cancelled — launcher is closing.", "info")
            return
        if self._model_lease_is_revoked(owner_token):
            self._log("Autoload cancelled — its external owner has gone.",
                      "info")
            return
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
            current_proc = self.server_process
            if self.slot(SLOT_PRIMARY).is_alive():
                self._log(
                    "Autoload skipped — a server is already running.",
                    "info")
                return
            # The reader's queued exit callback may lag behind the actual
            # process. Clear exactly that dead generation so ACQUIRE does not
            # wait four minutes for a server which no longer exists.
            self._on_server_exited(SLOT_PRIMARY, current_proc)
        # 3. Resolve the model to start.
        #
        # AP-QL-88: the assigned Brain wins over last_model. Until now this
        # replayed "whatever was clicked last", which is a side effect and not
        # a decision — and it drifted: on 2026-08-21 the ReachyMini production
        # path was autostarting Qwen3.8-27B-UD-Q4_K_XL while the dev start
        # script loaded Qwen3.8-27B-Q6_K, with nobody having chosen either.
        #
        # last_model stays as the fallback, so a config without a Brain role
        # behaves exactly as it did in 1.16. A Brain that is assigned but no
        # longer in the scan does NOT silently fall through to last_model —
        # that would hide a broken assignment behind a plausible start. It is
        # reported, and autoload skips.
        last_gpu   = (self.cfg.get("last_gpu_key", "") or "GPU 0").strip()
        role_name  = (self.cfg.get(ROLE_BRAIN, "") or "").strip()
        #
        # AP-QL-92: a model named in THIS call outranks the assigned Brain,
        # for this run only.  That is the same reasoning AP-QL-88 used when
        # it put the role above last_model: a caller's explicit request is a
        # decision, last_model is a side effect — and a request is no less a
        # decision than a role.  Nothing is written back to cfg here, so the
        # role keeps winning again from the next autoload on.
        #
        # Passed as arguments rather than kept on self on purpose: every gate
        # above returns early, and an attribute set before those returns would
        # survive into an unrelated later autoload.
        wanted = (want_model or "").strip()
        wanted_gpu = (want_gpu or "").strip()
        if wanted_gpu:
            last_gpu = wanted_gpu
        if wanted:
            model = self._model_by_filename(wanted)
            if not model:
                # Deliberately no fallback to the role or to last_model. A
                # request for a model that is not there is a broken request;
                # starting something else would hide it behind a plausible
                # start — the same trap AP-QL-88 closed for the role.
                self._log(
                    f"Autoload: the requested model '{wanted}' is not in "
                    f"the current scan — skipping. Check the name or "
                    f"rescan the library.", "warn")
                return
            last_model = wanted
            self._log(
                f"Autoload: starting the requested model — {wanted}", "info")
        elif role_name:
            model = self._model_by_filename(role_name)
            if not model:
                self._log(
                    f"Autoload: the assigned Brain '{role_name}' is not in "
                    f"the current scan — skipping. Reassign the role or "
                    f"restore the file.", "warn")
                return
            last_model = role_name
            self._log(f"Autoload: starting the assigned Brain — {role_name}",
                      "info")
        else:
            last_model = (self.cfg.get("last_model", "") or "").strip()
            if not last_model:
                self._log(
                    "Autoload: no Brain assigned and no previous model "
                    "recorded — skipping. Assign a Brain role, or start a "
                    "model manually once.", "warn")
                return
            model = self._model_by_filename(last_model)
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
                # Actually start on that row — not on the missing GPU.
                rows = card.get("gpu_rows") or []
                if rows:
                    last_gpu = rows[0].get("key") or last_gpu
        try:
            self._select_cell(model_idx, row_idx, scroll="into")
        except Exception:
            # Non-fatal; just means the selection highlight didn't update.
            pass
        self._log(
            f"Autoload: starting {last_model} on {last_gpu}.", "good")
        # Use _on_run_model so the same stop-then-start machinery kicks in
        # that a manual Run click would use.
        self._on_run_model(last_model, last_gpu, owner_token=owner_token)

        # AP-QL-90: und dann, falls gewuenscht, der zweite Platz. Das ist die
        # Zeile, die dev/tools/start_servers_linux.sh entbehrlich macht:
        # `--autostart` mit vision_autoload=true bringt beide Rollen hoch,
        # ohne dass ein zweites Werkzeug wissen muss, welche Modelle das sind.
        #
        # Kein Abbruch, wenn es scheitert: das Gehirn laeuft dann trotzdem,
        # und _start_vision_server() nennt den Grund. Ein fehlender
        # Bildserver darf den Sprachserver nicht mitnehmen.
        if self.cfg.get("vision_autoload", False):
            self._start_vision_server(owner_token=owner_token)

    # ─── IPC server (Autoload gated) ─────────────────────────────────────

    def _claim_ipc_control(self) -> bool:
        """Acquire the OS-backed, crash-releasing control-plane election.

        The small claim file intentionally remains on disk after unlock.  If
        it were unlinked, another process could lock the old inode while a
        third process created and locked a new inode under the same name.
        Keeping one stable inode makes the election exclusive on both POSIX
        (flock) and Windows (msvcrt byte-range locking).
        """
        if self._ipc_claim_file is not None:
            return True
        claim_path = LOCK_FILE.with_name(LOCK_FILE.name + ".claim")
        claim_file = None
        try:
            claim_path.parent.mkdir(parents=True, exist_ok=True)
            claim_file = open(claim_path, "a+b")
            if IS_WINDOWS:
                claim_file.seek(0, os.SEEK_END)
                if claim_file.tell() == 0:
                    claim_file.write(b"\0")
                    claim_file.flush()
                claim_file.seek(0)
                msvcrt.locking(claim_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(claim_file.fileno(),
                            fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._ipc_claim_file = claim_file
            return True
        except (OSError, IOError):
            if claim_file is not None:
                try:
                    claim_file.close()
                except Exception:
                    pass
            return False

    def _release_ipc_control_claim(self) -> None:
        """Release only this window's held OS claim; safe to call twice."""
        claim_file = self._ipc_claim_file
        if claim_file is None:
            return
        try:
            if IS_WINDOWS:
                claim_file.seek(0)
                msvcrt.locking(claim_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(claim_file.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            claim_file.close()
        except Exception:
            pass
        self._ipc_claim_file = None

    def _start_ipc_server(self) -> bool:
        """Bind a local TCP control socket and write the lock file.

        Exactly one GUI may own the discovery lock. Plain extra windows remain
        allowed, but they neither overwrite nor later delete another window's
        control plane. The JSON is published with an atomic replace so a
        concurrent CLI reader can see only the old complete file or the new
        complete file, never a truncated document.
        """
        if self._ipc_socket is not None:
            return True
        if not self._claim_ipc_control():
            self._log("IPC control plane was claimed concurrently; this "
                      "window remains secondary.", "info")
            return False
        existing = read_lock_file()
        if existing is not None:
            self._log("IPC control plane already belongs to running instance "
                      f"PID {existing['pid']}; this window will not replace "
                      "it.", "info")
            self._release_ipc_control_claim()
            return False
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
            self._release_ipc_control_claim()
            return False

        # Another process may have won between the first read and our bind.
        # Never replace a live owner's discovery record.
        existing = read_lock_file()
        if existing is not None:
            try:
                sock.close()
            except Exception:
                pass
            self._ipc_socket = None
            self._ipc_port = 0
            self._release_ipc_control_claim()
            self._log("IPC control plane was claimed concurrently by PID "
                      f"{existing['pid']}; this window remains secondary.",
                      "info")
            return False

        # Write beside the lock, flush the complete JSON, then atomically
        # publish it. A RELEASE racing startup can no longer unlink a
        # half-written file and strand the newly started launcher.
        temp_lock = LOCK_FILE.with_name(
            LOCK_FILE.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            with open(temp_lock, "x", encoding="utf-8") as f:
                json.dump({
                    "pid": os.getpid(),
                    "port": self._ipc_port,
                    "started": datetime.now().isoformat(timespec="seconds"),
                    "version": APP_VERSION,
                }, f)
                f.flush()
                os.fsync(f.fileno())
            # Last sequential-owner check. os.replace itself is atomic, so
            # readers see a complete document throughout publication.
            existing = read_lock_file()
            if existing is not None:
                raise FileExistsError(
                    f"IPC already owned by PID {existing['pid']}")
            os.replace(temp_lock, LOCK_FILE)
        except Exception as e:
            self._log(f"IPC: failed to write lock file: {e}", "warn")
            try:
                temp_lock.unlink()
            except Exception:
                pass
            try:
                sock.close()
            except Exception:
                pass
            self._ipc_socket = None
            self._ipc_port = 0
            self._release_ipc_control_claim()
            return False

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
        try:
            self._ipc_accept_thread.start()
        except Exception as e:
            self._log(f"IPC: failed to start control thread: {e}", "error")
            owned_port = self._ipc_port
            try:
                sock.close()
            except Exception:
                pass
            self._cleanup_lock_file(owned_port)
            self._ipc_socket = None
            self._ipc_port = 0
            self._ipc_accept_thread = None
            self._release_ipc_control_claim()
            return False
        return True

    def _stop_ipc_server(self):
        """Tear down the control socket + lock file. Safe to call twice."""
        if self._ipc_socket is None:
            # A secondary/plain GUI owns no control file. In particular it
            # must never delete another live window's discovery record.
            self._release_ipc_control_claim()
            return
        owned_port = self._ipc_port
        try:
            self._ipc_socket.close()
        except Exception:
            pass
        self._cleanup_lock_file(owned_port)
        self._ipc_socket = None
        self._ipc_port = 0
        self._ipc_accept_thread = None
        self._release_ipc_control_claim()

    def _release_model_lease(self, owner_token: str) -> None:
        """Unload only model slots created for this opaque capability.

        The launcher window and every manual/pre-existing model remain.  This
        is deliberately narrower than SHUTDOWN: ReachyMini can return the GPU
        memory it acquired without becoming owner of a QLauncher that happened
        to be open already.
        """
        if not valid_model_lease_token(owner_token):
            return
        self._revoke_model_lease(owner_token)
        released: List[str] = []
        failed: List[str] = []
        vision = self.slot(SLOT_VISION)
        if vision.owner_token == owner_token:
            if self._stop_vision_server(silent=False):
                released.append(SLOT_VISION)
            else:
                failed.append(SLOT_VISION)
        primary = self.slot(SLOT_PRIMARY)
        if primary.owner_token == owner_token:
            if self._stop_server(silent=False):
                released.append(SLOT_PRIMARY)
            else:
                failed.append(SLOT_PRIMARY)
        if failed:
            attempt = self._model_lease_release_retries.get(owner_token, 0) + 1
            self._model_lease_release_retries[owner_token] = attempt
            self._log("External model lease release not confirmed: "
                      + ", ".join(failed) + f" (attempt {attempt}/10).",
                      "error")
            if attempt < 10 and not self._closing:
                self.after(1000, self._release_model_lease, owner_token)
        elif released:
            self._model_lease_release_retries.pop(owner_token, None)
            self._log("External model lease released: "
                      + ", ".join(released) + ".", "info")
        else:
            self._model_lease_release_retries.pop(owner_token, None)
            self._log("External model lease released: no owned model was "
                      "running.", "info")

    def _owned_model_slots(self, owner_token: str) -> List[str]:
        """Live slots carrying exactly this external capability."""
        if not valid_model_lease_token(owner_token):
            return []
        with self._model_lease_state_lock:
            return [name for name in (SLOT_PRIMARY, SLOT_VISION)
                    if self.slot(name).owner_token == owner_token
                    and self.slot(name).is_alive()]

    def _model_lease_status(self, owner_token: str) -> dict:
        """Atomic proof state for RELEASE polling.

        ``pending`` stays true while any accepted ACQUIRE callback can still
        reach Popen.  The same lock spans Popen plus slot adoption, so a caller
        can never receive ``pending=false, owned_slots=[]`` immediately before
        a process with that token appears.
        """
        if not valid_model_lease_token(owner_token):
            return {"owned_slots": [], "pending": False}
        with self._model_lease_state_lock:
            owned = [name for name in (SLOT_PRIMARY, SLOT_VISION)
                     if self.slot(name).owner_token == owner_token
                     and self.slot(name).is_alive()]
            return {
                "owned_slots": owned,
                "pending": self._pending_model_leases.get(owner_token, 0) > 0,
            }

    def _cleanup_lock_file(self, owned_port: int) -> None:
        """Delete only this process+socket's own discovery record."""
        try:
            data = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
            if (int(data.get("pid", 0)) == os.getpid()
                    and int(data.get("port", 0)) == int(owned_port)):
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

                if (getattr(self, "_closing", False)
                        and cmd in ("AUTOSTART", "ACQUIRE")):
                    try:
                        conn.sendall(b"ERR launcher is closing\n")
                    except Exception:
                        pass
                    return

                if cmd == "SHUTDOWN":
                    # Reply first, THEN schedule the shutdown, so the
                    # caller sees a clean "OK" before we tear down. The
                    # 100 ms delay gives the reply time to flush.
                    try:
                        conn.sendall(b"OK shutting down\n")
                    except Exception:
                        pass
                    self._queue_main(self._on_close, delay_ms=100)

                elif cmd in ("AUTOSTART", "ACQUIRE"):
                    # AUTOSTART [MODEL] [GPU] and, since AP-QL-92,
                    # ACQUIRE TOKEN [MODEL] [GPU] — the same optional tail,
                    # so a caller no longer has to choose between naming the
                    # model and owning what it started.
                    #
                    # The tail is re-split with a bounded split instead of
                    # reusing `parts`: a GPU key is legitimately "GPU 0", and
                    # a plain split() cut it in half. That is why the
                    # documented AUTOSTART GPU argument never worked for a
                    # real key — it arrived as "GPU". Letting the last field
                    # absorb the remainder fixes both commands at once.
                    # Model filenames must therefore stay space-free; every
                    # GGUF name in the wild is, and _model_by_filename
                    # rejects anything that is not in the scan anyway.
                    tail = command.split(None, 3 if cmd == "ACQUIRE" else 2)
                    owner_token = None
                    if cmd == "ACQUIRE":
                        if (len(tail) < 2
                                or not valid_model_lease_token(tail[1])):
                            try:
                                conn.sendall(b"ERR invalid lease token\n")
                            except Exception:
                                pass
                            return
                        owner_token = tail[1]
                        new_model = tail[2] if len(tail) > 2 else None
                        new_gpu = tail[3] if len(tail) > 3 else None
                    else:
                        new_model = tail[1] if len(tail) > 1 else None
                        new_gpu = tail[2] if len(tail) > 2 else None
                    if new_model is not None and not valid_ipc_argument(
                            new_model):
                        try:
                            conn.sendall(b"ERR invalid model\n")
                        except Exception:
                            pass
                        return
                    if new_gpu is not None and not valid_ipc_argument(
                            new_gpu, allow_spaces=True):
                        try:
                            conn.sendall(b"ERR invalid gpu\n")
                        except Exception:
                            pass
                        return
                    if owner_token:
                        # AP-QL-117: a lease is never granted over a model this
                        # launcher did not start under this token. Without this
                        # the grant is accepted, _trigger_autoload_if_eligible
                        # then skips ("a server is already running"), the
                        # pending bit retires in its finally, and no slot ever
                        # carries the token -- which is EXACTLY the success
                        # conjunction _model_lease_status reports. A later
                        # --release then prints "released" and exits 0 while
                        # the model sits in VRAM. Refusing here is what keeps
                        # that exit 0 honest.
                        #
                        # Scoped to the primary slot on purpose: it is the same
                        # condition that makes the autoload skip, and a live
                        # vision slot does not stand in the way of a brain.
                        # Re-acquiring with the SAME token stays allowed, so a
                        # caller repeating its own request is not punished.
                        with self._model_lease_state_lock:
                            primary = self.slot(SLOT_PRIMARY)
                            foreign = (primary.is_alive()
                                       and primary.owner_token != owner_token)
                        if foreign:
                            try:
                                conn.sendall(
                                    b"ERR model already running unowned\n")
                            except Exception:
                                pass
                            return
                        self._register_pending_model_lease(owner_token)
                    def _fire():
                        if new_model:
                            self.cfg["last_model"] = new_model
                        if new_gpu:
                            self.cfg["last_gpu_key"] = new_gpu
                        if new_model or new_gpu:
                            save_config(self.cfg)
                        # Temporarily force-on the override for this run
                        # regardless of the toggle state, then fire.
                        prev = self._autostart_override
                        self._autostart_override = True
                        try:
                            self._run_pending_autoload(owner_token,
                                                       want_model=new_model,
                                                       want_gpu=new_gpu)
                        finally:
                            self._autostart_override = prev
                    try:
                        conn.sendall(
                            b"OK acquiring\n" if owner_token
                            else b"OK starting\n")
                    except Exception:
                        pass
                    if not self._queue_main(_fire, delay_ms=50):
                        self._finish_pending_model_lease(owner_token)
                        return

                elif cmd == "RELEASE":
                    if (len(parts) != 2
                            or not valid_model_lease_token(parts[1])):
                        try:
                            conn.sendall(b"ERR invalid lease token\n")
                        except Exception:
                            pass
                        return
                    owner_token = parts[1]
                    # Mark revoked in this handler before the reply. A newly
                    # launched GUI can still have its autostart queued in Tk;
                    # that callback must observe the release even if it runs
                    # before the cleanup callback below.
                    self._revoke_model_lease(owner_token)
                    try:
                        conn.sendall(b"OK releasing\n")
                    except Exception:
                        pass
                    self._queue_main(self._release_model_lease,
                                     owner_token)

                elif cmd == "LEASE_STATUS":
                    if (len(parts) != 2
                            or not valid_model_lease_token(parts[1])):
                        try:
                            conn.sendall(b"ERR invalid lease token\n")
                        except Exception:
                            pass
                        return
                    try:
                        conn.sendall((json.dumps(
                            self._model_lease_status(parts[1]))
                            + "\n").encode("utf-8"))
                    except Exception:
                        pass

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
                        # AP-QL-88: die Rollen gehoeren in den Bericht, sonst
                        # muesste ein externer Aufrufer (ReachyMini) die
                        # Config-Datei selbst lesen und ihr Format kennen.
                        # Der Pfad steht dabei, weil ein Dateiname allein den
                        # Aufrufer zwingen wuerde, die Bibliothek zu scannen.
                        # AP-QL-90: der zweite Platz. Ein Aufrufer, der das
                        # Bild woanders hinschicken will als den Text, braucht
                        # genau diese drei Angaben.
                        "vision_running": self.slot(SLOT_VISION).is_alive(),
                        "vision_model":   self.slot(SLOT_VISION).model,
                        "vision_port":    self.slot(SLOT_VISION).port,
                        "role_brain":    self.cfg.get(ROLE_BRAIN, ""),
                        "role_vision":   self.cfg.get(ROLE_VISION, ""),
                        "role_brain_path":
                            (lambda m: m.path if m else "")(
                                self._model_by_filename(
                                    self.cfg.get(ROLE_BRAIN, ""))),
                        "role_vision_path":
                            (lambda m: m.path if m else "")(
                                self._model_by_filename(
                                    self.cfg.get(ROLE_VISION, ""))),
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
        self._closing = True
        # AP-QL-90: beide Plaetze. Der Vision-Server ist ein eigener Prozess
        # und wuerde sonst den Launcher ueberleben — mit belegtem Port und
        # belegtem VRAM, und ohne dass irgendetwas ihn noch stoppen koennte.
        vision_ok = self._stop_vision_server(silent=True)
        primary_ok = True
        if self.server_process:
            primary_ok = self._stop_server(silent=True)
        if vision_ok is False or primary_ok is False:
            # Keep UI, slot ownership and IPC alive. Destroying them here
            # would turn the deliberately retained stubborn process into an
            # unreachable orphan and make every later release false-green.
            self._closing = False
            self._log("QLauncher close cancelled — at least one server "
                      "process is still alive and remains tracked.", "error")
            return
        # Only proven process exit permits the control plane and window to go.
        # The closing flag rejects new ACQUIRE work during this final gap.
        self._stop_ipc_server()
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
      --acquire TOKEN      Autostart with an opaque model-ownership token.
                           A pre-existing model is borrowed, never claimed.
      --model NAME         AP-QL-92. Start this GGUF (bare filename, as it
                           appears in the scan) instead of the recorded
                           last model — and instead of the assigned Brain
                           role, for this one run. Needs --autostart or
                           --acquire.
      --gpu KEY            The GPU row for --model, e.g. "GPU 0". Only
                           together with --model: on the wire the GPU is
                           the trailing field, so it cannot be sent
                           without one.
      --release TOKEN      Unload only model slots created for that token;
                           leave the QLauncher and every other model intact.
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
            # AP-QL-78 (Fund F): Beispiel wieder da. Die Option selbst war
            # intakt, nur unauffindbar — und die ReachyMini-App startet den
            # Launcher genau mit dieser Kombination.
            "  TurboQuant_QLauncher --autostart --minimized\n"
            "                                        same, but parked in the taskbar\n"
            "  TurboQuant_QLauncher --acquire TOKEN --minimized\n"
            "                                        leased autostart for another app\n"
            "  TurboQuant_QLauncher --acquire TOKEN --model M.gguf --gpu \"GPU 0\"\n"
            "                                        leased autostart of a named model\n"
            "  TurboQuant_QLauncher --release TOKEN  unload only that app's models\n"
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
    g.add_argument("--acquire", metavar="TOKEN",
                   help="Force autoload under an opaque ownership token; "
                        "a pre-existing model remains unowned. TOKEN is "
                        "single-use: once released it stays revoked, so mint "
                        "a fresh one for every acquire.")
    p.add_argument("-q", "--shutdown", action="store_true",
                   help="Tell a running instance to close cleanly and exit.")
    p.add_argument("--release", metavar="TOKEN",
                   help="Unload only models acquired with TOKEN and keep the "
                        "launcher open.")
    p.add_argument("--status", action="store_true",
                   help="Print JSON status from the running instance, "
                        "then exit.")
    p.add_argument("--minimized", action="store_true",
                   help="Open iconified (taskbar only). Meant for callers "
                        "that start a server in the background and do not "
                        "want the window jumping in front of their own UI.")
    # AP-QL-92: not in the mutually-exclusive group — these two qualify an
    # autostart, they are not a third kind of it.
    p.add_argument("--model", metavar="NAME",
                   help="Start this GGUF (bare filename from the scan) for "
                        "this run, outranking last_model and the assigned "
                        "Brain role. Requires --autostart or --acquire.")
    p.add_argument("--gpu", metavar="KEY",
                   help="GPU row for --model, e.g. \"GPU 0\". Only valid "
                        "together with --model.")
    p.add_argument("--version", action="version",
                   version=f"TurboQuant QLauncher v{APP_VERSION}")
    args = p.parse_args(argv)
    for token in (args.acquire, args.release):
        if token is not None and not valid_model_lease_token(token):
            p.error("model lease TOKEN must contain 16-128 ASCII letters, "
                    "digits, '_' or '-'")
    if args.release and (args.autostart or args.no_autostart or args.acquire
                         or args.shutdown or args.status):
        p.error("--release cannot be combined with another control command")
    # AP-QL-92. The GPU rule is not pedantry: the warm path forwards these
    # two positionally ("ACQUIRE TOKEN MODEL GPU"), so a GPU without a model
    # has no slot to travel in. Refusing it keeps the cold and the warm path
    # behaving identically instead of silently diverging.
    if (args.model or args.gpu) and not (args.autostart or args.acquire):
        p.error("--model/--gpu need --autostart or --acquire")
    if args.gpu and not args.model:
        p.error("--gpu is only valid together with --model")
    if args.model is not None and not valid_ipc_argument(args.model):
        p.error("--model must be a bare GGUF filename without spaces")
    if args.gpu is not None and not valid_ipc_argument(args.gpu,
                                                       allow_spaces=True):
        p.error("--gpu must be a GPU key such as \"GPU 0\"")
    return args


def _handle_cli_only_commands(args: argparse.Namespace) -> Optional[int]:
    """Handle commands that don't need a GUI: shutdown/status/release.

    Returns an exit code (0 or 1) if the command was fully handled here
    and the caller should exit now. Returns None when the launcher must
    proceed to open the GUI (covers plain launch and --autostart).
    """
    if not (args.shutdown or args.status or args.release):
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
        if read_lock_file() is not None:
            print("TurboQuant close was not confirmed; a server process may "
                  "still be alive.", file=sys.stderr)
            return 1
        print("TurboQuant closed.")
        return 0

    if args.release:
        reply = send_ipc_command(port, f"RELEASE {args.release}")
        if reply is None or not reply.startswith("OK "):
            print(
                f"Could not release model lease through "
                f"127.0.0.1:{port}.", file=sys.stderr)
            return 1
        # RELEASE replies before Tk performs the stop. Poll the token itself,
        # not a broad process name or port: a borrowed/pre-existing model may
        # correctly remain online while this lease becomes empty.
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            state = send_ipc_command(port, f"LEASE_STATUS {args.release}")
            try:
                lease_state = json.loads(state or "")
                owned = lease_state.get("owned_slots")
                # Missing means "not proven" when talking to an older
                # launcher. Failing closed is preferable to confirming an
                # unload while that launcher may still have a queued start.
                pending = lease_state.get("pending", True)
            except (ValueError, TypeError, AttributeError):
                owned = None
                pending = True
            if owned == [] and pending is False:
                print("TurboQuant model lease released.")
                return 0
            time.sleep(0.1)
        print("TurboQuant model lease release was not confirmed.",
              file=sys.stderr)
        return 1

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

    # CLI-only paths (shutdown/status/release) take precedence over GUI startup.
    # They neither require nor benefit from Tk, so we short-circuit here.
    rc = _handle_cli_only_commands(args)
    if rc is not None:
        sys.exit(rc)

    # v0.56 — Single-instance coordination for --autostart. If an
    # instance with Autoload is already running, forward the AUTOSTART
    # command and exit; don't try to spin up a second Tk. Plain
    # launches (no --autostart) just open a second window on purpose —
    # matches legacy behaviour.
    if args.autostart or args.acquire:
        lock = read_lock_file()
        if lock is not None:
            command = (f"ACQUIRE {args.acquire}"
                       if args.acquire else "AUTOSTART")
            # AP-QL-92: the optional tail travels positionally, GPU last, so
            # a key with a space survives the receiver's bounded split.
            # _parse_cli_args has already refused a --gpu without --model,
            # which is what makes the positional form safe here.
            if args.model:
                command += f" {args.model}"
                if args.gpu:
                    command += f" {args.gpu}"
            reply = send_ipc_command(lock["port"], command)
            if reply is not None and reply.startswith("OK "):
                print("TurboQuant already running — "
                      + ("model lease forwarded."
                         if args.acquire else "autostart forwarded."))
                sys.exit(0)
            if reply is not None and reply.startswith("ERR "):
                # AP-QL-117: a refusal is an ANSWER. Falling through here would
                # start the second Tk instance that this whole failure mode is
                # about -- so hunk 1 without this one would make things worse,
                # not better. Unreachable stays unreachable (reply is None);
                # only a spoken ERR ends the process.
                print(f"TurboQuant refused: {reply.strip()}", file=sys.stderr)
                sys.exit(1)
            # Fell through — existing instance is unreachable. Proceed
            # to start a new one rather than failing.
            print(
                "Existing instance did not reply; starting a new one.",
                file=sys.stderr)

    # ─── DPI awareness (MUST run before any Tk window is created) ─────────
    # AP-QL-76 (Delta-Audit 03.08., Fund D): woertlich aus der archivierten
    # Windows-Linie zurueckgeholt. Der Block war dort bereits auf win32
    # gegattert und haette beim Zusammenlegen risikofrei stehenbleiben
    # koennen — er ging trotzdem verloren.
    #
    # Without this, Windows applies DWM bitmap scaling on HiDPI displays:
    # Tk renders at 96 DPI into a small bitmap, Windows bilinearly upscales
    # → blurry text ("unscharf"). Declaring System-DPI-Aware disables the
    # bitmap upscale so text renders at native resolution. Per-Monitor-V2
    # would be nicer but Tk doesn't handle WM_DPICHANGED, so staying at
    # level 1 is the right trade-off.
    #
    # Nebenwirkung, die den Fund schwerer macht als "unscharf": ohne
    # DPI-Awareness meldet winfo_screenwidth/height VIRTUALISIERTE Werte.
    # Damit rechnen die tk-scaling-Heuristik und die Fenster-Deckelung im
    # Konstruktor mit einem Schirm, den es so nicht gibt.
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
    if args.autostart or args.acquire:
        autostart_override: Optional[bool] = True
    elif args.no_autostart:
        autostart_override = False
    else:
        autostart_override = None

    app = TurboQuantQLauncher(
        autostart_override=autostart_override,
        autostart_owner_token=args.acquire,
        autostart_model=args.model,
        autostart_gpu=args.gpu)
    # --minimized: park in the taskbar instead of opening in front of the
    # caller's window. Useful when another app starts a server in the
    # background (e.g. the ReachyMini chat app on an LLM cold start).
    # iconify() must run after the window exists but before mainloop().
    if getattr(args, "minimized", False):
        try:
            app.iconify()
        except Exception:
            # Never let a cosmetic wish stop the launcher from running.
            pass
    app.mainloop()

if __name__ == "__main__":
    main()
