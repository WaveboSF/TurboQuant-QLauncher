#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# TurboQuant QLauncher — Linux start script
# ═══════════════════════════════════════════════════════════════════════════════
# Activates the correct Conda environment, then launches the QLauncher GUI.
# Keep this script and TurboQuant_QLauncher_Linux.py in the same directory,
# or adjust QLAUNCHER_DIR below.
#
# Used by turboquant-qlauncher.desktop — do not rename without updating the
# Exec= line in the .desktop file.
# ═══════════════════════════════════════════════════════════════════════════════

set -e

# ─── Configuration ─────────────────────────────────────────────────────────────
# Conda installation root. Adjust if you move miniconda.
CONDA_ROOT="${HOME}/miniconda3"

# Conda environment to use. "base" works for now; change to a dedicated env
# later (e.g. "qlauncher") without touching anything else.
CONDA_ENV="base"

# Directory where TurboQuant_QLauncher_Linux.py lives. Resolved relative to
# this script's own location by default — move the whole folder around freely.
QLAUNCHER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── Activate Conda ────────────────────────────────────────────────────────────
if [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
else
    echo "ERROR: Conda not found at ${CONDA_ROOT}" >&2
    echo "Edit CONDA_ROOT in this script or install miniconda." >&2
    exit 1
fi

# ─── Sanity: tkinter must be importable ────────────────────────────────────────
if ! python3 -c "import tkinter" 2>/dev/null; then
    echo "ERROR: tkinter not available in Conda env '${CONDA_ENV}'." >&2
    echo "Fix: conda install -n ${CONDA_ENV} tk" >&2
    exit 1
fi

# ─── Launch ────────────────────────────────────────────────────────────────────
cd "${QLAUNCHER_DIR}"
exec python3 TurboQuant_QLauncher_Linux.py "$@"
