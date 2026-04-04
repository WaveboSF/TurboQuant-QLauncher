# TurboQuant QLauncher

**Model Switcher & Benchmark Tool for llama-server with TurboQuant KV-Cache Compression**

A lightweight desktop GUI for managing [llama-server](https://github.com/ggml-org/llama.cpp) with [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) KV-Cache compression. Launch models, switch KV configs, run benchmarks, and monitor VRAM — all without touching the command line.

Zero external dependencies. Python stdlib + tkinter only.

![TurboQuant QLauncher Screenshot](screenshot.png)

---

## Features

| | Feature | Description |
|---|---|---|
| ⚡ | **One-click Launch** | Double-click any model × GPU row to start llama-server. Double-click again to stop. |
| 📊 | **Live VRAM Bars** | Braille-style bars show effective VRAM usage — updated live when you switch KV config. |
| 🎨 | **KV-aware Colors** | Green = fits. Amber = tight. Red = overflow. Changes with compression selection. |
| 📈 | **Benchmarking** | Built-in llama-bench: single config or full KV × LA matrix in one run. |
| 🧠 | **Layer-Adaptive** | Modes 1/5/7 keep critical layers at q8_0 for better quality via `TURBO_LAYER_ADAPTIVE`. |
| 📁 | **Bench All** | Matrix benchmark: all KV configs × all LA modes. Stop anytime. Markdown output. |
| 💾 | **Persistent Config** | Window position, KV selection, port, and all settings saved automatically. |
| 🔍 | **Auto-scan Models** | Scans your model directory for `*.gguf` files with size, fit, and compression estimates. |
| 🖥️ | **Multi-GPU** | NVIDIA + AMD detection. Per-model GPU selector with `CUDA_VISIBLE_DEVICES` routing. |

---

## Quick Start

### Requirements

- **Python 3.10+** (with tkinter — included in standard Windows/macOS installs)
- **llama-server** binaries from the [spiritbuun TurboQuant CUDA fork](https://github.com/spiritbuun/llama-cpp-turboquant-cuda/tree/feature/turboquant-kv-cache)
- **GGUF models** (e.g. from [Hugging Face](https://huggingface.co/models?search=gguf))
- **NVIDIA GPU** with CUDA 12.x or 13.x (AMD via ROCm also supported)

### Run

```bash
python TurboQuant_QLauncher.py
```

On first launch, the Paths dialog opens automatically — set your models directory and `llama-server.exe` path.

### Pre-built Binary

A standalone Windows `.exe` (built with [Nuitka](https://nuitka.net/)) is available in [Releases](../../releases). No Python installation required.

---

## KV-Cache Configurations

| Button | Config | Compression | Best for |
|--------|--------|-------------|----------|
| `f16` | f16 / f16 | 1× (none) | Baseline, maximum quality |
| `q8₀+t4` | q8_0-K + turbo4-V | 2.7× | **Recommended for Q4_K_M models** |
| `t3/t3` | turbo3 / turbo3 | 5× | Maximum context length |
| `t4/t4` | turbo4 / turbo4 | 4× | Good balance |
| `q8₀+t3` | q8_0-K + turbo3-V | 2.9× | High compression, good quality |
| `q8₀/q8₀` | q8_0 / q8_0 | 2× | Standard 8-bit quantization |

The VRAM bars and ✓/✗ indicators update live as you click KV buttons.

---

## Layer-Adaptive Modes

| LA | Strategy | Use case |
|----|----------|----------|
| `off` | All layers equally compressed | Maximum compression |
| `1` | First 4 + last 4 layers at q8_0 | **Best quality** (recommended) |
| `5` | Aggressive compression | 128K context on 24 GB |
| `7` | First 2 + last 2 V-cache at q8_0 | Minimal VRAM overhead |

LA modes have negligible speed impact (<1%) — the benefit is output quality. LA=1 also provides slightly better decode speed due to native int8 dp4a at critical layers.

---

## Controls

| Action | Effect |
|--------|--------|
| **Double-click** row | Launch server (or stop if running) |
| **Single click** | Select model / GPU without launching |
| **↑ ↓ / ← →** | Navigate models / GPUs |
| **Enter** | Toggle server for selected row |
| **Escape** | Stop running server |
| **Click status label** | Open `http://localhost:{port}` in browser |
| **Run / Stop** | Start or cancel benchmark |
| **LA: off/1/5/7** | Select Layer-Adaptive mode |
| **☑ Bench All** | Select all KV × LA configs, then click Run |

---

## Benchmark Output

Results are saved as Markdown tables to `TurboQuant_Benchmark_results.md` — ready for GitHub posts:

```markdown
### model.gguf — RTX 4090, Build CUDA 12.8 (sm_89;120), Driver 13.2, 2026-04-04

| KV-Cache          | pp512 (t/s) | tg128 (t/s) | Δ Prefill | Δ Decode |
|-------------------|-------------|-------------|-----------|----------|
| **LA=off**        |             |             |           |          |
| f16 (default)     | 10,112.60   | 150.59      | —         | —        |
| q8_0-K + turbo4-V | 10,808.36   | 136.37      | +6.9%     | -9.4%    |
| ...               |             |             |           |          |
```

Each run includes model name, GPU, CUDA build version (sm targets), driver version, and timestamp.

---

## Benchmark Findings

Tested on AMD 9950X3D · 64 GB DDR5 · RTX 5090 (32 GB) + RTX 4090 (24 GB).
Build: spiritbuun `feature/turboquant-kv-cache` · CUDA 12.8 · `-DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON`.

### Key Results — Meta-Llama-3.1-8B-Instruct-Q4_K_M · LA=1

| Config | RTX 4090 (Ada, sm_89) | RTX 5090 (Blackwell, sm_120) |
|---|---|---|
| f16 baseline | 10,077 / 150.2 t/s | 12,761 / 231.9 t/s |
| q8₀+t4 | +8.7% / **-6.9%** | -3.2% / **-25.2%** |
| q8₀+t3 | +8.5% / **-8.2%** | -2.7% / **-30.2%** |
| t3/t3 | +3.8% / **-14.9%** | -7.5% / **-39.7%** |

### Notable Findings

- **Ada (sm_89) handles TurboQuant decode significantly better than Blackwell (sm_120).** RTX 4090 loses only 7% decode with q8₀+t4, while RTX 5090 loses 25%. This is structural: Ada's int8 dp4a instructions align well with TurboQuant's dequant path.
- **LA=1 improves decode speed** — not just quality. The q8_0 layers at critical positions use faster native int8 decoding.
- **CUDA 12.8 = CUDA 13.2 on this hardware.** sm_120a provides no measurable benefit over sm_120 for TurboQuant workloads.
- **`CUDA_DEVICE_ORDER=PCI_BUS_ID` is required** on mixed-arch systems (Ada + Blackwell) — without it, `CUDA_VISIBLE_DEVICES` silently routes to the wrong GPU.
- **Recommendation:** `q8_0-K + turbo4-V` with `LA=1` — best balance of compression, speed, and quality.

---

## Building llama-server

QLauncher works with the [spiritbuun TurboQuant CUDA fork](https://github.com/spiritbuun/llama-cpp-turboquant-cuda/tree/feature/turboquant-kv-cache). Build with Flash Attention flags enabled:

```cmd
git clone --branch feature/turboquant-kv-cache --depth 1 ^
  https://github.com/spiritbuun/llama-cpp-turboquant-cuda.git

cd llama-cpp-turboquant-cuda

cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON ^
  -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON ^
  -DCMAKE_CUDA_ARCHITECTURES="89;120" ^
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j
```

> **MSVC Note:** If you get an `M_PI` error, add `#define _USE_MATH_DEFINES` and `#include <math.h>` at the top of `ggml/src/ggml-turbo-quant.c`.

> **Multi-GPU Note:** Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` for correct GPU routing on mixed-architecture systems (e.g. Ada + Blackwell). QLauncher does this automatically.

---

## Related Projects

- [spiritbuun/llama-cpp-turboquant-cuda](https://github.com/spiritbuun/llama-cpp-turboquant-cuda) — TurboQuant CUDA implementation (this build)
- [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) — Metal + CUDA, block_size=128, Sparse V
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — Upstream llama.cpp
- [Discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969) — Community benchmarks & findings

---

## Acknowledgements

- **spiritbuun** — TurboQuant CUDA kernels, norm correction, fused FA
- **TheTom** — turboquant_plus, block_size=128 optimization, Metal support
- **seanrasch** — Ampere benchmarks, 256K context dual-GPU validation
- **AmesianX** — Blackwell / TurboQuant v1.4.0, IWHT FP16 bug fix
- **primoco** — Accuracy benchmarks (100% math accuracy with q8_0-K/tq4_0-V)

TurboQuant refers to the paper by Zandieh et al. ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874), ICLR 2026, Google Research). This tool is an independent community project, not affiliated with Google.

---

## License

MIT — © WaveboSF 2026

Built with assistance from Claude (Anthropic).
