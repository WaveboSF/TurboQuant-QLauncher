# Changelog

## 1.54

- Added support for compatible Inkling profiles through separately installed [Xolotl](https://github.com/Indras-Mirror/Xolotl); tested on Linux, with Windows support not yet verified.
- Highlight server builds with recognized support for the selected model and simplify server selection in Paths.
- Match No Thinking and DFlash2 controls to the launcher buttons, and organize credits and licenses in a readable, movable window.

Thanks to [Indras-Mirror and the Xolotl contributors](https://github.com/Indras-Mirror/Xolotl) for their work.

## 1.53

- Optional DFlash2 acceleration for compatible Ripple models, with a Q4 draft and per-device context estimation.
- Clearer running, stop, unload, browser and autoload controls; DFlash2 matches the other checkbox controls.
- Windows can open Microsoft's official Visual C++ runtime download when needed.
- Updated credits and third-party license notices in both release packages.

The bundled DFlash2 estimator is for Linux with the matching llama.cpp b12503 build. Windows DFlash2 requires a separately supplied matching helper and has not yet been tested.

## 1.44

- Native Linux and Windows packages, with separate llama.cpp / TurboQuant and Flash-Next / Strata backends.
- Comb.GPU model loading on compatible NVIDIA GPUs, editable context and a browser button.

See [Releases](https://github.com/WaveboSF/TurboQuant-QLauncher/releases) for downloads and detailed release notes. Changes to the historical Python version remain in the [legacy changelog](legacy/CHANGELOG.md).
