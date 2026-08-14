<div align="center">

# 🔊 Acoustic Analyzer

**A local edge-AI that listens through a soldered microphone and explains what it hears.**

Runs entirely on an NVIDIA Jetson — no cloud, and no raw audio ever leaves the box.

</div>

![Acoustic Analyzer architecture](docs/architecture/acoustic-analyzer.svg)

---

## Overview

Acoustic Analyzer is a hands-on **learning project** that combines three skills:
**soldering** an analog microphone front-end, **edge computing** on a Jetson, and **local AI**
with a small tool-calling LLM.

The core idea: a 3-billion-parameter model can't — and shouldn't — listen to a raw waveform.
A Python DSP layer distills each recording into **structured facts** (dominant frequency, tonal
peaks, band energy, a 60 Hz hum score), and the language model reasons over those. That keeps the
model small, the device fully offline, and every answer grounded in real measurements.

```text
you> what's that noise?
  [tool] capture_and_analyze(seconds=2)
  → { "label": "electrical hum (mains-related)", "dominant_hz": 60.0,
      "peaks_hz": [60, 120, 180], "hum_60hz_score": 1.00 }
agent> That's 60 Hz mains hum with harmonics at 120 and 180 Hz — likely a ground loop.
```

## Features

- 🎙️ **Real DSP** — live FFT spectrogram, dominant-tone and peak detection, band energy, and a
  60 Hz mains-hum score.
- 🧠 **Local LLM with tools** — `qwen2.5:3b` on [Ollama](https://ollama.com) calls the DSP as
  tools and explains sounds in plain language. No API keys, no cloud.
- 🔌 **Soldering project** — a ~$25 analog front-end (electret mic → MAX9814 → RC filter → USB).
- 💻 **Develop anywhere** — the same code runs on a Mac or PC today and the Jetson later; only the
  input device changes.

## How it works

```text
①  Analog front-end        ②  Jetson (DSP + LLM)           ③  Chat
   mic → MAX9814 →            capture → features.analyze()     "what's that noise?"
   RC filter → USB      →     → JSON facts → Ollama tools  →   plain-language answer
   (you solder)              (Python, offline)                (web terminal · ttyd)
```

The LLM never touches raw audio — `features.analyze()` is the bridge that turns sound into the
JSON the model reasons over.

## Getting started

Works on your Mac today using the built-in mic; identical on the Jetson later.

**Prerequisites:** [git](https://git-scm.com), Python 3.9+, and [Ollama](https://ollama.com).

```bash
# 1. Clone the repo
git clone <repo-url> acoustic-analyzer
cd acoustic-analyzer

# 2. Local LLM
#    macOS:        brew install ollama
#    Linux/Jetson: curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b

# 3. Python environment
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

# 4. Run the agent
./.venv/bin/python src/agent.py
#    you> what's that noise?
#    you> show me the spectrum        (saves spectrum.png)
```

### Configuration

| What | How |
|------|-----|
| **Model** | `--model <name>` on `src/agent.py` (default `qwen2.5:3b`) |
| **Mic / input device** | `--device N` (find N via `src/capture_test.py --list`) |
| **Ollama location** | defaults to `http://localhost:11434`; set `OLLAMA_HOST` for a remote or containerized Ollama |

### The tool service + terminal agent, locally (no mic needed)

[`src/service.py`](src/service.py) exposes the DSP over HTTP — `/analyze`, `/capture`, `/spectrum`.
With `ACOUSTIC_DEMO=1` it falls back to a synthetic 60 Hz hum, so you can exercise the whole loop
before the board is soldered:

```bash
PYTHONPATH=src ACOUSTIC_DEMO=1 ./.venv/bin/uvicorn service:app --host 0.0.0.0 --port 8000
curl -s localhost:8000/healthz
curl -s -X POST localhost:8000/capture -H 'Content-Type: application/json' -d '{"seconds":5}'
open http://localhost:8000/scope        # live scrolling spectrogram
```

Then point the reusable terminal agent at it — the same UI you get on the Jetson:

```bash
PYTHONPATH=src AGENT_NAME="Acoustic Analyzer" \
  TOOL_SERVERS=http://localhost:8000 OLLAMA_HOST=http://localhost:11434 \
  SYSTEM_PROMPT_FILE=deploy/agents/acoustic.prompt \
  SUGGESTIONS_FILE=deploy/agents/acoustic.suggestions \
  ./.venv/bin/python src/agent_shell.py
```

## Testing the soldered hardware (no Jetson needed)

The front-end presents as a standard **USB microphone**, so you can verify your soldering on a
**Mac or Windows PC** — plug in the USB audio adapter and run the capture tests. Clear,
OS-by-OS instructions (including the Windows/WSL caveats) are in
**[docs/mic-frontend/TEST-HARDWARE.md](docs/mic-frontend/TEST-HARDWARE.md)**.

## Repository layout

| File | Purpose |
|------|---------|
| [`src/capture_test.py`](src/capture_test.py) | Prove the mic works — level meter, record, device list |
| [`src/spectrogram.py`](src/spectrogram.py) | Live scrolling FFT spectrogram |
| [`src/features.py`](src/features.py) | `analyze()` — turns audio into structured JSON facts |
| [`src/acoustic_tools.py`](src/acoustic_tools.py) | LLM tools: `capture_and_analyze`, `get_spectrum_image`, `list_input_devices` |
| [`src/agent.py`](src/agent.py) | Terminal chat (local dev) — Ollama loop that calls the tools |
| [`src/service.py`](src/service.py) | FastAPI HTTP tool service — `/analyze`, `/spectrum` (the deployed DSP API) |
| [`src/agent_shell.py`](src/agent_shell.py) | Reusable terminal-agent engine — config + OpenAPI tool auto-discovery (the deployed front-end) |
| [`requirements.txt`](requirements.txt) | Python dependencies |

## Documentation

**[docs/README.md](docs/README.md) is the map.** The repo now hosts **two build tracks** on the one
Jetson — the acoustic analyzer (electrical) and a mechanical build — sharing the hardware, the
cluster, the local model, and where possible the DSP code.

| Doc | Contents |
|-----|----------|
| [docs/README.md](docs/README.md) | **Documentation map** — both tracks, shared platform, diagrams |
| **Acoustic (electrical)** — [docs/PROJECT.md](docs/PROJECT.md) | Full design, soldering, BOM, what to ask it, roadmap, future ideas |
| [HARDWARE-BENCH.md](docs/mic-frontend/HARDWARE-BENCH.md) | Bench companion: pinouts, soldering how-to, symptom→fix troubleshooting |
| [docs/mic-frontend/TEST-HARDWARE.md](docs/mic-frontend/TEST-HARDWARE.md) | Test the soldered board on a Mac or Windows PC (no Jetson) |
| [docs/mic-frontend/acoustic-wiring.svg](docs/mic-frontend/acoustic-wiring.svg) | Pin-level solder map + checklist |
| [DISPLAY-BUILD.md](docs/display/DISPLAY-BUILD.md) | Sound-reactive OLED + LED bar: schematic, BOM, current budget, bring-up |
| **Mechanical** — [docs/mechanical/README.md](docs/mechanical/README.md) | **Start here** for the mechanical track: orientation, control architecture, first-hour exercise |
| [docs/mechanical/OPTIONS.md](docs/mechanical/OPTIONS.md) | Three candidate rigs — BOMs, milestones, trade-offs, recommendation |
| **Platform** — [JETSON-SETUP.md](docs/platform/JETSON-SETUP.md) | Bootstrap the Jetson Orin Nano (Apple-Silicon-friendly) |
| [docs/platform/DEPLOY-K3S.md](docs/platform/DEPLOY-K3S.md) | Deploy the stack on k3s (Ollama + tool service + terminal UI, on NVMe) |

## Troubleshooting

- **`connection refused` / agent hangs** → Ollama isn't running. Start it (`ollama serve` on Mac,
  or `sudo systemctl status ollama` on Linux) and confirm `curl localhost:11434/api/tags`.
- **`model not found`** → `ollama pull qwen2.5:3b` (or match the `--model` you passed). In the
  deployed TUI, type `model` to list what Ollama actually has and switch.
- **`sounddevice`/PortAudio import error** → macOS: `pip install sounddevice` again; Linux/Jetson:
  install `portaudio19-dev`, then reinstall.
- **No sound / flat level meter** → wrong input device; run `src/capture_test.py --list` and pass
  `--device N`. On macOS grant mic permission when prompted.
- **No microphone found on the Jetson** → the USB audio adapter provides the mic input (the Jetson
  has none). Check `arecord -l` on the host: a capture device appears as `pcmC?D?c` (`c` = capture).
  See [docs/PROJECT.md](docs/PROJECT.md) for why.
- **Model calls no tool / rambles** → small models are weak at tool-calling; keep prompts direct
  ("what's that noise?"). `qwen2.5:3b` is the floor — `qwen2.5:7b` follows instructions better.
- **Slow on Jetson** → ensure `nvidia-jetpack` is installed and `jtop` shows GPU use; set
  `sudo nvpmodel -m 0 && sudo jetson_clocks`.

## Roadmap

- [x] DSP pipeline — capture, spectrogram, feature extraction
- [x] Local tool-calling agent (Ollama + `qwen2.5:3b`)
- [x] HTTP tool service (FastAPI over the tools — `/analyze`, `/spectrum`)
- [x] **Deployed on k3s** (Jetson · GPU · NVMe): Ollama + tool service + terminal UI
- [x] **Terminal-UI front-end** (a `ttyd` web terminal — open a URL, get the agent)
- [ ] Real mic (soldered MAX9814 front-end) — running in **demo mode** until then
- [ ] Log-mel CNN sound classifier
- [ ] NFC "tap-to-analyze" from an iPhone

## Hardware

A ~$25 analog front-end you solder:
**electret mic → MAX9814 preamp → anti-alias RC filter → 3.5 mm jack → USB audio adapter → Jetson.**
The [wiring map](docs/mic-frontend/acoustic-wiring.svg), bill of materials, and soldering steps are in
[docs/PROJECT.md](docs/PROJECT.md).

---

<div align="center">
<em>Built on a Mac, deploys to the Jetson unchanged — the DSP distills, the local LLM explains.</em>
</div>
