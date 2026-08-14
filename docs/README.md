# Documentation map

This repository hosts **two build tracks** running on one NVIDIA Jetson Orin Nano. They share the
hardware, the k3s cluster, the local AI model, and — where it makes sense — the signal-processing
code.

| Track | Discipline | Start here | Status |
|---|---|---|---|
| **Acoustic analyzer** | Electrical / signal processing | [PROJECT.md](PROJECT.md) | Built and deployed |
| **Mechanical build** | Mechanical | **[mechanical/README.md](mechanical/README.md)** | Being scoped — option not yet chosen |

New to one of the tracks? Open its "start here" document above and read it top to bottom before
anything else. Everything below is reference material you'll be pointed to when you need it.

---

## Acoustic analyzer (electrical track)

A hand-soldered microphone front-end feeding a DSP pipeline, which a local tool-calling LLM reads
and explains in plain language.

| Doc | Contents |
|---|---|
| [PROJECT.md](PROJECT.md) | **The hub** — full design, the soldering build, BOM, roadmap, alternatives considered |
| [mic-frontend/HARDWARE-BENCH.md](mic-frontend/HARDWARE-BENCH.md) | Bench companion: pinouts, soldering how-to, symptom→fix troubleshooting |
| [mic-frontend/TEST-HARDWARE.md](mic-frontend/TEST-HARDWARE.md) | Verify the soldered board on a Mac or Windows PC — no Jetson needed |
| [mic-frontend/acoustic-wiring.svg](mic-frontend/acoustic-wiring.svg) | Pin-level solder map plus an 8-step checklist, print-friendly |
| [display/DISPLAY-BUILD.md](display/DISPLAY-BUILD.md) | Sound-reactive OLED + LED bar: schematic, BOM, current budget, bring-up |
| [display/display-wiring.svg](display/display-wiring.svg) | Display wiring map |

## Mechanical build (mechanical track)

A machine — spinning, moving, or pushing — instrumented so the same pipeline can measure and
diagnose it.

| Doc | Contents |
|---|---|
| [mechanical/README.md](mechanical/README.md) | **Start here** — orientation, the control architecture, vocabulary, a first-hour exercise that needs no parts |
| [mechanical/OPTIONS.md](mechanical/OPTIONS.md) | The decision doc — three candidate projects with BOMs, milestones, trade-offs, and a recommendation |

## Shared platform

Applies to both tracks.

| Doc | Contents |
|---|---|
| [platform/JETSON-SETUP.md](platform/JETSON-SETUP.md) | Bootstrap the Jetson Orin Nano (Apple-Silicon-friendly) |
| [platform/DEPLOY-K3S.md](platform/DEPLOY-K3S.md) | Deploy on k3s — Ollama, tool service, terminal UI, on NVMe |

## Diagrams

| File | Contents |
|---|---|
| [architecture/acoustic-analyzer.drawio](architecture/acoustic-analyzer.drawio) | Editable source — page 1 system architecture, page 2 options considered |
| [architecture/acoustic-analyzer.svg](architecture/acoustic-analyzer.svg) | Rendered architecture diagram |
| [architecture/acoustic-analyzer.html](architecture/acoustic-analyzer.html) | Standalone viewer |
