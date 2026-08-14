# Mechanical build — start here

**You are the mechanical engineer on this project. This page is the orientation; read it once,
top to bottom, before anything else.** It takes about fifteen minutes and ends with something you
can run on your laptop today, with no parts and no purchases.

---

## 1 · Why this repo is called "acoustic-analyzer"

Fair question. This repository started as **one** build and is becoming a home for **two**, running
on the same computer:

| Track | What it is | Status |
|---|---|---|
| **Acoustic analyzer** (electrical) | A hand-soldered microphone front-end feeding a signal-processing pipeline, which a local AI model reads and explains in plain language | Built and deployed |
| **Mechanical build** (this track) | A machine — something that spins, moves, or pushes — instrumented so the same pipeline can measure and diagnose it | Being scoped; **the option is not chosen yet** |

The two tracks are deliberately coupled rather than independent. They share one **NVIDIA Jetson Orin
Nano** (a small Linux computer with a capable GPU), one local **AI model**, and — depending on which
option is chosen — one **signal-processing library**. The electrical track builds the *instrument*;
the mechanical track builds the *machine that gets measured*. The interface between them is a small,
well-defined piece of code, which is a much more interesting way to work than two isolated projects
that happen to sit on the same desk.

So: the repo name is historical. Don't read anything into it.

---

## 2 · The one fact that shapes every design decision

**The Jetson does not drive motors.** Internalize this before reading any further, because every
design in this project follows from it.

Three independent reasons:

1. **Linux is not a real-time operating system.** If you generate stepper-motor pulses from Python,
   the kernel's scheduler will occasionally preempt your loop for a few milliseconds. That shows up
   as lost steps, audible cogging, and a control loop that oscillates instead of settling. The
   timing has to come from hardware that has nothing else to do.
2. **The pins are 3.3 V at a few milliamps.** Motors want volts at amps.
3. **Motors push back.** Collapsing a motor's magnetic field generates a voltage spike in the
   opposite direction (back-EMF). Connected to a general-purpose pin, that spike will destroy the
   board.

So the system is built in **layers**, and — conveniently — each layer is a different discipline:

```
   ┌────────────────────────────────────────────────────────────────┐
   │  JETSON ORIN NANO  ·  Ubuntu Linux + k3s        [software]     │
   │  cameras, planning, FFT analysis, the AI model, the chat UI    │
   │  speaks INTENT:  "go to 120°"  ·  "spin to 2400 rpm"           │
   └───────────────────────────┬────────────────────────────────────┘
                               │  USB-serial — intent down,
                               │  telemetry up, ~10–50 times/sec
   ┌───────────────────────────▼────────────────────────────────────┐
   │  MICROCONTROLLER  ·  RP2040 (Pico) or ESP32     [firmware]     │
   │  the real-time loop, 1000 times/sec, never interrupted:        │
   │  generate PWM · count encoder pulses · run PID · watchdog      │
   └───────────────────────────┬────────────────────────────────────┘
                               │  3.3 V logic signals
   ┌───────────────────────────▼────────────────────────────────────┐
   │  MOTOR DRIVERS  ·  DRV8825 / TB6612 / ESC       [electrical]   │
   │  current limiting, heat, flyback protection, grounding         │
   └───────────────────────────┬────────────────────────────────────┘
                               │  motor power — its own supply
   ┌───────────────────────────▼────────────────────────────────────┐
   │  MECHANISM                                      [MECHANICAL ←] │
   │  motors, couplings, bearings, shafts, gears, belts, linkages,  │
   │  structure — torque budget · gear ratio · backlash · stiffness │
   └───────────────────────────┬────────────────────────────────────┘
                               │
   ┌───────────────────────────▼────────────────────────────────────┐
   │  SENSORS  ·  encoder, tachometer, accelerometer, load cell     │
   │  ─────────────────────────────► closes the loop back to the top│
   └────────────────────────────────────────────────────────────────┘
```

The bottom two boxes are yours. The layer above them is where this track meets the electrical
track. This is not a workaround or a simplification for beginners — it is how real machines are
built, from CNC mills to spacecraft: a general-purpose brain for perception and planning, and a
dedicated deterministic controller for motion.

---

## 3 · What you need coming in, and what you don't

**You need:**

- Mechanical fundamentals: statics, dynamics, and ideally a first course in vibrations. If you
  haven't had vibrations yet, that's fine — one of the options *teaches* it by hand.
- Willingness to read (not write) some Python. Roughly a hundred lines matter to you.
- Bench access: a drill, a hacksaw, files, a vise, hex keys, a multimeter. Every bill of materials
  assumes these. If you don't have them, budget about $80 more.
- Patience with the fact that the first version will be built from off-the-shelf parts, because
  there's no 3D printer yet. Designing your own parts is a planned second stage, not a prerequisite.

**You explicitly do not need:**

- To write the signal-processing code — it exists and works.
- To understand Kubernetes, containers, or the AI model internals. Someone else runs that layer;
  you'll be handed a URL and a command.
- To design circuit boards. The electrical layer is the other track's territory.
- A 3D printer or CAD skills to start.

---

## 4 · Vocabulary you'll hit immediately

The software side of this project uses words that don't appear in a mechanical curriculum. Here are
the only ones that matter, in plain terms:

| Term | What it actually means here |
|---|---|
| **Jetson Orin Nano** | A small Linux computer (~8 GB RAM) with a GPU. The "brain." |
| **MCU / microcontroller** | A tiny chip that runs one program forever with exact timing. The "reflexes." An RP2040 costs $6. |
| **PWM** | Switching a voltage on and off very fast; the on-fraction sets effective motor power or servo angle. |
| **Encoder / tachometer** | Sensors that report shaft position / shaft speed back to the controller. |
| **PID loop** | The standard feedback recipe: measure error, react to its size, its accumulation, and its rate of change. |
| **FFT** | Fast Fourier Transform — converts a wiggly signal over time into "how much of each frequency is present." The single most useful tool in this project. |
| **Spectrum / peak** | The FFT's output, and a spike in it. A spike at 40 Hz means something is happening 40 times a second. |
| **k3s / container / pod** | How software gets packaged and run on the Jetson so two projects don't collide. You'll use one command; you don't need the theory. |
| **Ollama** | The program that runs the AI model locally on the Jetson. No internet, no accounts. |
| **Tool-calling** | The AI model can't read raw sensor data, so it calls small functions that return tidy numbers, then reasons about those. Your rig will expose one or two such functions. |
| **`analyze()`** | The specific function, in [src/features.py](../../src/features.py), that turns a recorded signal into those tidy numbers. This is the handshake between the two tracks. |

---

## 5 · Do this now — the first hour, no parts required

This works on any Mac or PC with a built-in microphone, and it previews the core idea of the whole
project. From the repository root:

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python src/spectrogram.py
```

A live scrolling spectrogram appears — frequency on one axis, time on the other, brightness for
intensity. Now:

1. **Whistle a rising note.** A bright line climbs. That's a single frequency moving.
2. **Hum near a power brick or a laptop charger.** A band near 60 Hz lights up. That's mains hum —
   the electrical world leaking into the acoustic one.
3. **Now the mechanical part.** Tap the table rhythmically. Drag a chair. Run a desk fan nearby and
   watch its blade-passing frequency appear as a steady line — then put a small piece of tape on
   one blade and watch that line get *stronger*.

**That last step is a vibration measurement.** You just detected a rotating imbalance using nothing
but a laptop microphone and an FFT. One of the three candidate projects is essentially that
experiment, done properly, with a real shaft and real bearings and real numbers. Which is the
fastest way to understand why the two tracks belong in one repository: the microphone the electrical
track soldered is an *instrument*, and machines are what instruments measure.

Then run this to see the same signal reduced to structured numbers:

```bash
./.venv/bin/python src/features.py
```

It records two seconds and prints a small block of JSON — dominant frequency, the strongest peaks,
energy per frequency band, loudness in dB. That JSON is the entire interface you'd inherit. Nothing
about it is specific to sound; it works on any sampled signal, including a vibration pickup bolted
to a machine.

---

## 6 · What to read next, in order

| Order | Document | Why | Time |
|---|---|---|---|
| 1 | **[OPTIONS.md](OPTIONS.md)** — *the decision doc* | Three candidate projects with bills of materials, milestone ladders, and honest trade-offs. **This is the document to have an opinion about.** | 25 min |
| 2 | [../PROJECT.md](../PROJECT.md) | The existing acoustic project. Read §"The idea in one picture" and the signal-chain diagram; skip the soldering detail unless curious. | 15 min |
| 3 | [src/features.py](../../src/features.py) | About 100 lines. Read `analyze()` and nothing else. This is the handshake. | 15 min |
| 4 | [../platform/JETSON-SETUP.md](../platform/JETSON-SETUP.md) | Only when you actually need to touch the Jetson. Skim now, return later. | skim |
| 5 | [../platform/DEPLOY-K3S.md](../platform/DEPLOY-K3S.md) | Same — reference material for when your rig's service needs to run on the Jetson. | skim |

---

## 7 · Safety ground rules

These are not boilerplate. Every option in this project involves stored energy — a spinning mass, a
loaded spring, a servo with enough torque to break a finger — and the failure modes are physical.

| Rule | Why it exists |
|---|---|
| Motors get their **own power supply**, never the Jetson's 5 V rail | Motor inrush current browns out the board; back-EMF can destroy it |
| Grounds tied at **one point**; signal ground kept separate from motor return | Otherwise motor current flows through your signal ground and every reading is garbage |
| A **physical e-stop** — a switch that cuts *motor power*, not a software command | Software cannot stop a machine whose software has hung |
| Nothing spins or moves until the mechanism is **bolted down** | An unsecured rig walking off the bench is the classic first failure, and it happens fast |
| Anything spinning gets a **guard** between it and you | Fasteners leave rotating discs. Assume they will |
| Power up the first time with a **hand on the e-stop** and eyes on the current draw | The first ten seconds tell you whether the wiring is right |

---

## 8 · Where the two tracks converge

The long-term goal, once a mechanical option is built:

```
   your machine  ──── vibration ────►  the soldered analog front-end
     (mechanism)                          (instrument)
          │                                    │
          └──────────► shared analyze() ◄──────┘
                              │
                    structured numbers
                              │
                        local AI model
                              │
          "the 1× peak doubled and a 2× component appeared —
           the coupling has gone out of alignment"
```

One person builds the sensing chain, the other builds the machine, and `analyze()` is the contract
between them. That's a genuine division of labour with a real interface — the same shape as
professional engineering work, and considerably more satisfying than two separate school projects.

---

## 9 · Decisions still open

Bring an opinion on these after reading [OPTIONS.md](OPTIONS.md):

1. **Which of the three options** — the recommendation there is the vibration rig first, the robot
   arm second, and skipping the thrust stand. Disagree if you have reason to.
2. **Bench access** — confirmed, or does the tool budget need to grow?
3. **How much firmware should be yours?** Writing the real-time control loop yourself is an
   excellent lesson and adds a few weeks. Starting from a working template is a legitimate choice.
   Either answer is fine; it should be a decision rather than an accident.
4. **CAD and 3D printing as a second stage?** Both leading options get substantially better with
   them. Worth planning deliberately instead of never getting to it.
