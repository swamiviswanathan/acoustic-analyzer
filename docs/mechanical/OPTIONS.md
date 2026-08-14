# Mechanical build — three candidate projects

> **New here?** Read **[README.md](README.md)** first — it explains the repository, the control
> architecture, and the vocabulary this document assumes. This page is the decision doc.

Three candidate projects for the mechanical track, each sharing the Jetson, the k3s cluster, and the
local AI model with the existing [acoustic analyzer](../PROJECT.md). For two of the three, the
signal-processing code is shared too. Bills of materials, milestone ladders, and honest trade-offs
for each, so one can be chosen deliberately. **Nothing here is decided.**

> **Constraint assumed throughout:** *no CAD and no 3D printer yet.* Every bill of materials below
> is off-the-shelf parts plus hand-tool work — drill, hacksaw, files, vise, hex keys. CAD is listed
> as a stretch goal in each option, never as a prerequisite.

---

## 0 · The control stack (common to all three)

Summarized here for reference; the reasoning is in [README.md §2](README.md#2--the-one-fact-that-shapes-every-design-decision).

```
   JETSON (Linux, non-real-time)   →  intent: "spin to 2400 rpm"        [software]
        │ USB-serial
   MICROCONTROLLER (RP2040/ESP32)  →  real-time loop: PWM, PID @ 1 kHz  [firmware]
        │ 3.3 V logic
   MOTOR DRIVER (DRV8825/TB6612)   →  current, heat, flyback            [electrical]
        │ separate motor supply
   MECHANISM                       →  torque, ratios, backlash, stiffness  [MECHANICAL]
        │
   SENSORS (encoder, accel, load cell) ──► back up to the top
```

The four non-negotiable safety rules — separate motor supply, single-point ground, physical e-stop,
bolted-down mechanism — are in [README.md §7](README.md#7--safety-ground-rules). They apply to all
three options without exception.

---

## 1 · Option A — Rotor rig (vibration & balancing bench)

**A motor spins a disc that is deliberately unbalanced, and the Jetson diagnoses what's wrong.**

```
  motor ──flex coupling──┬── shaft ──┬── DISC (add/remove weights) ──┬── shaft ──┬
                         │           │                               │           │
                    pillow block  IR tach                       pillow block
                     bearing     (once-per-rev)                  bearing
                         │                                           │
                    ═════╧═══════════════ base plate ════════════════╧═════
                                          │
                              piezo/accel pickup ──► USB audio in ──► Jetson
```

### Why this option is unusual: the analysis code already exists

[src/features.py](../../src/features.py) already turns a sampled signal into structured facts —
dominant frequency, the strongest peaks, energy per band, loudness. **A vibration signal is just
another waveform.** Feed it an accelerometer instead of a microphone and that same FFT pipeline
becomes a machine-health diagnostician. No new signal processing is needed to get started.

Better still, the cheapest usable vibration pickup is a **$2 piezo disc**, and it plugs into the
**USB audio adapter the acoustic track already built** — so the first milestone needs *zero* new
electronics and samples at 44.1 kHz, far better than any hobby I²C accelerometer manages.

### The mechanical content is core coursework

This is Vibrations and Machine Dynamics, taught by hand:

| Fault created on purpose | Vibration signature | Concept |
|---|---|---|
| Add a bolt or tape to the disc | Peak at **1× rotation speed**, growing with mass and radius | Unbalance; centrifugal force = *mrω²* |
| Shim the motor off-axis | **2×** appears, plus an axial component | Shaft misalignment |
| Loosen a pillow-block bolt | **½×, 1×, 2×, 3×…** — a picket fence, plus rattle | Mechanical looseness |
| Run a deliberately gritty bearing | High-frequency bursts with **sidebands** | Bearing defect frequencies (BPFO/BPFI) |
| Sweep speed slowly up and down | A peak that stays at a **fixed Hz** while others track speed | **Resonance vs. forced response** |

That last row is the payoff. Order-tracked peaks move with rpm; a resonance sits still. Seeing that
distinction on a runup/coastdown plot is the moment a natural frequency stops being an eigenvalue on
a page and becomes a thing a machine does. It's hard to get that from a textbook.

Then the capstone: **single-plane balancing by the influence-coefficient method.** Measure amplitude
and phase at 1×, add a known trial weight, measure again, solve for the correction mass and angle,
install it, watch the 1× peak collapse. That requires a phase reference, which is why the IR
tachometer is in the bill of materials — a genuinely good reason for a sensor to exist.

### Bill of materials — Option A

| Part | Qty | ~$ | Notes |
|---|---|---|---|
| Piezo disc element, 27 mm | 2 | 2 | Contact vibration pickup → 3.5 mm plug → the existing USB adapter |
| **Milestone-1 rotor:** 120 mm PC case fan, 12 V | 1 | 8 | Zero-machining starter rotor — tape a washer to one blade |
| 12 V DC motor, ~3000–6000 rpm, 5 mm shaft | 1 | 12 | Plain brushed DC. Avoid a gearmotor — gear mesh pollutes the spectrum |
| Flexible shaft coupling, 5 mm → 8 mm | 1 | 7 | Also the misalignment-tolerance lesson |
| Steel shaft, 8 mm × 300 mm | 1 | 8 | Cut to length with a hacksaw |
| Pillow-block bearings, 8 mm (KP08 / KFL08) | 2 | 10 | Bolt-down, self-aligning — no press fit, no machining |
| Shaft collars, 8 mm | 2 | 5 | Axial location |
| Rotor disc — 100–150 mm acrylic or plywood round | 2 | 6 | Hand-drill a ring of holes for M4 weights; no CAD needed |
| M4 bolts, nuts, washers — the weight set | — | 6 | The unbalance you add and remove |
| MDF or plywood base, 300 × 200 mm, + rubber feet | 1 | 10 | Deliberately *changeable* — swapping in a stiffer base shifts the resonance |
| IR reflective sensor (TCRT5000 module) | 1 | 4 | Once-per-rev tachometer for phase reference; reflective tape on the shaft |
| RP2040 (Pico) or ESP32 | 1 | 6 | PWM speed command, tach counting, e-stop logic |
| MOSFET or TB6612 driver + 12 V 2 A supply | 1 | 18 | Speed control |
| Toggle e-stop switch, inline on motor power | 1 | 4 | |
| Polycarbonate or acrylic guard sheet | 1 | 10 | **Required** — bolts leave spinning discs |
| *Optional:* MPU6050 or ADXL345 accelerometer | 1 | 5 | Calibrated low-frequency g-units; 1 kHz sample rate caps you near 500 Hz |
| *Optional:* ADXL1002 (±50 g, 11 kHz bandwidth) | 1 | 35 | Analog accelerometer into the USB audio path — the "proper" sensor |
| | | **~$115** | ~$85 without the optional accelerometers |

### Milestones — Option A

| # | Deliverable | Effort |
|---|---|---|
| **M0** | PC fan + a scrap of tape on one blade + a piezo taped to the frame → **watch the 1× peak appear** in the existing spectrogram | one evening, ~$10 |
| **M1** | Real rig assembled: motor → coupling → shaft → two pillow blocks → disc, bolted down. Runs smoothly, guard fitted | 1 weekend |
| **M2** | Firmware: PWM speed command plus IR tachometer → true rpm reported to the Jetson over USB-serial | 1 weekend |
| **M3** | Unbalance experiment: sweep bolt mass and radius, plot 1× amplitude against *mrω²*, confirm the square law | 1 weekend |
| **M4** | **Runup/coastdown waterfall plot** — separate the fixed resonance from the speed-tracking orders | 1 weekend |
| **M5** | **Single-plane balancing** using amplitude and phase; correct the unbalance and quantify the improvement in dB | 1–2 weekends |
| **M6** | Fault library — unbalance, misalignment, looseness, each with a labelled spectrum — plus a `diagnose_vibration()` function the AI model can call | 1–2 weekends |
| **S1** | *Stretch:* swap the base for a stiffer or softer one and **predict** the resonance shift before measuring it | |
| **S2** | *Stretch, needs CAD:* a designed rotor disc with properly balanced tapped holes | |

### Trade-offs — Option A

**For:** cheapest path to a working result; reuses [features.py](../../src/features.py) unchanged;
the physics is unusually *visible*; small desktop footprint; a real industrial technique — machine
condition monitoring is precisely this; and it scales in difficulty from one evening to a semester.

**Against:** it's a *test bench*, not a robot — less immediately impressive to demonstrate than an
arm that picks things up, because the payoff lives in the plots. It rewards someone who enjoys
measurement. Spinning mass demands the guard and the bolted-down discipline from day one.

---

## 2 · Option B — Robot arm, 3–4 DOF, camera-guided pick-and-place

**A camera finds an object, the Jetson computes where it is, and the arm goes and gets it.**

```
   CSI camera ──► Jetson: detect object → (x,y,z) → inverse kinematics → joint angles
                    │
                    │ I²C
                    ▼
             PCA9685 servo driver ──► 4 × servos ──► aluminium bracket arm ──► gripper
                    ▲                                        │
             6 V 5 A supply (never the Jetson rail)     bolted to a heavy base
```

### Mechanical content

The core machine-design toolkit:

- **Forward and inverse kinematics** — the two-link planar solution via the law of cosines, then
  elbow-up/elbow-down branch selection and mapping the reachable workspace. Genuinely satisfying
  geometry.
- **Torque budget** — static torque at the shoulder is Σ(mass × moment arm). This is the calculation
  that determines whether the arm can lift anything at all, and the one most first-time builders
  skip before buying servos.
- **Gear reduction and backlash** — measure the play at the gripper with a dial indicator, then trace
  it back through the servo gear train and bracket clearances to find where it comes from.
- **Structural deflection** — load the arm and measure the sag. *Stiffness*, not strength, is what
  limits accuracy; this surprises people.
- **Repeatability metrology** — command the same pose thirty times and measure the scatter.
  Distinguishing *accuracy* from *repeatability* is a real professional distinction and this rig
  makes it concrete.
- **Gravity compensation** — a feed-forward term proportional to cos(θ). Where mechanics meets
  control.

### Bill of materials — Option B

| Part | Qty | ~$ | Notes |
|---|---|---|---|
| Aluminium robot-arm bracket kit (multi-purpose servo brackets) | 1 | 70 | **The no-CAD enabler** — pre-made brackets, fasteners, bearings included |
| Digital servos, 20 kg·cm metal-gear (DS3218 / MG996R class) | 4 | 60 | Buy the torque. Underspecified servos are the most common arm failure |
| Micro servo for the gripper | 1 | 6 | |
| PCA9685 16-channel PWM/servo driver (I²C) | 1 | 6 | Offloads all PWM timing from the Jetson |
| 6 V 5–6 A regulated supply | 1 | 18 | Servos draw multiple amps stalled. Separate rail, common ground |
| 1000 µF bulk capacitor on the servo rail | 1 | 2 | Prevents brownout resets — a good electrical lesson |
| Heavy base — 300 mm MDF plus a steel plate or clamps | 1 | 15 | An arm lighter than its reach tips over |
| CSI camera IMX219, or any USB webcam | 1 | 25 | The Jetson's actual specialty |
| RP2040 / ESP32 — optional but recommended | 1 | 6 | Limit switches, e-stop, smooth trajectory interpolation |
| Dial indicator with magnetic base | 1 | 30 | **The metrology tool** — makes backlash and repeatability measurable instead of anecdotal |
| Fasteners, cable management, ties | — | 10 | |
| | | **~$248** | ~$218 without the dial indicator — but buy the dial indicator |

### Milestones — Option B

| # | Deliverable | Effort |
|---|---|---|
| **M1** | One servo moving on command, Jetson → PCA9685; separate power rail proven to hold up under stall | 1 weekend |
| **M2** | Arm assembled, all four joints commandable, joint limits enforced in software | 1–2 weekends |
| **M3** | Forward kinematics: command joint angles, predict the tool position, verify with a ruler | 1 weekend |
| **M4** | Inverse kinematics: enter an (x, y, z) and the tool goes there. Map the reachable workspace | 2 weekends |
| **M5** | Torque budget against reality: measure maximum payload at full extension, compare with the calculation | 1 weekend |
| **M6** | Metrology: backlash and repeatability measured with the dial indicator; deflection-versus-payload curve | 1 weekend |
| **M7** | Camera → colour or shape detection → coordinates → pick and place | 2–3 weekends |
| **M8** | A `pick_up(object)` function the AI model can call — "put the red block in the cup" through the chat interface | 1–2 weekends |
| **S1** | *Stretch:* trajectory smoothing with a trapezoidal velocity profile, plus gravity feed-forward | |
| **S2** | *Stretch, needs CAD:* replace one bracket link with a designed part and compare stiffness | |

### Trade-offs — Option B

**For:** by far the most impressive to demonstrate; teaches the mechanical core — kinematics, torque,
gear trains, tolerance, metrology; and it's the only option that genuinely uses what the Jetson is
*for*, namely camera inference.

**Against:** roughly 2.5× the cost and 3× the build time of Option A; almost no reuse of existing
code, so the vision and kinematics stack is built from scratch; and the mechanical learning is partly
*bought* rather than *made*, because the brackets are off-the-shelf — which is exactly the gap CAD
and printing would close in a second stage. Hobby servos also have sloppy backlash, so absolute
accuracy will disappoint unless the goal is framed as *measuring* that sloppiness, which is the
honest and more interesting framing anyway.

---

## 3 · Option C — Thrust / load test stand

**A motor and propeller on a load cell; sweep the throttle and measure the thrust curve.**

```
                       ┌── prop ──┐   (inside a guard)
        motor + ESC ───┤          │
             │         └──────────┘
        arm on a pivot ─────────────► pushes on a LOAD CELL
             │                              │
        MCU: throttle PWM + rpm             │ HX711 24-bit amplifier
             └──────────────► Jetson ◄──────┘
                       thrust vs. rpm vs. power → efficiency curve
```

### Mechanical content

- **Statics** — free-body diagram of the pivot arm, moment balance, why load-cell placement sets
  your resolution, and why the pivot must be genuinely low-friction.
- **Calibration and metrology** — hang known masses, fit the calibration line, then quantify
  hysteresis, drift, and repeatability. The most rigorous *measurement* practice of the three.
- **Fluid mechanics** — the propeller relations give thrust ∝ ρn²D⁴; measure the exponents and see
  how close reality gets.
- **Efficiency** — thrust per watt, measuring current and voltage too. The curve that actually
  matters for anything that flies.
- **Structures** — arm stiffness and the arm's own resonance corrupting the reading. A real lesson in
  measurement-system design.

### Bill of materials — Option C

| Part | Qty | ~$ | Notes |
|---|---|---|---|
| Load cell, straight-bar, 5 kg | 1 | 6 | |
| HX711 24-bit load-cell amplifier | 1 | 5 | |
| Calibration masses (or known weights plus a kitchen scale) | 1 | 20 | Without calibration this is a toy |
| Brushless motor plus ESC, 2212 ~1000 kV class | 1 | 25 | |
| Propellers, 8–10 inch, assorted pitches | 4 | 12 | Varying diameter and pitch *is* the experiment |
| LiPo pack or 12 V 20 A bench supply | 1 | 35 | Propellers draw serious current |
| 2020 aluminium extrusion, brackets, pivot bearing | — | 30 | No machining — hacksaw and hex keys |
| Heavy base plus C-clamps to the bench | 1 | 15 | |
| **Polycarbonate guard enclosure** | 1 | 25 | **Mandatory, not optional** |
| INA219 or a shunt for volts and amps | 1 | 6 | Enables the efficiency curve |
| RP2040 / ESP32 plus e-stop | 1 | 10 | |
| | | **~$189** | |

### Milestones — Option C

| # | Deliverable | Effort |
|---|---|---|
| **M1** | Load cell and HX711 reading grams on the Jetson, calibrated against known masses with a documented error band | 1 weekend |
| **M2** | Stand built, motor mounted, guard fitted, e-stop verified **before the first spin-up** | 1 weekend |
| **M3** | First thrust curve: throttle sweep → thrust against rpm | 1 weekend |
| **M4** | Add volts and amps → thrust-per-watt efficiency curves | 1 weekend |
| **M5** | Compare three or four propellers; test the ρn²D⁴ relation against measured data | 1–2 weekends |
| **M6** | A `run_thrust_sweep()` function the AI model can call, so it writes the test report and picks the best propeller | 1 weekend |

### Trade-offs — Option C

**For:** the strongest *experimental method* content — calibration, uncertainty, curve fitting. It's
the only option that touches fluid mechanics. Cheap-ish, and quick to first data.

**Against:** **the least safe of the three by a wide margin.** A propeller at 10,000 rpm is a genuine
injury and projectile risk, and a thrown blade is not hypothetical. It's also loud enough to become a
household problem. And it's the least *mechanism* of the three: once built it mostly sits still, so
there's no kinematics, no gear train, no linkage. It shares nothing with the existing codebase.

---

## 4 · Side-by-side

| | **A · Rotor rig** | **B · Robot arm** | **C · Thrust stand** |
|---|---|---|---|
| Cost | **~$85–115** | ~$220–250 | ~$190 |
| Time to first result | **1 evening** | 2–3 weekends | 1–2 weekends |
| Time to "done" | 6–9 weekends | 12–18 weekends | 6–8 weekends |
| Mechanical topics | Vibrations, balancing, resonance, bearings | **Kinematics, torque, gears, backlash, metrology** | Statics, calibration, fluids |
| Reuses existing code | **Yes — `features.py` as-is** | No | No |
| Uses the Jetson's GPU | Barely — the FFT is trivial | **Yes — camera inference** | No |
| Safety burden | Moderate — guard, bolted down | **Low** | **High — spinning propeller** |
| Needs CAD to shine | Later, optional | Later, would help a lot | No |
| Demonstration appeal | Low–medium; the payoff is in the plots | **High** | Medium, and loud |
| Electrical-track overlap | Sensor front-end, motor driver | Servo power, current budget | Load-cell analog, power measurement |

**Recommendation: A, then B.** Option A costs almost nothing, produces a real result in a single
evening, teaches the one concept the others don't — resonance you can *see* — and reuses the existing
signal-processing layer, which makes the two tracks genuinely one system rather than two projects
sharing a box. By the time A is finished, the microcontroller link, the k3s service pattern, and the
AI tool layer all exist, so B becomes a project about kinematics and CAD instead of a project about
plumbing.

**Option C is the one to skip:** highest risk, least mechanism, no code reuse.

---

## 5 · Sharing the Jetson

Both tracks run on the one Orin Nano as separate k3s services in the existing namespace, alongside
[acoustic-tools.yaml](../../deploy/k8s/acoustic-tools.yaml) and
[ollama.yaml](../../deploy/k8s/ollama.yaml):

```
   Jetson Orin Nano · k3s
   ├── ollama              (shared — qwen2.5:3b, ~2 GB, one copy serves both agents)
   ├── acoustic-tools      :30080  ← acoustic DSP service
   ├── acoustic-chat       :30088  ← acoustic terminal UI
   ├── mech-tools          :30081  ← mechanical rig service   (new)
   └── mech-chat           :30089  ← mechanical terminal UI    (new)
```

**Genuinely shared:** Ollama and the model weights — the big win, one 2 GB model serving two agents —
plus the k3s cluster, the NVMe storage, the tool-calling pattern in [src/agent.py](../../src/agent.py),
and for Option A the signal processing itself.

**Must stay separate:** USB device claims (pin each device by ID, never by index — indices reshuffle
on reboot), NodePorts, and the microcontroller serial ports.

**Expected contention:** with a 3-billion-parameter model, inference is short and two agents
interleave comfortably. They'll queue behind each other for a second or two. Not a problem for
interactive use.

### When a second Jetson becomes worth buying

Sharing is the right call for now. The specific trigger is **Option B's camera loop**: continuous
vision inference wants the GPU and a couple of gigabytes of RAM *persistently*, and on an 8 GB board
that starts competing with Ollama for memory. One frame on request is fine; a live loop is not.
Concretely, add the second board when any of these becomes true:

1. The arm needs **live camera inference** rather than single frames — the real trigger.
2. Both tracks are being worked **at the same time** and are colliding over USB devices or reboots.
3. The rigs end up in **different rooms** — a bench rig and a desk robot rarely want to share a box.

Until then, keeping the service boundaries clean — each track in its own container, on its own port,
with its own config and no shared filesystem state — means moving the mechanical half to a second
board later is a `kubectl apply` against a different cluster rather than a rewrite. Worth designing
for from day one even if the second board is never bought.

---

## 6 · Open questions before committing

1. **Which option** — A, B, or C. Recommendation above: A, then B.
2. **Bench access** to a drill, hacksaw, files, and a vise. Every bill of materials assumes it; if
   not available, add roughly $80 of hand tools or shift toward pre-drilled extrusion.
3. **How much of the firmware should be written from scratch?** Writing the PID loop by hand is an
   excellent lesson and adds a few weeks; starting from a working template is a legitimate choice.
4. **CAD and 3D printing as a deliberate second stage?** Both A and B improve substantially with
   them. Better planned than perpetually deferred.
