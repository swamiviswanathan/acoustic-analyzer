"""
Stage 4 - expose the DSP to an LLM as callable TOOLS.

The language model never sees raw audio. It calls these functions, which
wrap features.analyze(), and reasons over the JSON facts they return.

These same functions plug into a standalone agent (agent.py) now, or into
Open WebUI later - the tool contract is identical.
"""

import json
import os
import sys
import threading
import time

import numpy as np

from features import analyze, BANDS, SAMPLE_RATE


def input_device():
    """Which input to record from: AUDIO_DEVICE (index or name substring), else the first
    device whose name contains 'USB', else PortAudio's default.

    Pinning this matters on the Jetson: PortAudio's default resolves to the onboard APE
    (card 2), which has nothing routed into it and delivers no samples - a blocking read
    there parks the caller forever with no error, which is exactly how the /scope waterfall
    died silently. The USB adapter is card 0.
    """
    import os
    import sounddevice as sd
    want = os.environ.get("AUDIO_DEVICE", "").strip()
    try:
        inputs = [(i, d["name"]) for i, d in enumerate(sd.query_devices())
                  if d["max_input_channels"] > 0]
    except Exception:
        return None
    if want:
        if want.lstrip("-").isdigit():
            return int(want)
        for i, name in inputs:
            if want.lower() in name.lower():
                return i
    for i, name in inputs:
        if "usb" in name.lower():
            return i
    return None


def _record(seconds, device=None):
    import sounddevice as sd  # imported lazily so wav-only use needs no PortAudio
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, device=input_device() if device is None else device)
    sd.wait()
    return audio.flatten()


def _read_wav(path):
    from scipy.io import wavfile
    sr, data = wavfile.read(path)
    if data.dtype.kind in "iu":  # int PCM -> float
        data = data.astype(np.float64) / np.iinfo(data.dtype).max
    samples = data[:, 0] if data.ndim > 1 else data
    return samples, sr


def _get_samples(seconds, wav_path, device):
    """Shared source: read a wav if given, else record from the mic."""
    if wav_path:
        samples, sr = _read_wav(wav_path)
        return samples, sr, wav_path
    return _record(seconds, device), SAMPLE_RATE, "microphone (%.1fs)" % seconds


# ---- the tools the LLM can call --------------------------------------------

def capture_and_analyze(seconds=2.0, wav_path=None, device=None):
    """Record `seconds` of mic audio (or read wav_path) and return acoustic facts."""
    try:
        seconds = float(seconds or 2.0)
        samples, sr, source = _get_samples(seconds, wav_path, device)
        facts = analyze(samples, sr)
        facts["source"] = source
        return facts
    except Exception as e:  # never crash the agent loop - report to the model
        return {"error": "%s: %s" % (type(e).__name__, e),
                "hint": "If there is no microphone/PortAudio, pass wav_path to analyze a .wav file."}


def get_spectrum_image(seconds=2.0, wav_path=None, out_path="spectrum.png", device=None):
    """Record (or read) audio and save a spectrogram PNG; return its path.

    Use when the user wants to SEE the sound (its spectrum / spectrogram / frequencies
    over time). The image spans 0-8 kHz on a dB scale.
    """
    try:
        import os
        import matplotlib
        matplotlib.use("Agg")  # headless: no window, just write the file
        import matplotlib.pyplot as plt
        from scipy.signal import spectrogram as _spec

        seconds = float(seconds or 2.0)
        samples, sr, source = _get_samples(seconds, wav_path, device)
        samples = np.asarray(samples, dtype=np.float64).flatten()

        f, t, sxx = _spec(samples, fs=sr, nperseg=1024, noverlap=512)
        keep = f <= 8000                      # most of the action is below 8 kHz
        sxx_db = 10 * np.log10(sxx[keep] + 1e-12)

        fig, ax = plt.subplots(figsize=(8, 4))
        m = ax.pcolormesh(t, f[keep] / 1000.0, sxx_db, shading="auto",
                          cmap="magma", vmin=-100, vmax=sxx_db.max())
        ax.set_xlabel("time (s)")
        ax.set_ylabel("frequency (kHz)")
        ax.set_title("Spectrogram - %s" % source)
        fig.colorbar(m, ax=ax, label="dB")
        fig.tight_layout()
        fig.savefig(out_path, dpi=110)
        plt.close(fig)

        return {"image_path": os.path.abspath(out_path), "source": source,
                "range": "0-8 kHz, dB scale",
                "note": "Spectrogram saved. Tell the user the file path so they can open it."}
    except Exception as e:
        return {"error": "%s: %s" % (type(e).__name__, e),
                "hint": "If there is no microphone/PortAudio, pass wav_path to render a .wav file."}


_DISPLAY = {"brightness": 8, "level": 0.0, "paused": False}   # lazily-opened OLED + SPI handles,
                                              # reused across calls; brightness/level persist so
                                              # set_display_brightness can re-light immediately;
                                              # paused controls the live feed from update_live_display


def _get_oled():
    """The cached panel handle, guaranteed to be powered ON.

    luma registers an atexit hook that sends DISPLAYOFF (0xAE) when a process holding a
    device exits - so any OTHER process that touches the panel (a bench run of
    `oled_display.py`, a debug script) leaves it dark. Our long-lived device object then
    keeps writing frames into display RAM with no error while nothing is visible. show()
    re-sends DISPLAYON, which is a single command byte and idempotent, so every render
    path calls this and self-heals instead of silently painting into the void.
    """
    if "oled" not in _DISPLAY:
        from oled_display import get_device
        _DISPLAY["oled"] = get_device()
    dev = _DISPLAY["oled"]
    dev.show()
    return dev


def _get_spi():
    if "spi" not in _DISPLAY:
        from led_strip import open_spi, all_off
        spi = open_spi()
        all_off(spi)   # never leave the strip in its undefined power-on state
        _DISPLAY["spi"] = spi
    return _DISPLAY["spi"]


# A running marquee is a THIRD writer to the panel, alongside the producer loop and the
# one-shot renders. Every mutator stops it (and waits for it to let go of the device) before
# touching the OLED itself - otherwise the scroll thread repaints over a clear/render.
_SCROLL = {"thread": None, "stop": None}
_SCROLL_LOCK = threading.Lock()   # FastAPI runs sync handlers in a threadpool: two chat
                                  # requests really can start/stop a marquee concurrently


def _stop_scroll():
    with _SCROLL_LOCK:
        thread, stop = _SCROLL["thread"], _SCROLL["stop"]
        _SCROLL["thread"] = _SCROLL["stop"] = None
    if stop is not None:
        stop.set()
    if thread is not None and thread.is_alive() and thread is not threading.current_thread():
        thread.join(timeout=2.0)


def _marquee(device, tile, span, stop, step, delay):
    """Slide a 128px window across `strip` until stopped. Never raises - a display fault must
    not kill the thread silently mid-frame without a log line."""
    from oled_display import WIDTH, HEIGHT
    try:
        # Pace against a moving DEADLINE, not by sleeping a fixed amount after each write.
        # Writing a frame costs ~29 ms, so `sleep(delay)` per frame silently added that to the
        # period and the strip crawled at roughly 60% of the requested px/s.
        next_frame = time.monotonic()
        while not stop.is_set():
            for off in range(0, span, step):
                if stop.is_set():
                    return
                device.display(tile.crop((off, 0, off + WIDTH, HEIGHT)))
                next_frame += delay
                wait = next_frame - time.monotonic()
                if wait <= 0:
                    next_frame = time.monotonic()   # behind schedule: reset, don't bank debt
                elif stop.wait(wait):               # doubles as frame delay AND cancel check
                    return
    except Exception as e:
        print("[_marquee] %s: %s" % (type(e).__name__, e), file=sys.stderr, flush=True)


def show_on_display(**_ignored):
    """Resume the live OLED spectrum + LED VU meter - they react continuously to the mic
    (driven by the same background loop that feeds /scope). Call this when the user asks
    to light up, turn on, resume, or show the display."""
    _stop_scroll()          # stop BEFORE unpausing, or scroll and producer both write at once
    # hand the bar back to the sound: "show the display" means live, not a held colour
    _DISPLAY.pop("rgb_override", None)
    _DISPLAY.pop("level_override", None)
    _DISPLAY["paused"] = False
    try:
        _get_oled()   # re-send DISPLAYON: another process may have left the panel dark
    except Exception as e:
        return {"display": "live", "warning": "%s: %s" % (type(e).__name__, e),
                "note": "Resumed, but the OLED could not be opened."}
    return {"display": "live", "note": "Updates automatically from the mic every ~1s."}


RGB_SMOOTH = 0.4      # per-frame EMA weight for the LED colour: responsive without flickering
RGB_GAMMA = 2.2       # >1 exaggerates the gap between bands so mixes read as a clear colour
                      # instead of washing out toward white


_BAND_AVG = [None, None, None]   # slow per-channel average of each group's share of total energy
BAND_AGC = 0.05                  # EMA rate for that average: ~7 s time constant at 3 fps


def band_rgb(band_energy):
    """Colour from the spectrum's SHAPE: bass -> red, mids -> green, treble -> blue.

    Each channel is scored against its OWN recent average, not against the other two. That
    distinction is the whole trick. Bass carries most of music's energy essentially all the
    time, so "which group is biggest" answers "bass" in almost every frame and the bar sits red
    through an entire track - which is exactly what the first two attempts here did. Comparing
    each group to its own baseline instead asks "is there more treble than usual right now?",
    which is the question whose answer actually changes with the music.

    The two scalars tried before this failed for the same reason in different clothes:
    dominant_hz locks onto the bass line and never moves, and the spectral centroid is an
    energy-weighted mean over the whole spectrum that drifted only 2% on this bench.

    Loudness deliberately plays no part here - it is the lit-pixel count's job.
    """
    if not band_energy:
        return (255, 0, 0)
    channels = [0.0, 0.0, 0.0]
    for name, energy in band_energy.items():
        low_edge = BANDS.get(name, (0, 0))[0]
        channels[0 if low_edge < 250 else 1 if low_edge < 2000 else 2] += max(0.0, float(energy))

    total = sum(channels)
    if total <= 0:
        return (0, 0, 0)
    share = [c / total for c in channels]

    relative = []
    for i, value in enumerate(share):
        if _BAND_AVG[i] is None:
            _BAND_AVG[i] = value
        else:
            _BAND_AVG[i] += BAND_AGC * (value - _BAND_AVG[i])
        relative.append(value / max(_BAND_AVG[i], 1e-4))   # 1.0 == this band's usual share

    peak = max(relative)
    if peak <= 0:
        return (0, 0, 0)
    return tuple(int(255 * min(1.0, r / peak) ** RGB_GAMMA) for r in relative)
MIN_FRAME_S = 0.035   # a full 128x64 frame costs ~29 ms over this I2C bus
SCROLL_PX_S = float(os.environ.get("SCROLL_PX_S", "30"))   # marquee speed, pixels per second


def _scroll_pacing(px_per_s):
    """Pixels per frame and seconds per frame for a given scroll speed.

    Speed is expressed in px/s rather than as a step+delay pair so the two knobs cannot
    disagree. One pixel per frame is the smoothest possible scroll, so we only take bigger
    steps when the panel physically cannot refresh fast enough for the speed asked for.
    """
    px_per_s = max(1.0, float(px_per_s))
    step = max(1, int(round(px_per_s * MIN_FRAME_S)))
    return step, step / px_per_s


def show_text_on_display(text=None, scroll=None, **_ignored):
    """Print text on the OLED, static or scrolling. Pauses the live spectrum first - otherwise
    the producer loop repaints bars over the message within ~1s - so the message stays up until
    show_on_display resumes the live feed."""
    text = ("" if text is None else str(text)).strip()
    if not text:
        return {"error": "no text given",
                "hint": "Pass the message to print, e.g. text='hello'."}
    try:
        _stop_scroll()              # a previous marquee must let go of the device first
        _DISPLAY["paused"] = True   # set before rendering so the next producer tick skips it
        from oled_display import show_text, marquee_tile
        device = _get_oled()

        lines = show_text(None, text, draw=False)   # measure only: does it fit the panel?
        overflows = "".join(text.split()) != "".join("".join(lines).split())
        if scroll is None:
            scroll = overflows      # auto: only switch modes when it would otherwise be cut off

        if not scroll:
            show_text(device, text)
            result = {"displayed": "\n".join(lines),
                      "note": "Text is on the OLED. The live spectrum is paused until "
                              "show_on_display is called."}
            if overflows:
                result["truncated"] = ("The panel fits 21 characters x 5 lines; the rest was "
                                       "dropped. Pass scroll=true to show it all.")
            return result

        stop = threading.Event()
        step, delay = _scroll_pacing(SCROLL_PX_S)
        tile, span = marquee_tile(text)
        thread = threading.Thread(target=_marquee,
                                  args=(device, tile, span, stop, step, delay),
                                  daemon=True)
        with _SCROLL_LOCK:
            _SCROLL["thread"], _SCROLL["stop"] = thread, stop
        thread.start()
        return {"displayed": text, "scrolling": True,
                "note": "Text is scrolling across the OLED on a loop. It keeps scrolling until "
                        "show_on_display or clear_display is called."}
    except Exception as e:
        return {"error": "%s: %s" % (type(e).__name__, e),
                "hint": "The OLED may not be wired up on this host."}


def clear_display(**_ignored):
    """Pause the live display: blank the OLED and turn off the LED bar. Call this when the
    user asks to turn off, stop, clear, or blank the display."""
    try:
        _stop_scroll()              # stop BEFORE clearing, else the marquee repaints the blank
        _DISPLAY["paused"] = True   # set first so the background loop's next tick skips a render
        _get_oled().clear()
        from led_strip import all_off
        all_off(_get_spi())
        _DISPLAY["level"] = 0.0
        _DISPLAY.pop("rgb_override", None)
        _DISPLAY.pop("level_override", None)
        return {"display": "cleared"}
    except Exception as e:
        return {"error": "%s: %s" % (type(e).__name__, e)}


def update_live_display(facts, samples=None, sr=44100):
    """Called continuously by the background producer loop (service.py) with the latest
    analyzed facts, and the raw audio behind them. No-op while paused. Never raises - a
    missing/broken display must not take down the producer thread that also feeds /scope.

    With `samples` the OLED shows a scope trace of the real waveform; the header line still
    carries the analyzed numbers. Without them it falls back to the band bars, which is what
    the standalone demo path still uses.
    """
    if _DISPLAY.get("paused", False):
        return
    try:
        from oled_display import show_bars, show_wave
        text = "%.0f Hz  %.0f dB" % (facts.get("dominant_hz", 0), facts.get("rms_db", 0))
        # -60..0 dB -> 0..1, the same scale the LED bar uses, so the OLED trace height and the
        # lit-pixel count always agree about how loud it is
        level = min(1.0, max(0.0, (facts.get("rms_db", -60) + 60) / 60))
        if samples is not None:
            show_wave(_get_oled(), samples, sr=sr, text=text,
                      dominant_hz=facts.get("dominant_hz", 0), level=level)
        else:
            bands = list(facts.get("band_energy", {}).values())
            peak = max(bands) if bands else 0.0
            bars = [(v / peak if peak else 0.0) for v in bands]
            show_bars(_get_oled(), bars, text)

        from led_strip import led_rgb_vu
        _DISPLAY["level"] = level
        # Smooth in colour space: a single cymbal crash stealing the band mix would otherwise
        # snap the whole bar to a different colour for one frame and back again.
        target = band_rgb(facts.get("band_energy", {}))
        previous = _DISPLAY.get("rgb")
        # keep tracking the sound even while overridden, so releasing the override snaps
        # straight back to the right colour instead of easing in from a stale one
        _DISPLAY["rgb"] = target if previous is None else tuple(
            int(p + RGB_SMOOTH * (t - p)) for p, t in zip(previous, target))

        rgb = _DISPLAY.get("rgb_override") or _DISPLAY["rgb"]
        held = _DISPLAY.get("level_override")
        led_rgb_vu(_get_spi(), level if held is None else held, rgb,
                   brightness=_DISPLAY["brightness"])
        _DISPLAY["last_error"] = None
    except Exception as e:
        # Recorded, not just printed: this is the only place the live display can fail, and a
        # log line nobody greps is how a dead panel or strip goes unnoticed. /healthz surfaces it.
        _DISPLAY["last_error"] = "%s: %s" % (type(e).__name__, e)
        print("[update_live_display] %s" % _DISPLAY["last_error"], file=sys.stderr, flush=True)


def get_display_status(**_ignored):
    """What the display is doing right now: colour being sent to the LED bar, how much of the
    bar is lit, whether the live feed is paused or scrolling text, and any render error. Call
    this when the user asks what the display/OLED/LED bar is doing or showing, what colour it
    is, or why it seems stuck."""
    status = dict(display_status())
    rgb = status.get("led_rgb")
    if rgb:
        peak = max(rgb) or 1
        named = "+".join(n for n, v in zip(("red", "green", "blue"), rgb) if v > 0.45 * peak)
        status["led_colour"] = named or "off"
    status["lit_pixels"] = "%d of 8" % int(status.get("led_level", 0.0) * 8)
    if status.get("paused"):
        status["why_frozen"] = ("The live feed is PAUSED, so the display is holding its last "
                                "frame - text or a marquee is up. Call show_on_display to resume.")
    return status


def display_status():
    """What the service believes about the display. Note the honest limits: `led_open` means the
    SPI handle exists, NOT that pixels light - SPI has no ACK, so a disconnected strip is
    indistinguishable from a working one. Only run_display_selftest and human eyes settle that."""
    return {"oled_open": "oled" in _DISPLAY,
            "led_open": "spi" in _DISPLAY,
            "paused": _DISPLAY.get("paused", False),
            "scrolling": _SCROLL.get("thread") is not None,
            "led_brightness_31": _DISPLAY.get("brightness"),
            "led_level": round(_DISPLAY.get("level", 0.0), 3),
            # hue is the one thing about the strip that no log line can show you, and the strip
            # cannot report back over SPI - so surface it here to debug colour without eyes
            "led_rgb": _DISPLAY.get("rgb"),
            "manual_colour": _DISPLAY.get("rgb_override"),
            "manual_level": _DISPLAY.get("level_override"),
            "last_render_error": _DISPLAY.get("last_error")}


COLOURS = {
    "red": (255, 0, 0), "orange": (255, 90, 0), "amber": (255, 160, 0),
    "yellow": (255, 255, 0), "green": (0, 255, 0), "lime": (140, 255, 0),
    "cyan": (0, 255, 255), "teal": (0, 200, 160), "blue": (0, 0, 255),
    "purple": (140, 0, 255), "violet": (140, 0, 255), "magenta": (255, 0, 200),
    "pink": (255, 80, 160), "white": (255, 255, 255), "off": (0, 0, 0),
}


def _parse_colour(colour):
    """Accept a name, #RRGGBB, or 'r,g,b'. Returns an (r, g, b) tuple or None."""
    if colour is None:
        return None
    if isinstance(colour, (list, tuple)) and len(colour) == 3:
        return tuple(max(0, min(255, int(c))) for c in colour)
    text = str(colour).strip().lower()
    if text in COLOURS:
        return COLOURS[text]
    if text.startswith("#") and len(text) == 7:
        return tuple(int(text[i:i + 2], 16) for i in (1, 3, 5))
    if "," in text:
        parts = [p.strip() for p in text.split(",")]
        if len(parts) == 3 and all(p.lstrip("-").isdigit() for p in parts):
            return tuple(max(0, min(255, int(p))) for p in parts)
    return None


def set_led_bar(colour=None, level=None, brightness=None, **_ignored):
    """Take manual control of the LED bar: hold it at a colour, a fill level, or both.

    Colour and level are overridden INDEPENDENTLY, and neither pauses the OLED. Setting only a
    colour leaves the bar still bouncing with loudness - it just bounces in your colour. That is
    almost always what "make the LEDs blue" means; freezing the whole display would be a
    surprising side effect of a request that only mentioned colour.
    """
    try:
        applied = {}
        if colour is not None:
            rgb = _parse_colour(colour)
            if rgb is None:
                return {"error": "unknown colour: %s" % colour,
                        "hint": "Use a name (%s), #RRGGBB, or 'r,g,b'." %
                                ", ".join(sorted(COLOURS)[:8])}
            _DISPLAY["rgb_override"] = rgb
            applied["colour"] = colour
            applied["rgb"] = rgb
        if level is not None:
            fraction = max(0.0, min(1.0, float(level) / 100.0))
            _DISPLAY["level_override"] = fraction
            applied["level_percent"] = round(fraction * 100)
        if brightness is not None:
            _DISPLAY["brightness"] = round(max(0.0, min(100.0, float(brightness))) / 100 * 31)
            applied["brightness_percent"] = float(brightness)
        if not applied:
            return {"error": "nothing to set",
                    "hint": "Pass a colour (e.g. 'blue'), a level 0-100, or a brightness 0-100."}

        from led_strip import led_rgb_vu
        rgb = _DISPLAY.get("rgb_override") or _DISPLAY.get("rgb") or (255, 0, 0)
        lit = _DISPLAY.get("level_override")
        led_rgb_vu(_get_spi(), _DISPLAY["level"] if lit is None else lit, rgb,
                   brightness=_DISPLAY["brightness"])
        applied["note"] = ("LED bar set. %s Call show_on_display to hand it back to the sound." %
                           ("Colour is held; the bar still moves with loudness."
                            if "level_percent" not in applied else "Colour and level are held."))
        return applied
    except Exception as e:
        return {"error": "%s: %s" % (type(e).__name__, e),
                "hint": "The LED bar may not be wired up on this host."}


def _colour_response_test(spi, cycles=3, hold=0.9):
    """Alternate bass-heavy and treble-heavy spectra through the REAL colour path.

    Tests band_rgb + led_rgb_vu end to end without needing music: with the AGC baseline sitting
    between the two extremes, bass frames must come out red and treble frames blue. If both come
    out the same colour, the mapping is broken - which is exactly the bug that shipped twice.

    The AGC baselines are saved and restored, so running a test does not leave the live display
    colour-shifted for the next ~7 seconds.
    """
    from led_strip import led_rgb_vu
    saved = list(_BAND_AVG)
    seen = []
    try:
        bassy = {"sub_20_60": 0.0, "bass_60_250": 0.90, "low_mid_250_500": 0.04,
                 "mid_500_2k": 0.03, "high_mid_2k_6k": 0.03, "high_6k_20k": 0.0}
        trebly = {"sub_20_60": 0.0, "bass_60_250": 0.10, "low_mid_250_500": 0.10,
                  "mid_500_2k": 0.25, "high_mid_2k_6k": 0.50, "high_6k_20k": 0.05}
        for _ in range(cycles):
            for label, spectrum in (("bass", bassy), ("treble", trebly)):
                rgb = band_rgb(spectrum)
                led_rgb_vu(spi, 1.0, rgb, brightness=8)
                seen.append((label, rgb))
                time.sleep(hold)
    finally:
        _BAND_AVG[:] = saved
    return seen


def run_display_selftest(**_ignored):
    """Run a visual self-test of the OLED and the LED bar, then return to the live display.
    Takes about 10 seconds. Call this when the user asks to test/check the display, OLED or
    LED bar, or asks whether the lights still work."""
    try:
        _stop_scroll()
        _DISPLAY["paused"] = True     # hold the producer off the panel for the duration
        from oled_display import show_text
        from led_strip import selftest, led_rgb_vu
        show_text(_get_oled(), "DISPLAY SELF TEST")
        stages = selftest(_get_spi(), hold=0.5)
        colours = _colour_response_test(_get_spi())
        _DISPLAY["paused"] = False    # hand the panel back to the live scope trace

        bass = [rgb for kind, rgb in colours if kind == "bass"][-1]
        treble = [rgb for kind, rgb in colours if kind == "treble"][-1]
        return {"self_test": "complete",
                "expected": ["OLED: the words DISPLAY SELF TEST"] +
                            ["LED bar: %s" % s for s in stages] +
                            ["LED bar: alternating red-ish and blue-ish, 3 times - this is the "
                             "sound-to-colour mapping being exercised directly"] +
                            ["OLED: back to the live waveform"],
                "colour_check": {"bass_spectrum_rgb": bass, "treble_spectrum_rgb": treble,
                                 "distinct": bass != treble},
                "note": "The test has finished running. SPI and I2C cannot report whether the "
                        "pixels actually lit, so ASK THE USER whether they saw each of these - "
                        "especially whether the last stage alternated between two DIFFERENT "
                        "colours, which is what proves colour tracks the sound."}
    except Exception as e:
        _DISPLAY["paused"] = False
        return {"error": "%s: %s" % (type(e).__name__, e),
                "hint": "The OLED or LED bar may not be wired up on this host."}


def set_display_brightness(percent=50):
    """Set the LED bar's brightness from 0 (off) to 100 (max) and re-light it immediately
    at the current level. Call this when the user asks to dim, brighten, or set the LED
    brightness/lighting."""
    try:
        percent = max(0.0, min(100.0, float(percent if percent is not None else 50)))
        _DISPLAY["brightness"] = round(percent / 100 * 31)
        from led_strip import led_rgb_vu
        led_rgb_vu(_get_spi(), _DISPLAY["level"], _DISPLAY.get("rgb", (255, 0, 0)),
                   brightness=_DISPLAY["brightness"])
        return {"led_brightness_percent": percent}
    except Exception as e:
        return {"error": "%s: %s" % (type(e).__name__, e),
                "hint": "The LED bar may not be wired up on this host."}


def list_input_devices():
    """List available audio input devices and their indices."""
    try:
        import sounddevice as sd
        devs = [{"index": i, "name": d["name"], "channels": d["max_input_channels"]}
                for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
        return {"input_devices": devs}
    except Exception as e:
        return {"error": "%s: %s" % (type(e).__name__, e)}


# ---- schemas + dispatch (Ollama / OpenAI function-calling format) ----------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "capture_and_analyze",
            "description": (
                "Record a few seconds of audio from the microphone (or read a WAV file) "
                "and return structured acoustic facts: loudness (rms_db), dominant "
                "frequency (dominant_hz), strongest tones (peaks_hz), spectral centroid, "
                "60 Hz mains-hum score (hum_60hz_score), per-band energy, and a heuristic "
                "label. Call this whenever the user asks what a sound is, how loud it is, "
                "whether there is hum, what frequency something is, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {"type": "number",
                                "description": "How many seconds to record (default 2)."},
                    "wav_path": {"type": "string",
                                 "description": "Optional path to a .wav file to analyze instead of recording."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_spectrum_image",
            "description": (
                "Record a few seconds of audio (or read a WAV file) and save a spectrogram "
                "PNG showing frequency content over time (0-8 kHz). Call this when the user "
                "wants to SEE the sound / its spectrum / spectrogram / frequencies. Returns "
                "the saved image path - relay that path to the user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {"type": "number",
                                "description": "How many seconds to record (default 2)."},
                    "wav_path": {"type": "string",
                                 "description": "Optional path to a .wav file to render instead of recording."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_input_devices",
            "description": "List available audio input devices and their indices.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_on_display",
            "description": (
                "Resume the LIVE OLED spectrum display and LED VU meter - once resumed they "
                "react continuously to the microphone on their own, with no need to call this "
                "again. Call this when the user asks to light up, turn on, resume, or show the "
                "display/OLED/LED bar."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_text_on_display",
            "description": (
                "Print a text message on the physical OLED screen. Call this whenever the "
                "user asks to write, print, display, or put SPECIFIC WORDS on the display/"
                "OLED/screen - for example 'show hello on the display' or 'put my name on "
                "the OLED'. This does NOT listen to the microphone. It is different from "
                "show_on_display, which shows the live sound spectrum rather than words. "
                "The message stays up until show_on_display or clear_display is called."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string",
                             "description": "The message to print. Fits 21 characters per "
                                            "line, 5 lines."},
                    "scroll": {"type": "boolean",
                               "description": "Scroll the message across the screen as a "
                                              "marquee instead of printing it statically. Set "
                                              "true when the user asks for scrolling/ticker/"
                                              "marquee text. Left unset, long messages that "
                                              "would not fit scroll automatically."},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clear_display",
            "description": (
                "Pause the live display: blank the OLED screen and turn off the LED bar, and "
                "STOP them from updating until show_on_display is called again. Call this "
                "whenever the user asks to turn OFF, stop, disable, clear, or blank the "
                "display/OLED/LED bar."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_led_bar",
            "description": (
                "Manually control the physical LED bar: set its COLOUR, how much of it is LIT, "
                "or its brightness. Call this when the user asks to make the LEDs/bar/lights a "
                "colour ('make the leds blue', 'turn the bar red'), to fill it to a level, or to "
                "hold it steady. Setting only a colour keeps the bar reacting to loudness in "
                "that colour. Call show_on_display to hand it back to the sound."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "colour": {"type": "string",
                               "description": "Colour name (red, orange, amber, yellow, green, "
                                              "lime, cyan, teal, blue, purple, magenta, pink, "
                                              "white, off), #RRGGBB, or 'r,g,b'."},
                    "level": {"type": "number",
                              "description": "How much of the bar to light, 0-100 percent. Omit "
                                             "to keep it reacting to loudness."},
                    "brightness": {"type": "number",
                                   "description": "LED brightness 0-100. Omit to leave unchanged."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_display_status",
            "description": (
                "Report what the physical display is doing right now: the colour currently sent "
                "to the LED bar, how many pixels are lit, whether the live feed is paused or "
                "scrolling text, and any render error. Call this when the user asks what the "
                "display / OLED / LED bar is doing or showing, what colour it is, whether it is "
                "working, or why it looks stuck or frozen."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_display_selftest",
            "description": (
                "Run a visual self-test of the physical OLED and LED bar: solid colours, a "
                "brightness ramp, a single-pixel chase and a VU sweep, then back to the live "
                "display. Takes about 10 seconds. Call this when the user asks to test or check "
                "the display / OLED / LED bar / lights, or asks whether they still work. "
                "Afterwards, tell the user what should have appeared and ask if they saw it."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_display_brightness",
            "description": (
                "Set the LED bar's brightness from 0 (off) to 100 (max) and re-light it "
                "immediately at the current level. Call this when the user asks to dim, "
                "brighten, or set the LED brightness/lighting."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "percent": {"type": "number",
                                "description": "0 (off) to 100 (max brightness). Default 50."},
                },
            },
        },
    },
]

DISPATCH = {
    "capture_and_analyze": capture_and_analyze,
    "get_spectrum_image": get_spectrum_image,
    "list_input_devices": list_input_devices,
    "show_on_display": show_on_display,
    "show_text_on_display": show_text_on_display,
    "clear_display": clear_display,
    "get_display_status": get_display_status,
    "set_led_bar": set_led_bar,
    "run_display_selftest": run_display_selftest,
    "set_display_brightness": set_display_brightness,
}


def call(name, arguments):
    """Dispatch a tool call by name with a dict (or JSON string) of arguments."""
    if isinstance(arguments, str):
        arguments = json.loads(arguments or "{}")
    fn = DISPATCH.get(name)
    if fn is None:
        return {"error": "unknown tool: %s" % name}
    return fn(**(arguments or {}))


if __name__ == "__main__":  # quick smoke test without the LLM
    print(json.dumps(capture_and_analyze(seconds=2.0), indent=2))
