"""
Stage 4 - expose the DSP to an LLM as callable TOOLS.

The language model never sees raw audio. It calls these functions, which
wrap features.analyze(), and reasons over the JSON facts they return.

These same functions plug into a standalone agent (agent.py) now, or into
Open WebUI later - the tool contract is identical.
"""

import json
import sys
import threading

import numpy as np

from features import analyze, SAMPLE_RATE


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


def _marquee(device, strip, stop, step, delay):
    """Slide a 128px window across `strip` until stopped. Never raises - a display fault must
    not kill the thread silently mid-frame without a log line."""
    from oled_display import WIDTH, HEIGHT
    try:
        span = max(1, strip.width - WIDTH)
        while not stop.is_set():
            for off in range(0, span, step):
                if stop.is_set():
                    return
                device.display(strip.crop((off, 0, off + WIDTH, HEIGHT)))
                if stop.wait(delay):    # doubles as the frame delay AND the cancel check
                    return
    except Exception as e:
        print("[_marquee] %s: %s" % (type(e).__name__, e), file=sys.stderr, flush=True)


def show_on_display(**_ignored):
    """Resume the live OLED spectrum + LED VU meter - they react continuously to the mic
    (driven by the same background loop that feeds /scope). Call this when the user asks
    to light up, turn on, resume, or show the display."""
    _stop_scroll()          # stop BEFORE unpausing, or scroll and producer both write at once
    _DISPLAY["paused"] = False
    try:
        _get_oled()   # re-send DISPLAYON: another process may have left the panel dark
    except Exception as e:
        return {"display": "live", "warning": "%s: %s" % (type(e).__name__, e),
                "note": "Resumed, but the OLED could not be opened."}
    return {"display": "live", "note": "Updates automatically from the mic every ~1s."}


SCROLL_STEP = 2      # pixels per frame
SCROLL_DELAY = 0.04  # ~25 fps; a full frame costs ~29 ms over this I2C bus, so this is the
                     # practical ceiling. Together: ~50 px/s, readable without being sluggish.


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
        from oled_display import show_text, text_strip
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
        thread = threading.Thread(target=_marquee,
                                  args=(device, text_strip(text), stop,
                                        SCROLL_STEP, SCROLL_DELAY),
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
        if samples is not None:
            show_wave(_get_oled(), samples, sr=sr, text=text,
                      dominant_hz=facts.get("dominant_hz", 0))
        else:
            bands = list(facts.get("band_energy", {}).values())
            peak = max(bands) if bands else 0.0
            bars = [(v / peak if peak else 0.0) for v in bands]
            show_bars(_get_oled(), bars, text)

        from led_strip import led_vu
        level = min(1.0, max(0.0, (facts.get("rms_db", -60) + 60) / 60))  # -60..0 dB -> 0..1
        _DISPLAY["level"] = level
        led_vu(_get_spi(), level, brightness=_DISPLAY["brightness"])
    except Exception as e:
        print("[update_live_display] %s: %s" % (type(e).__name__, e), file=sys.stderr, flush=True)


def set_display_brightness(percent=50):
    """Set the LED bar's brightness from 0 (off) to 100 (max) and re-light it immediately
    at the current level. Call this when the user asks to dim, brighten, or set the LED
    brightness/lighting."""
    try:
        percent = max(0.0, min(100.0, float(percent if percent is not None else 50)))
        _DISPLAY["brightness"] = round(percent / 100 * 31)
        from led_strip import led_vu
        led_vu(_get_spi(), _DISPLAY["level"], brightness=_DISPLAY["brightness"])
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
