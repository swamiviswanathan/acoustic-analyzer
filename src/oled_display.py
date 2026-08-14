"""
Display build - SSD1306 OLED renderer (I2C).

Confirmed on the bench: bus 7, address 0x3C (see docs/display/DISPLAY-BUILD.md).

  python oled_display.py --hello   # smoke test: static message
  python oled_display.py --demo    # smoke test: animated fake spectrum bars

show_bars(bars, text) is the function the capture loop calls once
features.analyze() is wired in - see DISPLAY-BUILD.md "Software hooks".
"""

import argparse
import textwrap
import time

from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306

I2C_PORT = 7
I2C_ADDRESS = 0x3C

WIDTH, HEIGHT = 128, 64
BAR_AREA_TOP = 14          # leave room for the text line above the bars
NUM_BARS = 6               # one per features.BANDS entry

# The built-in PIL bitmap font is a fixed 6x11 cell, so the panel holds a hard
# 21 chars x 5 lines. Anything past that is dropped rather than drawn off-screen.
CHAR_W, LINE_H = 6, 12
MAX_CHARS = WIDTH // CHAR_W
MAX_LINES = HEIGHT // LINE_H


def get_device():
    serial = i2c(port=I2C_PORT, address=I2C_ADDRESS)
    device = ssd1306(serial, width=WIDTH, height=HEIGHT)
    # luma's atexit cleanup sends DISPLAYOFF when this process ends. The panel is shared with
    # the long-running tool service, whose cached handle would then paint into a dark screen
    # with no error - so bench runs must leave the panel powered.
    device.persist = True
    return device


def show_bars(device, bars, text=""):
    """bars: up to 6 floats in [0, 1], one per band, low to high. text: header line."""
    bar_h_max = HEIGHT - BAR_AREA_TOP
    bar_w = WIDTH // NUM_BARS
    gap = 2

    with canvas(device) as draw:
        draw.text((0, 0), text, fill="white")
        for i, v in enumerate(bars[:NUM_BARS]):
            h = int(max(0.0, min(1.0, v)) * bar_h_max)
            if h <= 0:
                continue   # nothing to draw for a silent band
            x0 = i * bar_w + gap // 2
            x1 = x0 + bar_w - gap
            y0 = HEIGHT - h
            y1 = HEIGHT - 1
            draw.rectangle((x0, y0, x1, y1), fill="white")


def show_text(device, text, draw=True):
    """Word-wrap `text` and paint it on the panel. Returns the lines actually drawn.

    draw=False measures without touching the device (and accepts device=None), so callers can
    ask "does this fit?" before choosing between a static render and a scrolling marquee.
    """
    lines = []
    for para in str(text).splitlines() or [""]:
        # wrap() eats blank lines entirely; keep them so the caller's layout survives
        lines.extend(textwrap.wrap(para, width=MAX_CHARS) or [""])
    lines = lines[:MAX_LINES]

    if draw:
        with canvas(device) as d:
            for i, line in enumerate(lines):
                d.text((0, i * LINE_H), line, fill="white")
    return lines


WAVE_MS = 25.0      # fallback window when there is no clear pitch to lock onto
WAVE_FLOOR = 0.02   # peak below this stays small instead of being normalized up into noise
WAVE_CYCLES = 2.5   # cycles of the dominant tone to fit across the 128 px panel
WAVE_MS_MIN, WAVE_MS_MAX = 2.0, 40.0


def _window_ms(dominant_hz):
    """Milliseconds to plot. Scaled so ~2.5 cycles of the dominant tone span the screen.

    A fixed window only looks like a wave for one pitch: 25 ms is a clean sine at 60 Hz but
    crushes a 1 kHz tone into 25 cycles of solid fill. Tracking the pitch keeps the trace
    recognisably a wave across the whole audio range.
    """
    if not dominant_hz or dominant_hz <= 0:
        return WAVE_MS
    return min(WAVE_MS_MAX, max(WAVE_MS_MIN, WAVE_CYCLES * 1000.0 / float(dominant_hz)))


def show_wave(device, samples, sr=44100, text="", dominant_hz=0):
    """Oscilloscope trace: plot the actual waveform below the header line.

    Two details make it readable rather than a jittering smear:

    * Trigger. The window starts at a rising zero-crossing, so a steady tone stands still
      instead of sliding sideways every frame - the same thing a scope's trigger does.
    * Min/max per column. Each of the 128 columns draws a vertical line spanning the range
      of the samples it covers, so a tone shows a clean curve while broadband noise fills
      in as a band. Plotting one sampled point per column would alias high frequencies into
      nonsense instead.
    """
    import numpy as np
    x = np.asarray(samples, dtype=np.float64).flatten()
    win = max(8, int(sr * _window_ms(dominant_hz) / 1000.0))
    if x.size < win // 2:
        return

    seg = x[-2 * win:]                     # twice the window, so the trigger can slide
    head = seg[:max(1, seg.size - win)]
    rising = np.nonzero((head[:-1] <= 0) & (head[1:] > 0))[0]
    start = int(rising[0]) if rising.size else 0
    seg = seg[start:start + win]
    if seg.size < 2:
        return

    seg = seg / max(float(np.max(np.abs(seg))), WAVE_FLOOR)
    top = BAR_AREA_TOP
    mid = top + (HEIGHT - top) // 2
    amp = (HEIGHT - top) // 2 - 1

    def y_of(v):
        return mid - int(np.clip(v, -1, 1) * amp)

    with canvas(device) as draw:
        draw.text((0, 0), text, fill="white")
        if seg.size >= WIDTH:
            # Dense: more samples than columns. Each column spans a range of samples, so draw
            # its min..max - that renders noise as a band and a tone as a solid curve.
            edges = np.linspace(0, seg.size, WIDTH + 1).astype(int)
            for col in range(WIDTH):
                chunk = seg[edges[col]:edges[col + 1]]
                if chunk.size:
                    draw.line((col, y_of(chunk.max()), col, y_of(chunk.min())), fill="white")
        else:
            # Sparse: a high-pitched window holds fewer samples than the panel is wide, so
            # min/max would leave most columns empty and the trace would break into dots.
            # Interpolate and join the points into a continuous line instead.
            ys = [y_of(v) for v in np.interp(np.linspace(0, seg.size - 1, WIDTH),
                                             np.arange(seg.size), seg)]
            draw.line([(c, y) for c, y in enumerate(ys)], fill="white")


def text_strip(text):
    """Render `text` as one long single-line image for a marquee.

    Padded with a blank screen-width at each end so the message scrolls fully off before
    it re-enters. The default PIL bitmap font is used deliberately: python:3.11-slim ships
    no TrueType fonts, so ImageFont.truetype() works on a dev Mac but raises in the container.
    """
    from PIL import Image, ImageDraw
    text = " ".join(str(text).split())          # a marquee is one line: flatten any newlines
    strip = Image.new("1", (CHAR_W * len(text) + 2 * WIDTH, HEIGHT))
    ImageDraw.Draw(strip).text((WIDTH, (HEIGHT - 11) // 2), text, fill="white")
    return strip


def hello(device):
    with canvas(device) as draw:
        draw.text((0, 0), "Acoustic Analyzer", fill="white")
        draw.text((0, 16), "OLED test OK", fill="white")
        draw.text((0, 32), f"I2C {I2C_PORT}@0x{I2C_ADDRESS:02X}", fill="white")


def demo(device):
    import math
    print("Animating fake spectrum bars... Ctrl-C to stop.")
    t = 0.0
    try:
        while True:
            bars = [0.5 + 0.5 * math.sin(t + i) for i in range(NUM_BARS)]
            show_bars(device, bars, text="demo mode")
            t += 0.3
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hello", action="store_true", help="static smoke test")
    ap.add_argument("--demo", action="store_true", help="animated fake bars")
    args = ap.parse_args()

    dev = get_device()
    if args.demo:
        demo(dev)
    else:
        hello(dev)
        print("Wrote hello screen. Ctrl-C to exit and clear.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            dev.clear()
