"""
Display build - SK9822 / APA102 LED bar driver (SPI, via the 74AHCT125
level shifter). See docs/display/DISPLAY-BUILD.md.

  python led_strip.py --off          # blank the strip (do this first, always -
                                      #   SK9822 powers up in an undefined state,
                                      #   which is what tripped the Jetson's 5V
                                      #   protection during bring-up)
  python led_strip.py --test         # light pixel 0 white, then blank

led_vu(level) is the function the capture loop calls once features.analyze()
is wired in - see DISPLAY-BUILD.md "Software hooks".
"""

import argparse
import colorsys
import math
import time

import spidev

NUM_LEDS = 8
SPI_BUS, SPI_DEVICE = 0, 0
SPI_HZ = 1_000_000   # conservative - keeps ringing on the DI line low


def open_spi():
    spi = spidev.SpiDev()
    spi.open(SPI_BUS, SPI_DEVICE)
    spi.max_speed_hz = SPI_HZ
    spi.mode = 0
    return spi


def _led_bytes(r, g, b, brightness=31):
    brightness = max(0, min(31, brightness))
    return [0xE0 | brightness, b, g, r]


def _end_frame(num_leds):
    # SK9822 wants a zero end frame (not 0xFF like some APA102 code uses),
    # with enough extra clocks to shift the last pixel's data through.
    return [0x00] * max(4, (num_leds + 15) // 16)


def write_frame(spi, pixels):
    """pixels: list of (r, g, b[, brightness]) tuples, brightness 0-31."""
    data = [0x00, 0x00, 0x00, 0x00]  # start frame
    for r, g, b, *rest in pixels:
        brightness = rest[0] if rest else 31
        data += _led_bytes(r, g, b, brightness)
    data += _end_frame(len(pixels))
    spi.writebytes(data)


def all_off(spi, num_leds=NUM_LEDS):
    write_frame(spi, [(0, 0, 0, 0)] * num_leds)


def led_vu(spi, level, num_leds=NUM_LEDS, brightness=8):
    """level: 0.0 (silent) to 1.0 (loudest). Lights a bar green->amber->red.
    brightness: 0-31 (APA102 5-bit) for the LIT pixels; unlit pixels always stay off."""
    lit = int(max(0.0, min(1.0, level)) * num_leds)
    pixels = []
    for i in range(num_leds):
        if i >= lit:
            pixels.append((0, 0, 0, 0))
            continue
        frac = i / max(1, num_leds - 1)   # 0 at the start, 1 at the top
        r = int(255 * min(1.0, frac * 2))
        g = int(255 * min(1.0, (1.0 - frac) * 2))
        pixels.append((r, g, 0, brightness))
    write_frame(spi, pixels)


def led_rgb_vu(spi, level, rgb, num_leds=NUM_LEDS, brightness=8):
    """Two independent axes on one strip: how MANY pixels light tracks loudness, and their
    COLOUR carries the spectrum's shape. Loudness and tone stay readable at a glance.

    Takes a colour rather than deriving one, so the meaning of "what colour is this sound"
    lives with the DSP (acoustic_tools) and this module stays a dumb pixel driver.
    """
    lit = int(max(0.0, min(1.0, level)) * num_leds)
    r, g, b = (max(0, min(255, int(c))) for c in rgb)
    write_frame(spi, [(r, g, b, brightness) if i < lit else (0, 0, 0, 0)
                      for i in range(num_leds)])


def selftest(spi, num_leds=NUM_LEDS, hold=0.6, chase=True):
    """Staged visual test of the whole strip.

    Worth having as a first-class function rather than a one-off script: SPI has no ACK, so
    write_frame() returns happily into a disconnected strip, dead level shifter or unpowered
    rail. Nothing in the software can tell you the bar stopped working - only eyes can, and
    only if there is something to look at.

    Each stage isolates one failure mode: solid colours catch RGB byte-order mistakes, the
    brightness ramp catches a stuck 5-bit field, the chase catches NUM_LEDS being wrong or a
    broken joint mid-strip, and the VU sweep exercises the function the live loop actually calls.
    """
    all_off(spi, num_leds)
    time.sleep(0.2)
    for colour in ((255, 0, 0), (0, 255, 0), (0, 0, 255)):
        write_frame(spi, [colour + (6,)] * num_leds)
        time.sleep(hold)
    for b in (1, 8, 31):
        write_frame(spi, [(255, 255, 255, b)] * num_leds)
        time.sleep(hold / 2)
    if chase:
        for i in range(num_leds):
            pixels = [(0, 0, 0, 0)] * num_leds
            pixels[i] = (255, 255, 255, 8)
            write_frame(spi, pixels)
            time.sleep(hold / 3)
    for step in range(11):
        led_vu(spi, step / 10.0, num_leds, brightness=8)
        time.sleep(hold / 6)
    all_off(spi, num_leds)
    return ["solid red, green, blue across all %d pixels" % num_leds,
            "white brightness ramp, dim to full",
            "single white pixel walking 1 to %d" % num_leds,
            "green-to-red VU bar filling from empty to full"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", action="store_true", help="blank the strip")
    ap.add_argument("--test", action="store_true", help="light pixel 0 white, then blank")
    ap.add_argument("--selftest", action="store_true", help="full staged visual test")
    ap.add_argument("--num-leds", type=int, default=NUM_LEDS)
    args = ap.parse_args()

    spi = open_spi()
    all_off(spi, args.num_leds)   # always start blank

    if args.selftest:
        print("Running LED self-test on %d pixels. You should see:" % args.num_leds)
        for line in selftest(spi, args.num_leds):
            print("  - %s" % line)
        print("Blanked.")
    elif args.test:
        print(f"Lighting pixel 0 on a {args.num_leds}-LED strip for 2s...")
        pixels = [(0, 0, 0, 0)] * args.num_leds
        pixels[0] = (255, 255, 255, 8)
        write_frame(spi, pixels)
        time.sleep(2)
        all_off(spi, args.num_leds)
        print("Blanked.")
    elif not args.off:
        print("Blanked. Pass --test to light pixel 0, or --off to just blank (done).")

    spi.close()
