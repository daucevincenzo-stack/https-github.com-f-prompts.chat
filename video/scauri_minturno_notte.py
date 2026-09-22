"""Genera un video notturno stilizzato del golfo visto da Scauri (Minturno, LT).

Scena procedurale: cielo stellato, luna sul mare, promontorio di Gaeta e costa
di Formia illuminati all'orizzonte, Monte di Scauri, lungomare con lampioni e palme.

Uso:  python3 scauri_minturno_notte.py [--seconds 15] [--fps 30] [--out scauri_minturno_notte.mp4]
Dipendenze: numpy, pillow, imageio, imageio-ffmpeg
"""
import argparse
import math

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1280, 720
HORIZON = 410
rng = np.random.default_rng(7)
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)


def ridge(xs, base, peaks):
    """Profilo di una collina: base + somma di gaussiane + rumore leggero."""
    y = np.full_like(xs, base, dtype=np.float32)
    for cx, h, w in peaks:
        y -= h * np.exp(-((xs - cx) / w) ** 2)
    y += 1.5 * np.sin(xs * 0.07) + 1.0 * np.sin(xs * 0.19 + 1.3)
    return y


def build_static():
    xs = np.arange(W, dtype=np.float32)
    t = np.clip(YY / HORIZON, 0, 1)[..., None]
    top = np.array([4, 6, 20], np.float32)
    mid = np.array([14, 22, 52], np.float32)
    low = np.array([48, 44, 78], np.float32)  # alone caldo delle luci costiere
    sky = np.where(t < 0.6, top + (mid - top) * (t / 0.6), mid + (low - mid) * ((t - 0.6) / 0.4))
    sky = np.broadcast_to(sky, (H, W, 3)).copy()

    # Via Lattea tenue in diagonale
    band = np.exp(-(((YY - (0.35 * XX - 60)) / 90) ** 2))[..., None]
    sky += band * np.array([10, 10, 16], np.float32) * (YY[..., None] < HORIZON)

    # Promontorio di Gaeta (sinistra) e Monte di Scauri (destra), costa bassa di Formia in mezzo
    gaeta = ridge(xs, HORIZON, [(150, 55, 110), (260, 30, 70), (60, 25, 60)])
    formia = ridge(xs, HORIZON, [(470, 14, 160), (620, 9, 120)])
    aurunci = ridge(xs, HORIZON, [(560, 70, 220), (780, 45, 160)])
    scauri = ridge(xs, HORIZON, [(1120, 95, 150), (1250, 60, 90), (980, 25, 80)])
    layers = [
        (aurunci, np.array([22, 24, 44], np.float32)),
        (gaeta, np.array([12, 13, 28], np.float32)),
        (formia, np.array([14, 15, 30], np.float32)),
        (scauri, np.array([9, 10, 22], np.float32)),
    ]
    land = np.zeros((H, W), bool)
    for prof, col in layers:
        m = (YY >= prof[None, :]) & (YY < HORIZON)
        sky[m] = col
        land |= m

    # Luci costiere (Gaeta, Formia, Scauri)
    lights = []
    for cx, spread, n, ymin in [(170, 120, 90, 0.2), (480, 170, 110, 0.0), (1110, 130, 70, 0.15)]:
        for _ in range(n):
            x = int(np.clip(rng.normal(cx, spread / 2), 2, W - 3))
            top_y = min(gaeta[x], formia[x], aurunci[x], scauri[x])
            depth = HORIZON - top_y
            y = int(HORIZON - 2 - rng.uniform(0, max(depth * (0.35 + ymin), 3)))
            warm = rng.random() < 0.8
            col = (255, 200, 120) if warm else (220, 235, 255)
            lights.append((x, y, col, rng.uniform(0.5, 1.0), rng.uniform(0, 6.28)))

    stars = []
    while len(stars) < 420:
        x, y = rng.integers(0, W), rng.integers(0, HORIZON - 40)
        if not land[y, x]:
            stars.append((x, y, rng.uniform(0.2, 1.0), rng.uniform(0.5, 3.0), rng.uniform(0, 6.28)))
    return sky, lights, stars


def build_foreground():
    """Lungomare di Scauri: ringhiera, lampioni e palme in silhouette (RGBA)."""
    fg = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(fg)
    d.rectangle([0, H - 70, W, H], fill=(6, 6, 12, 255))  # passeggiata
    d.rectangle([0, H - 96, W, H - 92], fill=(8, 8, 16, 255))  # corrimano
    for x in range(0, W, 26):
        d.rectangle([x, H - 96, x + 3, H - 70], fill=(8, 8, 16, 255))
    lamps = [110, 430, 750, 1070]
    for lx in lamps:
        d.rectangle([lx - 3, H - 250, lx + 3, H - 70], fill=(10, 10, 18, 255))
        d.line([lx, H - 250, lx + 28, H - 262], fill=(10, 10, 18, 255), width=5)
        d.rectangle([lx + 22, H - 262, lx + 40, H - 254], fill=(10, 10, 18, 255))
    for px, lean in [(270, -0.12), (920, 0.1), (1210, -0.05)]:
        top = (px + int(lean * 260), H - 330)
        d.line([px, H - 70, top[0], top[1]], fill=(7, 7, 14, 255), width=14)
        # fronde ricadenti: archi che salgono poco e poi scendono
        for a in (-170, -150, -125, -100, -80, -55, -30, -10, 200, 160, 20):
            r = math.radians(a)
            pts = []
            for k in range(13):
                s = k / 12
                pts.append((top[0] + 120 * s * math.cos(r),
                            top[1] + 60 * s * math.sin(r) + 110 * s * s))
            for k in range(12):
                wdt = max(2, int(9 * (1 - k / 12)))
                d.line([pts[k], pts[k + 1]], fill=(7, 7, 14, 255), width=wdt)
            for k in range(3, 12):  # foglioline
                x0, y0 = pts[k]
                d.line([(x0, y0), (x0 + 6, y0 + 16)], fill=(7, 7, 14, 255), width=2)
                d.line([(x0, y0), (x0 - 6, y0 + 16)], fill=(7, 7, 14, 255), width=2)
    lamp_pos = [(lx + 31, H - 252) for lx in lamps]
    return np.asarray(fg).astype(np.float32) / 255.0, lamp_pos


def glow(frame, x, y, color, radius, strength):
    x0, x1 = max(int(x - radius * 3), 0), min(int(x + radius * 3), W)
    y0, y1 = max(int(y - radius * 3), 0), min(int(y + radius * 3), H)
    if x0 >= x1 or y0 >= y1:
        return
    d2 = (XX[y0:y1, x0:x1] - x) ** 2 + (YY[y0:y1, x0:x1] - y) ** 2
    g = np.exp(-d2 / (2 * radius**2))[..., None] * strength
    frame[y0:y1, x0:x1] += g * np.array(color, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default="scauri_minturno_notte.mp4")
    args = ap.parse_args()

    sky, lights, stars = build_static()
    fg, lamp_pos = build_foreground()
    moon_x, moon_y = 780.0, 120.0

    sea_rows = np.arange(HORIZON, H, dtype=np.float32)
    depth = (sea_rows - HORIZON) / (H - HORIZON)  # 0 all'orizzonte, 1 in primo piano
    sea_base = (np.array([6, 10, 26], np.float32)[None, :]
                + depth[:, None] * np.array([4, 8, 16], np.float32))
    sx, sy = XX[HORIZON:], YY[HORIZON:]

    try:
        font = ImageFont.truetype("DejaVuSerif.ttf", 54)
        small = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        font = small = ImageFont.load_default()

    n = int(args.seconds * args.fps)
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8,
                                pixelformat="yuv420p", macro_block_size=8)
    for i in range(n):
        t = i / args.fps
        f = sky.copy()

        for x, y, b, sp, ph in stars:
            v = b * (0.65 + 0.35 * math.sin(t * sp + ph))
            f[y, x] += 230 * v
            if b > 0.85:
                glow(f, x, y, (180, 190, 255), 1.6, 0.5 * v)

        mx = moon_x - t * 1.2  # la luna scorre piano
        glow(f, mx, moon_y, (120, 120, 150), 70, 0.8)
        d2 = (XX[:HORIZON] - mx) ** 2 + (YY[:HORIZON] - moon_y) ** 2
        disk = np.clip(24 - np.sqrt(d2), 0, 1)[..., None]
        crater = 1.0
        for cx, cy, cr in ((-7, -5, 7), (6, 4, 5), (-2, 9, 4), (9, -8, 3)):
            crater = crater - 0.09 * np.exp(-((XX[:HORIZON] - mx - cx) ** 2
                                             + (YY[:HORIZON] - moon_y - cy) ** 2) / cr**2)[..., None]
        f[:HORIZON] = f[:HORIZON] * (1 - disk) + disk * np.array([245, 240, 222], np.float32) * crater
        # ri-disegna le colline davanti al bagliore lunare
        # (la luna è alta, quindi basta il cielo sopra le creste)

        # Mare: onde + riflesso lunare frammentato
        waves = (np.sin(sx * 0.045 + sy * 0.9 - t * 1.6)
                 + 0.6 * np.sin(sx * 0.11 - sy * 0.35 + t * 2.3)
                 + 0.4 * np.sin(sx * 0.021 + sy * 1.7 + t * 0.9))
        sea = sea_base[:, None, :] + (waves[..., None] * 2.2) * (0.4 + depth[:, None, None])
        spread = 10 + depth[:, None] * 170
        beam = np.exp(-((sx - mx) / spread) ** 2)
        sparkle = np.clip((waves - 0.9) * 1.4, 0, 1)
        sea += (beam * sparkle)[..., None] * np.array([235, 228, 200], np.float32)
        sea += (beam * 0.18)[..., None] * np.array([60, 60, 80], np.float32)

        # Riflessi verticali delle luci della costa
        for x, y, col, b, ph in lights[::3]:
            streak = np.exp(-((sx - x) / 2.5) ** 2) * np.exp(-(sy - HORIZON) / 45.0)
            fl = np.clip((np.sin(sy * 0.8 + t * 3 + ph) + 0.2), 0, 1)
            sea += (streak * fl * b * 0.35)[..., None] * np.array(col, np.float32)
        f[HORIZON:] = sea

        for x, y, col, b, ph in lights:
            v = b * (0.8 + 0.2 * math.sin(t * 4 + ph))
            f[y, x] = np.maximum(f[y, x], np.array(col, np.float32) * v)
            glow(f, x, y, col, 2.2, 0.25 * v)

        # Faro sul promontorio di Gaeta
        beacon = max(0.0, math.sin(t * 2.2)) ** 8
        glow(f, 88, HORIZON - 40, (255, 240, 200), 6, 1.2 * beacon)

        # Primo piano
        f = f * (1 - fg[..., 3:4]) + fg[..., :3] * 255 * fg[..., 3:4]
        for lx, ly in lamp_pos:
            glow(f, lx, ly + 6, (255, 190, 110), 40, 0.55)
            glow(f, lx, ly + 4, (255, 225, 170), 8, 1.4)
            glow(f, lx, H - 60, (255, 170, 90), 70, 0.10)

        img = Image.fromarray(np.clip(f, 0, 255).astype(np.uint8))

        # Titolo in dissolvenza
        a = min(1.0, max(0.0, (t - 1.5) / 2.0)) * min(1.0, max(0.0, (args.seconds - 1 - t) / 1.5))
        if a > 0:
            layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            d = ImageDraw.Draw(layer)
            title, sub = "Scauri · Minturno", "Golfo di Gaeta — di notte"
            tw = d.textlength(title, font=font)
            sw = d.textlength(sub, font=small)
            d.text(((W - tw) / 2, 250), title, font=font, fill=(250, 240, 220, int(235 * a)))
            d.text(((W - sw) / 2, 318), sub, font=small, fill=(210, 210, 230, int(200 * a)))
            shadow = layer.filter(ImageFilter.GaussianBlur(6))
            img = img.convert("RGBA")
            img.alpha_composite(shadow)
            img.alpha_composite(layer)
            img = img.convert("RGB")

        # Dissolvenza in apertura/chiusura
        k = min(1.0, t / 1.0, (args.seconds - t) / 1.0)
        arr = (np.asarray(img).astype(np.float32) * max(k, 0)).astype(np.uint8)
        writer.append_data(arr)
    writer.close()
    print(f"Scritto {args.out} ({n} frame, {W}x{H} @ {args.fps}fps)")


if __name__ == "__main__":
    main()
