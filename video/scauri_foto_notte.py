"""Video notturno dalla foto reale della spiaggia di Scauri (Minturno) verso il Monte d'Oro.

Dalla foto scattata all'ora blu (scauri_foto.jpg) ricava una scena notturna:
orizzonte raddrizzato, cielo notturno con stelle e luna, riflesso lunare sulle onde,
luci del lungomare e della torre, mare che si muove (effetto cinemagraph), zoom lento e titolo.

Uso:  python3 scauri_foto_notte.py [--seconds 15] [--verticale] [--out scauri_foto_notte.mp4]
"""
import argparse
import math

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

W, H = 1280, 720  # --verticale imposta 1080x1920
CENTER_X = 1240  # centro dell'inquadratura verticale (torre a sinistra, albergo a destra), coordinate foto
MARGIN = 1.12  # margine per lo zoom lento
TILT = math.degrees(math.atan(0.096))  # l'orizzonte nella foto sale di ~5.5° verso destra


def horizon_y(x):
    return 868 - 0.096 * x  # coordinate della foto originale 2576x1932


def masks_original(img):
    """Maschere nel sistema della foto: cielo e mare, più il profilo della costa."""
    a = np.asarray(img).astype(np.float32)
    h, w, _ = a.shape
    lum = a.mean(2)
    lum_s = np.asarray(Image.fromarray(lum.astype(np.uint8)).filter(ImageFilter.BoxBlur(4))).astype(np.float32)
    xs = np.arange(w)
    hz = horizon_y(xs)
    land_top = hz.copy()
    for x in range(w):
        col = lum_s[: int(hz[x]), x]
        dark = np.nonzero(col < 120)[0]  # le colline sono molto più scure del cielo
        if dark.size:
            land_top[x] = dark[0]
    land_top = np.convolve(np.pad(land_top, 6, mode="edge"), np.ones(13) / 13, "valid")
    yy = np.arange(h)[:, None]
    sky = (yy < land_top[None, :] - 1).astype(np.float32)
    sea = (yy > hz[None, :] + 1).astype(np.float32)
    return sky, sea, land_top


def light_layers(size):
    """Luci del lungomare, della torre e del lido disegnate sulla foto originale."""
    w, h = size
    rng = np.random.default_rng(11)
    lights = Image.new("RGB", size, (0, 0, 0))
    d = ImageDraw.Draw(lights)
    # riva: da sotto il promontorio fino al bordo destro
    shore = [(800, 789), (1100, 755), (1400, 725), (1675, 696), (2100, 660), (2576, 619)]

    def shore_y(x):
        for (x0, y0), (x1, y1) in zip(shore, shore[1:]):
            if x0 <= x <= x1:
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return shore[-1][1]

    for _ in range(170):
        x = rng.uniform(820, 2570)
        if rng.random() < 0.35:
            x = rng.uniform(1670, 1850)  # l'albergo bianco
        y = shore_y(x) - rng.uniform(4, 60 if 1670 < x < 1850 else 30)
        warm = rng.random() < 0.82
        c = (255, 196, 120) if warm else (215, 230, 255)
        r = rng.uniform(1.6, 3.2)
        d.ellipse([x - r, y - r, x + r, y + r], fill=c)
    halo = lights.filter(ImageFilter.GaussianBlur(9))
    # lucina rossa di segnalazione visibile nella foto sotto il promontorio
    red = Image.new("L", size, 0)
    ImageDraw.Draw(red).ellipse([1050, 740, 1064, 754], fill=255)
    red = red.filter(ImageFilter.GaussianBlur(5))
    # torre sul Monte d'Oro: maschera morbida per illuminarla dal basso con luce calda
    wash = Image.new("L", size, 0)
    ImageDraw.Draw(wash).ellipse([955, 640, 1005, 700], fill=255)
    wash = wash.filter(ImageFilter.GaussianBlur(10))
    phase = (rng.random((h // 8, w // 8)) * 255).astype(np.uint8)
    phase = Image.fromarray(phase).resize(size, Image.NEAREST)
    return lights, halo, phase, wash, red


def fit_transform(img, resample):
    """Raddrizza, ritaglia i bordi neri e adatta a 16:9 con margine per lo zoom."""
    w, h = img.size
    r = img.rotate(-TILT, resample=resample, center=(w / 2, h / 2))
    t = math.radians(TILT)
    cw = int((w * math.cos(t) - h * math.sin(t)) / math.cos(2 * t))
    ch = int((h * math.cos(t) - w * math.sin(t)) / math.cos(2 * t))
    r = r.crop(((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2))
    if W > H:  # 16:9: tiene l'orizzonte a ~45% dall'alto
        tw = r.width
        th = int(tw * H / W)
        top = int(min(max(r.height * 0.36 - th * 0.45 + 180, 0), r.height - th))
        r = r.crop((0, top, tw, top + th))
    else:  # 9:16: tutta l'altezza, stretto sul promontorio
        th = r.height
        tw = int(th * W / H)
        cx = CENTER_X - (w - cw) // 2
        left = int(min(max(cx - tw / 2, 0), r.width - tw))
        r = r.crop((left, 0, left + tw, th))
    return r.resize((int(W * MARGIN), int(H * MARGIN)), resample)


def to_f(img):
    return np.asarray(img).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="scauri_foto.jpg")
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default="scauri_foto_notte.mp4")
    ap.add_argument("--no-title", action="store_true")
    ap.add_argument("--verticale", action="store_true", help="1080x1920 per Reels/TikTok")
    args = ap.parse_args()
    global W, H
    if args.verticale:
        W, H = 1080, 1920

    src = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")
    sky_o, sea_o, _ = masks_original(src)
    lights_o, halo_o, phase_o, wash_o, red_o = light_layers(src.size)

    photo = to_f(fit_transform(src, Image.BICUBIC)) / 255
    sky = to_f(fit_transform(Image.fromarray((sky_o * 255).astype(np.uint8)), Image.BILINEAR)) / 255
    sea = to_f(fit_transform(Image.fromarray((sea_o * 255).astype(np.uint8)), Image.BILINEAR)) / 255
    sky = to_f(Image.fromarray((sky * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))) / 255
    lights = to_f(fit_transform(lights_o, Image.BILINEAR)) / 255
    halo = to_f(fit_transform(halo_o, Image.BILINEAR)) / 255
    phase = to_f(fit_transform(phase_o, Image.NEAREST)) / 255 * 6.28
    wash = to_f(fit_transform(wash_o, Image.BILINEAR))[..., None] / 255
    red = to_f(fit_transform(red_o, Image.BILINEAR)) / 255
    red = red / max(red.max(), 1e-6)
    bh, bw = sky.shape
    land = np.clip(1 - sky - sea, 0, 1)
    yy, xx = np.mgrid[0:bh, 0:bw].astype(np.float32)
    sc = bh / 806.0  # scala delle misure in pixel rispetto al formato 1280x720
    xs, ys = xx / sc, yy / sc

    # Orizzonte nel fotogramma di lavoro: prima riga di mare per colonna
    hz_row = np.argmax(sea > 0.5, axis=0).astype(np.float32)
    hz_row[hz_row == 0] = bh
    horizon = float(np.median(hz_row[: bw // 3]))
    depth = np.clip((yy - hz_row[None, :]) / (bh - horizon), 0, 1)

    # --- grading notturno ---
    lum = photo @ np.array([0.299, 0.587, 0.114], np.float32)
    t_sky = np.clip(yy / horizon, 0, 1)[..., None]
    night_sky = (np.array([0.02, 0.03, 0.09]) * (1 - t_sky) + np.array([0.11, 0.12, 0.22]) * t_sky)
    warm = np.exp(-(((xx - bw) / (bw * 0.35)) ** 2) - ((yy - horizon) / (bh * 0.18)) ** 2)[..., None]
    night_sky = night_sky + warm * np.array([0.10, 0.06, 0.06])  # ultimo chiarore a ovest
    night_sky = night_sky + (photo - photo.mean((0, 1))) * 0.10  # tiene la texture delle nuvole
    land_n = (photo ** 1.3) * 0.22 * np.array([0.7, 0.8, 1.1])
    sea_n = (photo ** 1.7) * 0.42 * np.array([0.62, 0.75, 1.1])

    base = (night_sky * sky[..., None] + land_n * land[..., None] + sea_n * sea[..., None])
    base += photo * wash * np.array([1.1, 0.7, 0.35]) * 0.6  # torre illuminata

    # Riflessi delle luci della costa: strisce verticali sotto la riva
    top = int(horizon - 40 * sc)
    rl = int(140 * sc)
    strip = Image.fromarray((np.clip(lights[top:int(horizon)] + halo[top:int(horizon)] * 0.5, 0, 1) * 255).astype(np.uint8))
    strip = strip.transpose(Image.FLIP_TOP_BOTTOM).resize((bw, rl), Image.BILINEAR)
    strip = strip.filter(ImageFilter.BoxBlur(1)).filter(ImageFilter.GaussianBlur(1.2))
    refl = np.zeros_like(base)
    y0 = int(horizon) + 1
    y1 = min(bh, y0 + rl)
    fade = np.exp(-np.arange(y1 - y0) / (55.0 * sc))[:, None, None]
    refl[y0:y1] = to_f(strip)[: y1 - y0] / 255 * fade * 0.55
    base += refl * sea[..., None]

    foam = np.clip((lum - 0.55) / 0.35, 0, 1) * sea  # creste e schiuma, colpite dalla luna

    # --- stelle ---
    rng = np.random.default_rng(5)
    stars = []
    while len(stars) < int(320 * bw * bh / (1433 * 806)):
        x, y = int(rng.integers(0, bw)), int(rng.integers(0, int(horizon * 0.95)))
        if sky[y, x] > 0.99:
            stars.append((x, y, rng.uniform(0.25, 1.0), rng.uniform(0.6, 3.0), rng.uniform(0, 6.28)))

    if args.verticale:
        moon_x, moon_y = bw * 0.26, bh * 0.12
    else:
        moon_x, moon_y = bw * 0.30, bh * 0.17
    moon_r = 20.0 * sc

    try:
        font = ImageFont.truetype("DejaVuSerif.ttf", 54 if W > H else 88)
        small = ImageFont.truetype("DejaVuSans.ttf", 22 if W > H else 38)
    except OSError:
        font = small = ImageFont.load_default()

    n = int(args.seconds * args.fps)
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8,
                                pixelformat="yuv420p", macro_block_size=8)
    seay = sea > 0.5
    for i in range(n):
        t = i / args.fps
        p = i / max(n - 1, 1)
        e = p * p * (3 - 2 * p)

        # Mare vivo: spostamento ondulatorio che cresce verso la riva
        amp = (0.6 + 4.5 * depth) * sc
        dx = amp * np.sin(ys * 0.045 - t * 1.4 + xs * 0.004)
        dy = amp * 0.45 * np.sin(ys * 0.06 + xs * 0.012 - t * 1.1)
        sxf = np.clip(xx + dx * seay, 0, bw - 1.001)
        syf = np.clip(yy + dy * seay, 0, bh - 1.001)
        syf = np.where(seay, np.maximum(syf, hz_row[sxf.astype(np.int32)] + 2), yy)
        syf = np.minimum(syf, bh - 1.001)
        x0, y0i = sxf.astype(np.int32), syf.astype(np.int32)
        fx, fy = (sxf - x0)[..., None], (syf - y0i)[..., None]

        def samp(img):
            im = img if img.ndim == 3 else img[..., None]
            r = (im[y0i, x0] * (1 - fx) * (1 - fy) + im[y0i, x0 + 1] * fx * (1 - fy)
                 + im[y0i + 1, x0] * (1 - fx) * fy + im[y0i + 1, x0 + 1] * fx * fy)
            return r if img.ndim == 3 else r[..., 0]

        f = samp(base)
        fm = samp(foam)

        # Luna e riflesso
        d = np.sqrt((xx - moon_x) ** 2 + (yy - moon_y) ** 2)
        f += (np.exp(-(d / (90 * sc)) ** 2) * 0.22 * sky)[..., None] * np.array([0.6, 0.65, 0.85])
        disk = np.clip(moon_r - d, 0, 1)[..., None]
        f = f * (1 - disk) + disk * np.array([0.97, 0.95, 0.88])
        spread = (12 + depth * 260) * sc
        beam = np.exp(-((xx - moon_x) / spread) ** 2) * sea
        shimmer = 0.65 + 0.35 * np.sin(t * 2.6 + phase * 3)
        f += (beam * (0.06 + 0.75 * fm * shimmer))[..., None] * np.array([0.85, 0.83, 0.72])
        f += (fm * 0.16)[..., None] * np.array([0.7, 0.78, 0.95])  # schiuma al chiaro di luna

        # Stelle
        for x, y, b, sp, ph in stars:
            f[y, x] += 0.9 * b * (0.6 + 0.4 * math.sin(t * sp + ph)) * sky[y, x]

        # Luci della costa (tremolio) e luce rossa di segnalazione
        flick = 0.8 + 0.2 * np.sin(t * 5 + phase[..., None])
        f = np.maximum(f, lights * flick) + halo * 0.9
        blink = max(0.0, math.sin(t * 2.4)) ** 6
        f += (red * blink)[..., None] * np.array([1.0, 0.1, 0.05])

        # Zoom lento
        img = Image.fromarray((np.clip(f, 0, 1) * 255).astype(np.uint8))
        z = 1 + (MARGIN - 1) * e
        cw, ch = bw / z, bh / z
        cx, cy = bw / 2 + (bw - cw) * (0.15 if W > H else 0.0) * e, bh / 2
        img = img.resize((W, H), Image.BICUBIC, box=(cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2))

        if not args.no_title:
            a = min(1.0, max(0.0, (t - 1.5) / 2.0)) * min(1.0, max(0.0, (args.seconds - 1 - t) / 1.5))
            if a > 0:
                layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                dr = ImageDraw.Draw(layer)
                title, sub = "Scauri · Minturno", "Monte d'Oro, di notte"
                tw, sw = dr.textlength(title, font=font), dr.textlength(sub, font=small)
                ty = H - 190 if W > H else H - 420
                dr.text(((W - tw) / 2, ty), title, font=font, fill=(250, 242, 225, int(240 * a)))
                dr.text(((W - sw) / 2, ty + font.size * 1.25), sub, font=small, fill=(215, 218, 235, int(210 * a)))
                shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                shadow.putalpha(layer.getchannel("A").filter(ImageFilter.GaussianBlur(8)))
                img = img.convert("RGBA")
                img.alpha_composite(shadow)
                img.alpha_composite(layer)
                img = img.convert("RGB")

        k = max(0.0, min(1.0, t / 1.0, (args.seconds - t) / 1.0))
        writer.append_data((np.asarray(img).astype(np.float32) * k).astype(np.uint8))
    writer.close()
    print(f"Scritto {args.out} ({n} frame, {W}x{H} @ {args.fps}fps)")


if __name__ == "__main__":
    main()
