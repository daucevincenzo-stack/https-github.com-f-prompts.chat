"""Trasforma una foto reale (es. Scauri/Minturno di notte) in un breve video animato.

Effetti: zoom e panoramica lenti (Ken Burns), luci della foto che brillano,
eventuale resa notturna di una foto diurna (--notte), titolo in dissolvenza, vignettatura.

Uso:
  python3 foto_in_video.py --image scauri.jpg
  python3 foto_in_video.py --image scauri_giorno.jpg --notte --verticale --out reel.mp4
Dipendenze: numpy, pillow, imageio, imageio-ffmpeg
"""
import argparse

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


def notte(img):
    """Resa "day for night": abbassa l'esposizione e vira al blu, conserva le alte luci."""
    a = np.asarray(img).astype(np.float32) / 255.0
    lum = a @ np.array([0.299, 0.587, 0.114], np.float32)
    dark = (a ** 1.6) * 0.38
    dark = dark * np.array([0.55, 0.7, 1.15], np.float32)
    highlights = np.clip((lum - 0.85) / 0.15, 0, 1)[..., None]
    out = dark * (1 - highlights) + a * np.array([1.0, 0.85, 0.6], np.float32) * highlights
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="scauri_minturno_foto.mp4")
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--title", default="Scauri · Minturno")
    ap.add_argument("--subtitle", default="Golfo di Gaeta, di notte")
    ap.add_argument("--no-title", action="store_true")
    ap.add_argument("--notte", action="store_true", help="scurisce una foto diurna")
    ap.add_argument("--verticale", action="store_true", help="9:16 (1080x1920) per Reels/TikTok")
    ap.add_argument("--zoom", type=float, default=1.15, help="zoom finale (1.0 = fermo)")
    args = ap.parse_args()

    W, H = (1080, 1920) if args.verticale else (1280, 720)
    src = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")
    if args.notte:
        src = notte(src)

    # Ritaglio al formato di uscita con margine per lo zoom
    base = ImageOps.fit(src, (int(W * args.zoom) + 2, int(H * args.zoom) + 2), Image.LANCZOS)
    arr = np.asarray(base).astype(np.float32)

    # Punti luce: pixel molto chiari, che faremo brillare
    lum = arr @ np.array([0.299, 0.587, 0.114], np.float32)
    spots = np.clip((lum - 200) / 55, 0, 1)
    rng = np.random.default_rng(3)
    phase = np.asarray(Image.fromarray((rng.random(spots.shape) * 255).astype(np.uint8))
                       .filter(ImageFilter.GaussianBlur(3))).astype(np.float32) / 255 * 40
    bloom = np.asarray(Image.fromarray((spots * 255).astype(np.uint8))
                       .filter(ImageFilter.GaussianBlur(6))).astype(np.float32) / 255

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    vign = 1 - 0.45 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) ** 1.2
    vign = np.clip(vign, 0.35, 1)[..., None]

    fs = int(H * (0.075 if not args.verticale else 0.045))
    try:
        font = ImageFont.truetype("DejaVuSerif.ttf", fs)
        small = ImageFont.truetype("DejaVuSans.ttf", int(fs * 0.42))
    except OSError:
        font = small = ImageFont.load_default()

    n = int(args.seconds * args.fps)
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8,
                                pixelformat="yuv420p", macro_block_size=8)
    bw, bh = base.size
    for i in range(n):
        t = i / args.fps
        p = i / max(n - 1, 1)
        e = p * p * (3 - 2 * p)  # easing morbido

        twinkle = 1 + 0.35 * spots * np.sin(t * 3.0 + phase)
        glow = bloom * (0.5 + 0.5 * np.sin(t * 1.7 + phase * 0.5))
        frame = arr * twinkle[..., None] + glow[..., None] * np.array([60, 45, 25], np.float32)
        img = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8))

        z = args.zoom - (args.zoom - 1.0) * (1 - e)  # da 1.0 a zoom
        cw, ch = bw / z, bh / z
        cx = bw / 2 + (bw - cw) / 2 * (0.6 * e - 0.3)  # leggera panoramica verso destra
        cy = bh / 2
        box = (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)
        img = img.resize((W, H), Image.BICUBIC, box=box)
        out = np.asarray(img).astype(np.float32) * vign

        if not args.no_title:
            a = min(1.0, max(0.0, (t - 1.5) / 2.0)) * min(1.0, max(0.0, (args.seconds - 1 - t) / 1.5))
            if a > 0:
                layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                d = ImageDraw.Draw(layer)
                ty = H * 0.72
                tw = d.textlength(args.title, font=font)
                sw = d.textlength(args.subtitle, font=small)
                d.text(((W - tw) / 2, ty), args.title, font=font, fill=(250, 242, 225, int(240 * a)))
                d.text(((W - sw) / 2, ty + fs * 1.25), args.subtitle, font=small,
                       fill=(220, 220, 235, int(210 * a)))
                shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                shadow.putalpha(layer.getchannel("A").filter(ImageFilter.GaussianBlur(8)))
                comp = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).convert("RGBA")
                comp.alpha_composite(shadow)
                comp.alpha_composite(layer)
                out = np.asarray(comp.convert("RGB")).astype(np.float32)

        k = max(0.0, min(1.0, t / 1.0, (args.seconds - t) / 1.0))
        writer.append_data(np.clip(out * k, 0, 255).astype(np.uint8))
    writer.close()
    print(f"Scritto {args.out} ({n} frame, {W}x{H} @ {args.fps}fps)")


if __name__ == "__main__":
    main()
