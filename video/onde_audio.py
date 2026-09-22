"""Sintetizza il suono della risacca (stereo, 44.1 kHz) e lo unisce a un video.

Nessun campione esterno: rumore filtrato nello spettro, con onde che si infrangono
a intervalli irregolari, schiuma che sfrigola e un fondo basso continuo.

Uso:  python3 onde_audio.py video.mp4 video_con_audio.mp4 [--seed 1]
"""
import argparse
import subprocess
import wave

import imageio_ffmpeg
import numpy as np

SR = 44100


def shaped_noise(n, rng, lo, hi, tilt):
    """Rumore bianco filtrato in frequenza: banda [lo, hi] Hz, pendenza 1/f^tilt."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    g = np.where(f > 0, (np.maximum(f, 1) / 100.0) ** (-tilt / 2), 0)
    g *= 1 / (1 + (lo / np.maximum(f, 1)) ** 4)  # passa-alto morbido
    g *= 1 / (1 + (f / hi) ** 4)  # passa-basso morbido
    x = np.fft.irfft(spec * g, n)
    return x / np.abs(x).max()


def envelope(n, dur, rng):
    t = np.arange(n) / SR
    env = np.zeros(n)
    wash = np.zeros(n)
    s = rng.uniform(0.3, 1.5)
    while s < dur + 3:
        rise, fall = rng.uniform(0.8, 1.6), rng.uniform(2.5, 4.5)
        a = rng.uniform(0.6, 1.0)
        k = t - s
        e = np.where(k < 0, 0, np.where(k < rise, (k / rise) ** 2, np.exp(-(k - rise) / (fall / 3))))
        env += a * e
        # la schiuma che si ritira arriva dopo il frangente
        k2 = k - rise - 0.4
        wash += a * np.where(k2 < 0, 0, np.sin(np.clip(k2 / fall, 0, 1) * np.pi) ** 2)
        s += rng.uniform(4.2, 7.0)
    return env, wash


def synth(dur, seed):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    out = []
    env, wash = envelope(n, dur, rng)
    for ch in range(2):
        r = np.random.default_rng(seed * 10 + ch)
        rumble = shaped_noise(n, r, 30, 380, 1.0)
        crash = shaped_noise(n, r, 180, 4200, 0.6)
        fizz = shaped_noise(n, r, 1800, 11000, 0.2)
        grain = np.clip(shaped_noise(n, r, 2, 40, 0) * 3, 0, 1)  # crepitio irregolare delle bolle
        x = (0.35 * rumble * (0.55 + 0.45 * env)
             + 0.55 * crash * env
             + 0.22 * fizz * wash * (0.4 + 0.6 * grain))
        out.append(x)
    x = np.stack(out, 1)
    x = 0.8 * x + 0.2 * x[:, ::-1]  # un po' di correlazione tra i canali
    fade = np.minimum(1, np.minimum(np.arange(n) / SR / 1.5, (n - np.arange(n)) / SR / 1.5))
    x *= fade[:, None]
    return (x / np.abs(x).max() * 0.8 * 32767).astype(np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    _, dur = imageio_ffmpeg.count_frames_and_secs(args.video)
    wav = args.out + ".wav"
    with wave.open(wav, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(synth(dur, args.seed).tobytes())
    subprocess.run([ff, "-loglevel", "error", "-y", "-i", args.video, "-i", wav, "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", args.out], check=True)
    print(f"Scritto {args.out} ({dur:.1f}s)")


if __name__ == "__main__":
    main()
