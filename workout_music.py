#!/usr/bin/env python3
"""ワークアウト用 EDM トラックを数値合成で生成するスクリプト.

外部のサンプル音源や有料 API は使わず、numpy / scipy だけで
キック・スネア・ハイハット・ベース・パッド・リードを合成し、
ウォームアップ → ビルド → ドロップ → ブレイク → ドロップ → クールダウン
の構成で 1 曲を書き出す。

使い方:
    python workout_music.py                       # 128BPM, 約4分, output/workout_128bpm.wav/.mp3
    python workout_music.py --bpm 140 --drops 3   # 高強度向け: 速め & ドロップ3回
    python workout_music.py --bpm 100 --key D     # ウォーキング/ストレッチ向け
"""

import argparse
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
from scipy import signal

SR = 44100

NOTE_OFFSETS = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
                "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}

# マイナーキーの i - VI - III - VII 進行 (例: Am - F - C - G)。根音からの半音数とコード構成音。
PROGRESSION = [
    (0, [0, 3, 7]),
    (-4, [0, 4, 7]),
    (3, [0, 4, 7]),
    (-2, [0, 4, 7]),
]


def midi_to_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


# ---------------------------------------------------------------- 音源

def kick(rng):
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    freq = 45 + 110 * np.exp(-t * 35)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    body = np.sin(phase) * np.exp(-t * 7)
    click = rng.standard_normal(n) * np.exp(-t * 300) * 0.3
    return np.tanh((body + click) * 1.8) * 0.95


def snare(rng):
    n = int(0.25 * SR)
    t = np.arange(n) / SR
    noise = signal.lfilter(*signal.butter(2, [1500, 9000], "band", fs=SR), rng.standard_normal(n))
    tone = np.sin(2 * np.pi * 190 * t) * np.exp(-t * 30)
    return (noise * np.exp(-t * 18) * 0.7 + tone * 0.5) * 0.8


def hat(rng, open_=False):
    n = int((0.25 if open_ else 0.06) * SR)
    t = np.arange(n) / SR
    noise = signal.lfilter(*signal.butter(2, 7000, "high", fs=SR), rng.standard_normal(n))
    return noise * np.exp(-t * (12 if open_ else 70)) * (0.35 if open_ else 0.3)


def saw(freq, n, detune_cents=(0,), rng=None):
    t = np.arange(n) / SR
    out = np.zeros(n)
    for c in detune_cents:
        f = freq * 2 ** (c / 1200)
        ph = rng.random() if rng is not None else 0.0
        out += 2 * ((t * f + ph) % 1.0) - 1
    return out / len(detune_cents)


def adsr(n, a=0.005, d=0.1, s=0.7, r=0.05):
    env = np.full(n, s)
    na, nd, nr = int(a * SR), int(d * SR), int(r * SR)
    na = min(na, n)
    env[:na] = np.linspace(0, 1, na, endpoint=False)
    nd = min(nd, n - na)
    env[na:na + nd] = np.linspace(1, s, nd, endpoint=False)
    nr = min(nr, n)
    if nr:
        env[n - nr:] *= np.linspace(1, 0, nr)
    return env


def lowpass(x, cutoff, order=2):
    cutoff = float(np.clip(cutoff, 30, SR / 2 - 100))
    return signal.lfilter(*signal.butter(order, cutoff, "low", fs=SR), x)


def sweep_lowpass(x, start, end, block=1024):
    """ブロック単位でカットオフを動かすフィルタースイープ."""
    out = np.zeros_like(x)
    zi = None
    nblocks = int(np.ceil(len(x) / block))
    for i in range(nblocks):
        frac = i / max(nblocks - 1, 1)
        fc = start * (end / start) ** frac
        b, a = signal.butter(2, fc, "low", fs=SR)
        if zi is None:
            zi = np.zeros(max(len(a), len(b)) - 1)
        seg = x[i * block:(i + 1) * block]
        out[i * block:i * block + len(seg)], zi = signal.lfilter(b, a, seg, zi=zi)
    return out


# ---------------------------------------------------------------- トラック

class Track:
    def __init__(self, bpm, key, seed):
        self.bpm = bpm
        self.beat = int(round(60 / bpm * SR))
        self.bar = self.beat * 4
        self.step = self.beat // 4  # 16 分音符
        self.root = 45 + (NOTE_OFFSETS[key] - 9)  # A2=45 を基準にした根音
        self.rng = np.random.default_rng(seed)
        self.kick = kick(self.rng)
        self.snare = snare(self.rng)
        self.hat_c = hat(self.rng)
        self.hat_o = hat(self.rng, open_=True)
        self.stems = {}
        self.cursor = 0
        self.sections = []

    def _stem(self, name, length):
        buf = self.stems.setdefault(name, np.zeros(0))
        if len(buf) < length:
            buf = np.concatenate([buf, np.zeros(length - len(buf))])
            self.stems[name] = buf
        return buf

    def add(self, name, pos, sound, gain=1.0):
        buf = self._stem(name, pos + len(sound))
        buf[pos:pos + len(sound)] += sound * gain

    def chord(self, bar_idx):
        return PROGRESSION[bar_idx % len(PROGRESSION)]

    # ---- パート

    def drums(self, start, bars, kick_on=True, snare_on=True, hats="full", gain=1.0):
        for b in range(bars):
            bar0 = start + b * self.bar
            for beat in range(4):
                p = bar0 + beat * self.beat
                if kick_on:
                    self.add("kick", p, self.kick, gain)
                if snare_on and beat in (1, 3):
                    self.add("drums", p, self.snare, 0.8 * gain)
                if hats == "full":
                    self.add("drums", p + 2 * self.step, self.hat_o, 0.7 * gain)
                    for s in (1, 3):
                        self.add("drums", p + s * self.step, self.hat_c, 0.5 * gain)
                elif hats == "off":
                    self.add("drums", p + 2 * self.step, self.hat_c, 0.8 * gain)

    def bass(self, start, bars, cutoff=900):
        for b in range(bars):
            root_off, _ = self.chord(b)
            f = midi_to_hz(self.root + root_off)
            for s in range(16):
                if s % 4 == 0:  # キックと被る拍頭は空ける (オフベースのベース)
                    continue
                n = self.step
                tone = saw(f, n, (-7, 7), self.rng) * 0.6 + np.sin(2 * np.pi * f / 2 * np.arange(n) / SR) * 0.6
                tone = lowpass(tone, cutoff + 600 * (s % 4 == 2)) * adsr(n, 0.002, 0.05, 0.6, 0.01)
                self.add("bass", start + b * self.bar + s * self.step, tone, 0.55)

    def pad(self, start, bars, cutoff_start=600, cutoff_end=600, gain=0.25):
        total = np.zeros(bars * self.bar)
        for b in range(bars):
            root_off, tri = self.chord(b)
            n = self.bar
            x = np.zeros(n)
            for iv in tri:
                x += saw(midi_to_hz(self.root + 12 + root_off + iv), n, (-12, 0, 12), self.rng)
            total[b * n:(b + 1) * n] = x * adsr(n, 0.15, 0.2, 0.8, 0.15)
        total = sweep_lowpass(total, cutoff_start, cutoff_end)
        self.add("pad", start, total, gain)

    def arp(self, start, bars, cutoff=3000, gain=0.18):
        pattern = [0, 1, 2, 1, 2, 0, 1, 2]
        for b in range(bars):
            root_off, tri = self.chord(b)
            for s in range(16):
                iv = tri[pattern[s % len(pattern)]] + (12 if s % 8 >= 4 else 0)
                n = self.step
                tone = saw(midi_to_hz(self.root + 24 + root_off + iv), n, (-5, 5), self.rng)
                tone = lowpass(tone, cutoff) * adsr(n, 0.002, 0.08, 0.2, 0.01)
                self.add("lead", start + b * self.bar + s * self.step, tone, gain)

    def stabs(self, start, bars, gain=0.22):
        """ドロップ用のスーパーソー・コードスタブ (付点リズム)."""
        hits = [0, 3, 6, 10, 12]
        for b in range(bars):
            root_off, tri = self.chord(b)
            for h in hits:
                n = int(self.step * 1.6)
                x = np.zeros(n)
                for iv in tri + [12]:
                    x += saw(midi_to_hz(self.root + 24 + root_off + iv), n, (-18, -6, 0, 6, 18), self.rng)
                x = lowpass(x, 5000) * adsr(n, 0.003, 0.12, 0.35, 0.03)
                self.add("lead", start + b * self.bar + h * self.step, x / 3, gain)

    def riser(self, start, bars):
        n = bars * self.bar
        t = np.arange(n) / SR
        noise = self.rng.standard_normal(n)
        noise = sweep_lowpass(noise, 300, 12000) * np.linspace(0.02, 0.4, n)
        f = np.geomspace(200, 1200, n)
        tone = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.linspace(0, 0.12, n)
        self.add("fx", start, noise + tone)
        # スネアロール: 4分 → 8分 → 16分 と加速
        for b in range(bars):
            div = [1, 2, 4, 4][min(b * 4 // bars, 3)]
            for k in range(4 * div):
                p = start + b * self.bar + k * self.beat // div
                self.add("drums", p, self.snare, 0.3 + 0.5 * (b * 4 * div + k) / (bars * 4 * div))

    def impact(self, pos):
        n = int(1.5 * SR)
        t = np.arange(n) / SR
        boom = np.sin(2 * np.pi * (40 + 60 * np.exp(-t * 8)) * t) * np.exp(-t * 3)
        crash = signal.lfilter(*signal.butter(2, 4000, "high", fs=SR), self.rng.standard_normal(n)) * np.exp(-t * 2.5) * 0.3
        self.add("fx", pos, boom * 0.6 + crash)

    # ---- 構成

    def section(self, name, bars):
        self.sections.append((name, self.cursor, bars))
        start = self.cursor
        self.cursor += bars * self.bar
        return start

    def build(self, drops):
        s = self.section("ウォームアップ", 16)
        self.drums(s, 8, snare_on=False, hats="off", gain=0.8)
        self.drums(s + 8 * self.bar, 8, snare_on=True, hats="full", gain=0.9)
        self.pad(s, 16, 300, 2500)

        for i in range(drops):
            s = self.section(f"ビルドアップ{i + 1}", 8)
            self.drums(s, 8, kick_on=True, snare_on=False, hats="off")
            self.bass(s, 8, cutoff=400)
            self.pad(s, 8, 800, 6000, gain=0.2)
            self.arp(s, 8, cutoff=2000)
            self.riser(s, 8)

            s = self.section(f"ドロップ{i + 1} (全力)", 32)
            self.impact(s)
            self.drums(s, 32)
            self.bass(s, 32, cutoff=1100)
            self.stabs(s, 32)
            self.pad(s, 32, 3000, 3000, gain=0.12)
            if i > 0:
                self.arp(s + 16 * self.bar, 16, cutoff=5000, gain=0.12)

            if i < drops - 1:
                s = self.section(f"ブレイク{i + 1} (呼吸を整える)", 16)
                self.pad(s, 16, 4000, 800, gain=0.28)
                self.arp(s + 8 * self.bar, 8, cutoff=1500, gain=0.14)
                self.drums(s + 8 * self.bar, 8, kick_on=False, snare_on=False, hats="off", gain=0.6)

        s = self.section("クールダウン", 16)
        self.drums(s, 8, snare_on=False, hats="off", gain=0.8)
        self.pad(s, 16, 2500, 250, gain=0.3)

    def mix(self):
        length = self.cursor + 2 * SR
        def get(name):
            buf = self.stems.get(name, np.zeros(0))
            out = np.zeros(length)
            out[:min(len(buf), length)] = buf[:length]
            return out

        # キックに合わせたサイドチェイン (ポンピング)
        kick_env = np.ones(length)
        duck = 1 - 0.75 * np.exp(-np.arange(self.beat) / SR * 9)
        kicks = get("kick")
        for k in range(self.cursor // self.beat):
            seg = slice(k * self.beat, (k + 1) * self.beat)
            if np.abs(kicks[seg]).max() > 0.01:
                kick_env[seg] = duck

        mono = (get("kick") * 0.9 + get("drums") * 0.6
                + (get("bass") + get("pad") + get("lead")) * kick_env
                + get("fx") * 0.5)
        # 簡易ステレオ: パッドとリードを少しずらして広がりを出す
        delay = int(0.012 * SR)
        wide = (get("pad") + get("lead")) * kick_env * 0.35
        left = mono + np.concatenate([np.zeros(delay), wide[:-delay]]) - wide
        right = mono + wide - np.concatenate([np.zeros(delay), wide[:-delay]])
        stereo = np.stack([left, right], axis=1)

        # 終盤フェードアウト
        fade_len = 8 * self.bar
        fade_start = self.cursor - fade_len
        stereo[fade_start:self.cursor] *= np.linspace(1, 0, fade_len)[:, None]
        stereo[self.cursor:] = 0

        # ソフトクリップ & ノーマライズ
        stereo = np.tanh(stereo / np.abs(stereo).max() * 1.6)
        stereo /= np.abs(stereo).max()
        return stereo * 0.89


def write_wav(path, data):
    pcm = (np.clip(data, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def fmt_time(samples):
    s = samples / SR
    return f"{int(s // 60)}:{int(s % 60):02d}"


def main():
    ap = argparse.ArgumentParser(description="ワークアウト用 EDM トラック生成")
    ap.add_argument("--bpm", type=float, default=128, help="テンポ (ランニング 150-170 / 筋トレ 120-140 / ウォーキング 100-120)")
    ap.add_argument("--key", default="A", choices=NOTE_OFFSETS.keys(), help="マイナーキーの主音")
    ap.add_argument("--drops", type=int, default=2, help="ドロップ(高強度パート)の回数")
    ap.add_argument("--seed", type=int, default=1, help="乱数シード (音色の微妙な揺らぎ)")
    ap.add_argument("--out", default="output", help="出力フォルダ")
    args = ap.parse_args()

    track = Track(args.bpm, args.key, args.seed)
    track.build(args.drops)
    audio = track.mix()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"workout_{int(args.bpm)}bpm_{args.key.replace('#', 's')}m"
    wav = out / f"{stem}.wav"
    write_wav(wav, audio)
    print(f"WAV: {wav}  ({fmt_time(len(audio))})")

    if shutil.which("ffmpeg"):
        mp3 = out / f"{stem}.mp3"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
                        "-codec:a", "libmp3lame", "-b:a", "192k",
                        "-metadata", f"title=Workout {int(args.bpm)} BPM ({args.key}m)", str(mp3)], check=True)
        print(f"MP3: {mp3}")

    print("\n構成:")
    for name, start, bars in track.sections:
        print(f"  {fmt_time(start)}  {name}  ({bars}小節)")


if __name__ == "__main__":
    main()
