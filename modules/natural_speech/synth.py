"""Source-filter formant synthesizer, from scratch.

Voiced sounds: a glottal pulse train (with jitter) is passed through a
parallel bank of damped-sinusoid resonators whose center frequencies are the
phoneme's formants. The parallel resonator response is evaluated analytically,
so no per-sample Python recursion is needed and synthesis is fast.

Unvoiced sounds: shaped white noise (FFT band-pass) driven by a fricative /
burst / aspiration spec.

Piecewise-constant formant control points with short glide segments handle
transitions between phones.
"""
import numpy as np

from . import phonemes as ph


class FormantSynthesizer:
    def __init__(self, sample_rate=22050, base_f0=185.0, f0_range=0.18,
                 breathiness=0.018):
        self.sr = sample_rate
        self.base_f0 = base_f0
        self.f0_range = f0_range
        self.breathiness = breathiness
        self.transition_ms = 28.0
        self.ramp_ms = 6.0

    # ------------------------------------------------------------------
    def synthesize(self, events):
        """events: list of dicts {ph, start, dur, f0_start, f0_end, amp}
        (times in seconds). Returns float32 waveform in [-1, 1]."""
        if not events:
            return np.zeros(0, dtype=np.float32)
        total = max(e["start"] + e["dur"] for e in events) + 0.05
        n_total = int(total * self.sr)
        audio = np.zeros(n_total, dtype=np.float64)

        pulses = self._pulse_train(events, n_total)
        segments = self._formant_schedule(events)

        for seg in segments:
            s0 = int(seg["start"] * self.sr)
            s1 = s0 + max(1, int(seg["dur"] * self.sr))
            if seg["voiced"]:
                audio[s0:s1] += self._render_voiced(seg, pulses)
            if seg["noise"] is not None:
                audio[s0:s1] += self._render_noise(seg)
        audio = self._finish(audio)
        return audio.astype(np.float32)

    # ------------------------------------------------------------------
    def _pulse_train(self, events, n_total):
        """Glottal pulse positions and amplitudes with slight jitter."""
        pulse_t = []
        pulse_amp = []
        for e in events:
            if not self._event_voiced(e):
                continue
            s0 = int(e["start"] * self.sr)
            n = max(1, int(e["dur"] * self.sr))
            f0a = e["f0_start"]
            f0b = e["f0_end"]
            amp = e["amp"]
            rng = np.random.default_rng(int(s0) + 7919)
            tloc = 0.0
            guard = 0
            while tloc < n and guard < 100000:
                guard += 1
                frac = tloc / max(n, 1)
                f0 = f0a + (f0b - f0a) * frac
                period = self.sr / max(f0, 40.0)
                jitter = 1.0 + rng.uniform(-0.02, 0.02)
                pos = s0 + int(round(tloc))
                if pos < n_total:
                    a = amp * (0.94 + 0.06 * rng.uniform())
                    pulse_t.append(pos)
                    pulse_amp.append(a)
                tloc += period * jitter
        if not pulse_t:
            return (np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.float64))
        pulse_t = np.array(pulse_t, dtype=np.int64)
        pulse_amp = np.array(pulse_amp, dtype=np.float64)
        order = np.argsort(pulse_t)
        return pulse_t[order], pulse_amp[order]

    @staticmethod
    def _event_voiced(e):
        return e["ph"] in ph.VOICED

    # ------------------------------------------------------------------
    def _formant_schedule(self, events):
        """Build piecewise-constant formant control segments from events."""
        segments = []
        prev_formants = None
        for e in events:
            for s in self._event_segments(e, prev_formants):
                segments.append(s)
                if s["formants"] is not None:
                    prev_formants = s["formants"]
        return segments

    def _event_segments(self, e, prev_formants):
        name = e["ph"]
        group = ph.phone_group(name)
        start = e["start"]
        dur = e["dur"]
        segs = []
        noise = None
        formants = None

        if group == "vowel":
            spec = ph.VOWELS[name]
            formants = {"f": list(spec["f"]), "bw": list(spec["bw"])}
        elif group == "diphthong":
            spec = ph.DIPHTHONGS[name]
            a, b = spec["from"], spec["to"]
            p1 = [a[k] + (b[k] - a[k]) * 0.45 for k in range(3)]
            p2 = [a[k] + (b[k] - a[k]) * 0.78 for k in range(3)]
            return self._voiced_piecewise(e, prev_formants,
                                          [(0.35, p1), (0.70, p2), (1.0, b)])
        elif group in ("nasal", "liquid", "glide"):
            table = {"nasal": ph.NASALS, "liquid": ph.LIQUIDS,
                     "glide": ph.GLIDES}[group]
            spec = table[name]
            formants = {"f": list(spec["f"]), "bw": list(spec["bw"])}
        elif group == "fricative":
            spec = ph.FRICATIVES[name]
            noise = {"kind": "fricative", "spec": spec}
            if spec.get("voiced"):
                formants = {"f": [250, 1200, 2400], "bw": [150, 300, 400]}
            else:
                segs.append(self._simple(start, dur, None, noise, e))
                return segs
        elif group == "stop":
            spec = ph.STOPS[name]
            noise = {"kind": "stop", "spec": spec}
            if spec.get("voiced"):
                formants = {"f": [250, 1100, 2300], "bw": [150, 350, 450]}
            else:
                segs.append(self._simple(start, dur, None, noise, e))
                return segs
        elif group == "affricate":
            spec = ph.AFFRICATES[name]
            stop_spec = ph.STOPS[spec["stop"]]
            fric_spec = ph.FRICATIVES[spec["fric"]]
            closure = dur * 0.45
            segs.append(self._simple(start, closure, None,
                                     {"kind": "closure", "spec": stop_spec},
                                     e))
            segs.append(self._simple(start + closure, dur - closure, None,
                                     {"kind": "fricative",
                                      "spec": fric_spec}, e))
            return segs
        else:
            segs.append(self._simple(start, dur, None,
                                     {"kind": "silence",
                                      "spec": {"level": 0}}, e))
            return segs

        return self._voiced_piecewise_with_noise(e, prev_formants, formants,
                                                 noise)

    def _voiced_piecewise(self, e, prev_formants, points):
        """points: [(frac, formants), ...]: fraction of duration at which the
        formants reach that value. Emits a lead glide plus steady pieces."""
        segs = []
        dur = e["dur"]
        start = e["start"]
        cursor = 0.0
        fracs = [p[0] for p in points]
        vals = [p[1] for p in points]

        lead = self.transition_ms / 1000.0
        if prev_formants is not None and lead < dur * 0.5:
            mid = [(prev_formants["f"][k] + vals[0][k]) / 2.0 for k in range(3)]
            segs.append(self._make_seg(start, 0.0, lead, mid,
                                       self._default_bw(), e))
            cursor = lead
        for idx in range(len(points)):
            end = fracs[idx] * dur
            if end <= cursor:
                continue
            segs.append(self._make_seg(start, cursor, end - cursor, vals[idx],
                                       self._default_bw(), e))
            cursor = end
        if cursor < dur:
            segs.append(self._make_seg(start, cursor, dur - cursor,
                                       vals[-1], self._default_bw(), e))
        return segs

    def _voiced_piecewise_with_noise(self, e, prev_formants, formants, noise):
        segs = []
        start, dur = e["start"], e["dur"]
        cursor = 0.0
        lead = self.transition_ms / 1000.0
        if prev_formants is not None and lead < dur * 0.6:
            mid = [(prev_formants["f"][k] + formants["f"][k]) / 2.0
                   for k in range(3)]
            segs.append(self._make_seg(start, 0.0, lead, mid,
                                       formants["bw"], e))
            cursor = lead
        segs.append(self._make_seg(start, cursor, dur - cursor,
                                   formants["f"], formants["bw"], e))
        for s in segs:
            s["noise"] = noise
        return segs

    def _default_bw(self):
        return [80, 110, 160]

    def _make_seg(self, start, off, dur, formants, bw, event):
        return {
            "start": start + off,
            "dur": max(dur, 1.0 / self.sr),
            "formants": {"f": list(formants), "bw": list(bw)},
            "voiced": True,
            "noise": None,
            "event": event,
            "amp": event["amp"],
        }

    def _simple(self, start, dur, formants, noise, event):
        return {
            "start": start,
            "dur": max(dur, 1.0 / self.sr),
            "formants": formants,
            "voiced": formants is not None,
            "noise": noise,
            "event": event,
            "amp": event["amp"],
        }

    # ------------------------------------------------------------------
    def _render_voiced(self, seg, pulses):
        p_t, p_a = pulses
        s0 = int(seg["start"] * self.sr)
        s1 = s0 + max(1, int(seg["dur"] * self.sr))
        if len(p_t) == 0:
            return np.zeros(s1 - s0)
        lo_idx = np.searchsorted(p_t, int((seg["start"] - 0.25) * self.sr))
        hi_idx = np.searchsorted(p_t, s1)
        if lo_idx >= hi_idx:
            return np.zeros(s1 - s0)
        pt = p_t[lo_idx:hi_idx].astype(np.float64)
        pa = p_a[lo_idx:hi_idx]
        n = np.arange(s0, s1, dtype=np.float64)[:, None] - pt[None, :]
        mask = n >= 0
        n = np.where(mask, n / self.sr, 0.0)

        out = np.zeros(s1 - s0)
        f = seg["formants"]["f"]
        bw = seg["formants"]["bw"]
        gains = [1.0, 0.55, 0.28]
        for k in range(3):
            resp = (np.exp(-np.pi * bw[k] * n) *
                    np.sin(2 * np.pi * f[k] * n) * gains[k])
            contrib = np.where(mask, resp * pa[None, :], 0.0).sum(axis=1)
            out += contrib
        env = self._amp_envelope(s1 - s0)
        rng = np.random.default_rng(s0 * 17 + 3)
        breath = rng.uniform(-1, 1, s1 - s0) * self.breathiness
        out += breath * env
        out *= env
        return out

    def _render_noise(self, seg):
        spec = seg["noise"]["spec"]
        kind = seg["noise"]["kind"]
        s0 = int(seg["start"] * self.sr)
        n_samples = max(1, int(seg["dur"] * self.sr))
        if kind in ("silence", "closure"):
            return np.zeros(n_samples)
        rng = np.random.default_rng(s0 * 31 + 7)
        noise = rng.uniform(-1, 1, n_samples)
        env = self._amp_envelope(n_samples)
        if kind == "fricative" and "next_vowel" in spec:
            return (self._fft_bandpass(noise, 1400, 2000) * 0.20 * env +
                    self._fft_bandpass(noise, 500, 900) * 0.10 * env)
        if kind == "fricative":
            center, width = spec["noise"]
            sig = self._fft_bandpass(noise, center, width)
            sig += self._fft_bandpass(noise, center * 1.8, width * 1.6) * 0.5
            return sig * spec.get("level", 0.2) * env
        if kind == "stop":
            out = np.zeros(n_samples)
            burst_n = min(max(1, int(0.010 * self.sr)), n_samples)
            sig = self._fft_bandpass(noise[:burst_n], spec["burst"],
                                     spec["bw"]) * 1.4
            out[:burst_n] = sig
            asp_ms = spec.get("aspirate", 0)
            if asp_ms > 0:
                asp_n = min(int(asp_ms / 1000.0 * self.sr),
                            n_samples - burst_n)
                if asp_n > 0:
                    asp = self._fft_bandpass(
                        noise[burst_n:burst_n + asp_n], 1200, 1800) * 0.20
                    out[burst_n:burst_n + asp_n] += asp
            return out * env
        return noise * 0.1 * env

    def _fft_bandpass(self, noise, center, width):
        n = len(noise)
        spec = np.fft.rfft(noise)
        freqs = np.fft.rfftfreq(n, 1.0 / self.sr)
        mag = self._resonator_mag(freqs, center, width)
        mag = mag / (mag.max() + 1e-9)
        return np.fft.irfft(spec * mag, n)

    @staticmethod
    def _resonator_mag(freqs, center, width):
        q = max(center / max(width, 50.0), 0.3)
        x = freqs / max(center, 1.0)
        return 1.0 / np.sqrt((1 - x * x) ** 2 + (x / q) ** 2)

    def _amp_envelope(self, n_samples):
        ramp = min(int(self.ramp_ms / 1000.0 * self.sr), n_samples // 2)
        env = np.ones(n_samples)
        if ramp > 0:
            env[:ramp] = np.linspace(0.0, 1.0, ramp)
            env[-ramp:] = np.linspace(1.0, 0.0, ramp)
        return env

    # ------------------------------------------------------------------
    def _finish(self, audio):
        audio = audio - np.mean(audio)
        peak = np.max(np.abs(audio)) if len(audio) else 0.0
        if peak > 0:
            audio = audio / peak * 0.92
        return np.clip(audio, -1.0, 1.0)
