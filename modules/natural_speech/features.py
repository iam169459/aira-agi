"""MFCC feature extraction, from scratch.

Pipeline: pre-emphasis -> 25 ms Hamming frames (10 ms hop) -> power spectrum
-> mel filterbank (triangular filters, no external data) -> log -> DCT-II ->
cepstral coefficients + log energy -> cepstral mean normalization.
"""
import numpy as np


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def build_mel_filterbank(n_filters, n_fft, sample_rate, fmin=50.0, fmax=None):
    fmax = fmax or sample_rate / 2.0
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_filters + 2)
    hz_pts = mel_to_hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sample_rate).astype(int)
    filters = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(n_filters):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center == left:
            center = left + 1
        if right == center:
            right = center + 1
        for k in range(left, min(center, filters.shape[1])):
            filters[i, k] = (k - left) / max(center - left, 1)
        for k in range(center, min(right, filters.shape[1])):
            filters[i, k] = (right - k) / max(right - center, 1)
    return filters


_DCT_CACHE = {}


def dct_matrix(n, k):
    key = (n, k)
    if key not in _DCT_CACHE:
        basis = np.arange(n)[None, :]
        freqs = np.arange(k)[:, None]
        _DCT_CACHE[key] = np.cos(np.pi * freqs * (2 * basis + 1) / (2 * n))
    return _DCT_CACHE[key]


class MFCCExtractor:
    def __init__(self, sample_rate=16000, frame_ms=25, hop_ms=10,
                 n_filters=26, n_cepstra=13, preemphasis=0.97):
        self.sr = sample_rate
        self.frame_len = int(frame_ms / 1000.0 * sample_rate)
        self.hop = int(hop_ms / 1000.0 * sample_rate)
        self.n_filters = n_filters
        self.n_cepstra = n_cepstra
        self.preemphasis = preemphasis
        self.window = np.hamming(self.frame_len)
        self.filterbank = build_mel_filterbank(n_filters, self.frame_len,
                                               sample_rate)

    def extract(self, samples):
        """Return (cepstra, frame_energies).

        cepstra: (n_frames, n_cepstra), cepstral mean/variance normalized.
        frame_energies: (n_frames,) log energy, used for VAD/segmentation
        only (kept out of the matching features on purpose).
        """
        samples = np.asarray(samples, dtype=np.float64)
        if len(samples) < self.frame_len:
            samples = np.pad(samples, (0, self.frame_len - len(samples)))
        emphasized = np.append(samples[0], samples[1:] -
                               self.preemphasis * samples[:-1])
        n_frames = 1 + max(0, (len(emphasized) - self.frame_len) // self.hop)
        idx = np.arange(self.frame_len)[None, :] + \
            self.hop * np.arange(n_frames)[:, None]
        frames = emphasized[idx] * self.window[None, :]

        spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
        mel = spec @ self.filterbank.T
        mel = np.where(mel < 1e-10, 1e-10, mel)
        log_mel = np.log(mel)

        dct = dct_matrix(self.n_filters, self.n_cepstra)
        cepstra = log_mel @ dct.T

        frame_energy = np.log(np.sum(frames ** 2, axis=1) + 1e-10)
        return self.normalize(cepstra), frame_energy

    @staticmethod
    def normalize(cepstra):
        mean = cepstra.mean(axis=0, keepdims=True)
        std = cepstra.std(axis=0, keepdims=True)
        return (cepstra - mean) / np.maximum(std, 1e-6)
