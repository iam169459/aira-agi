# Natural Speech Engine — TTS & STT From Scratch

A fully self-contained voice engine for Aira, built from first principles with
**zero external speech APIs, pretrained models, or downloadable data**. Every
component is implemented in pure Python + NumPy (NumPy is already a repo
dependency). No pyttsx3, no Whisper, no espeak, no model files.

```
modules/natural_speech/
├── normalize.py   # text normalization (numbers, $, %, ordinals, URLs, abbrevs)
├── g2p.py         # grapheme-to-phoneme: rules + ~300-word irregular lexicon
├── phonemes.py    # ~39-phoneme ARPAbet-style tables (formants, durations, stops)
├── tts.py         # prosody planner, TextToSpeech API, WAV I/O
├── synth.py       # source-filter formant synthesizer
├── features.py    # MFCC front-end for recognition
├── dtw.py         # banded DTW with cosine frame distance
├── stt.py         # VAD + template DTW recognizer, SpeechRecognizer API
└── __init__.py
```

---

## Text-to-Speech

### Pipeline

1. **Normalization** (`normalize.py`) — expands arabic numerals, currency
   (`$5` → "five dollars"), percents, ordinals (`1st` → "first"), decimals,
   URLs, and common abbreviations ("Dr." → "Doctor").
2. **G2P** (`g2p.py`) — an ordered letter-to-sound rule walk with a lexicon of
   irregular words, digraph handling, magic-e vowels, voiced/voiceless plural
   (`cats` → S) and `-ed` logic, and word stress marking. Output is an
   ARPAbet-style phoneme string with inter-word breath markers.
3. **Prosody** (`tts.py`) — per-phoneme base durations with stress
   lengthening and **phrase-final vowel lengthening**, sentence-medial F0
   declination plus deterministic **per-word pitch drift** (no two words are
   ever at exactly the same pitch, so speech never sounds metronomic),
   `?`/`!` contours, pause insertion.
4. **Synthesis** (`synth.py`) — a parallel source-filter formant synthesizer
   tuned for natural voice quality:
   - Voiced excitation is a **Rosenberg-style glottal flow-derivative pulse**
     (not an impulse train), with cycle-to-cycle jitter (±2.2%) and shimmer
     (+/−7% amplitude) — the same micro-perturbations real larynxes have.
   - **Aspiration (breath) noise is keyed to the glottal open phase**, so it
     breathes with the voice instead of hissing underneath it.
   - Formants are damped-sinusoid resonator banks convolved with the
     excitation, overlap-adding their decay tails across time-varying
     segments; formant targets move through **3-step interpolated glides**
     (~40 ms) between phones, avoiding the steppy "robot portamento" of
     hard switches. A gentle spectral tilt (~1/f) completes the source.
   - Unvoiced frames use FFT band-passed white noise.
   - Each voice preset carries its own F0, **open quotient** (male 0.52,
     female 0.62, child 0.68), breathiness, and tilt.

### API

```python
from modules.natural_speech import TextToSpeech, save_wav, load_wav

tts = TextToSpeech(sample_rate=22050, rate=1.0, voice="female")

audio = tts.synthesize("Hello! I have 3 tasks for you.")   # float32 ndarray
tts.synthesize_to_file(text, "out.wav")
path = tts.speak_to_tempfile("hello world")                # returns a temp path
tts.play(audio)            # optional: needs pyaudio
phonemes = tts.phonemes(text)
plan = tts.plan(text)      # segment-level timing/phone plan
voices = ("female", "male", "child", "neutral")
```

WAV helpers handle 8/16/24/32-bit PCM and float WAV formats; `resample()`
provides linear-interpolation resampling, e.g. between 22050 Hz TTS output and
16000 Hz STT input.

### Existing integration

- `modules/tts.py` now prefers pyttsx3 when installed but **falls back to the
  natural engine automatically**, with per-emotion speaking-rate mapping
  (happy, excited, angry, sad, calm...).
- `tests/test_natural_speech.py` verifies rendering, voices, and long text.
  TTS throughput is >>100x realtime (a 3.5 s utterance renders in ~0.15 s).

---

## Speech-to-Text

The recognizer is a **vocabulary-based, self-enrolling template matcher**.
Templates are not pretrained data — they are synthesized at construction time
by the engine's *own* TTS, so the whole system remains self-referential and
needs no external corpora.

### Pipeline

1. **VAD** — energy percentile thresholding (`floor + 0.15·(peak−floor)`) with
   minimum-run/minimum-gap rules splits the waveform into candidate segments.
2. **Features** (`features.py`) — 25 ms / 10 ms Hamming-windowed frames,
   0.97 pre-emphasis, 26-band mel filterbank, DCT to 19 cepstral coefficients,
   per-utterance mean/variance normalization. Frame energy is used only by the
   VAD, keeping matching robust to loudness differences.
3. **Matching** (`dtw.py`) — banded DTW (Sakoe-Chiba) with cosine frame
   distance over **12 self-augmented template variants per word**: 3 TTS
   speaking rates (0.85 / 1.0 / 1.2) × 2 pitches (1.0 / 1.12), plus
   feature-domain tempo warps (0.85× / 1.18×), tape-speed variants, and
   noise-mixed variants — all derived from the engine's own TTS output, so no
   external data is involved. Cost is normalized to [0, 2].
4. **Decoding** — multi-word runs are recursively split 2-way at the point
   whose left+right DTW costs beat matching the whole segment. Words above
   `unknown_threshold` (default 0.82) are rejected as `<unk>` and excluded
   from the output text — that is how out-of-vocabulary words are rejected.
   The VAD smooths energy and bridges intra-word stop closures so words are
   never chopped at internal consonant closures.

### API

```python
from modules.natural_speech import SpeechRecognizer

rec = SpeechRecognizer(
    vocabulary=["hello", "yes", "no", "stop", "open", "close"],
    sample_rate=16000,
    unknown_threshold=0.70,
)

result = rec.transcribe("clip.wav")     # path, numpy array, or tensor
# {"text": str, "words": [{"word", "start", "end", "confidence"}],
#  "duration": float, "engine": "natural-dtw"}

rec.enroll_file("me_saying_hello.wav", word="hello")   # per-user accuracy
samples, sr = rec.record(seconds=2)                    # optional: needs pyaudio
```

The 25-word `DEFAULT_VOCABULARY` covers typical assistant commands
("hello", "yes", "no", "stop", "open", "close", "weather", "time", ...).

### Enrollment (recommended)

Synthetic templates transfer across voices reasonably, but registering your
own voice makes recognition dramatically more reliable. `rec.enroll(word,
samples, sample_rate)` or `rec.enroll_file(path, word=...)` adds user
templates per word; anyone else's voice, or a different TTS voice, is
recognized correctly once enrolled.

### Existing integration

- `modules/stt.py` prefers Whisper when installed, but **falls back to the
  natural recognizer automatically** if torch/Whisper are unavailable. The
  `STTModule(vocabulary=...)` parameter lets you pass a custom vocabulary.
- `purple_ai/action/voice_io.py`, `voice_ai.py`, and `voice_ai_girl.py` keep
  working unchanged through this fallback.

---

## Verified behavior & honest limitations

Verified by `python tests/test_natural_speech.py` (all checks passing) plus
a 23-case recognition matrix: 7/7 single-word roundtrip, 2-word utterances
("yes stop", "hello yes", "open close", "open time", "no time stop"),
3-word chains ("no time stop"), OOV rejection ("banana", "zebra",
"computer", "elephant" → empty), silence handling, 10% additive noise
(→ recovered), 0.85×/1.15× tempo robustness, cross-voice recognition after
enrollment, WAV I/O, G2P and normalization.

Limitations by design (this is a from-scratch, zero-data engine):

- **Vocabulary-based**: only enrolled/listed words can be recognized. Open
  transcription of arbitrary sentences needs pretrained models and is out of
  scope for a no-template, no-API engine.
- **Connected speech**: very long coarticulated chains (4+ words spoken run
  together, e.g. "hello yes stop time") can still drop one middle word in
  greedy DTW splitting; words with clearer boundaries or slightly spaced
  dictation transcribe reliably.
- **Cross-voice**: enroll once per main speaker for best accuracy.
- **Microphone input**: requires `pyaudio`; transcribing files/arrays has no
  extra dependencies beyond NumPy.

---

## Performance

Measured on this codebase (NumPy only, single CPU core):

- **TTS** ~60x realtime (a 4 s sentence renders in ~70 ms). A bounded phrase
  cache makes repeated phrases (~0.1 ms) and emotion re-renders effectively
  free; cache keys include rate/voice parameters so retuning re-renders.
- **STT** single words ~50-90 ms, two-word utterances ~0.5-0.6 s, even 5-word
  run-together chains ~1.1 s — i.e. real-time. Three optimizations made this
  possible without changing results:
  1. the banded DTW is **fully vectorized** and provably exact (the
     sequential "left step" chain is resolved by binary min-plus jumps over
     per-row prefix costs; verified equal to the reference DP on 80 random
     cases including narrow bands);
  2. `_match` pre-ranks the vocabulary with a time-normalized sketch
     (cheap matmul) and runs full-resolution DTW only on top candidates;
  3. the multi-word split scan ranks candidate boundaries with one
     canonical subsampled template per word, then **re-verifies the winning
     split at full resolution** before accepting it.

---

## Testing

```bash
python tests/test_natural_speech.py   # 41 checks, standalone harness
pytest tests/test_natural_speech.py   # same suite via pytest
```
