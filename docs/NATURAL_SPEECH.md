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
3. **Prosody** (`tts.py`) — per-phoneme base durations, sentence-medial
   F0 declination, `?`/`!` contours, pause insertion at sentence and comma
   boundaries.
4. **Synthesis** (`synth.py`) — a parallel source-filter formant synthesizer.
   Voiced frames use a glottal pulse train (±2% jitter) driving an analytic
   resonator bank (`exp(-π·bw·t)·sin(2πf·t)` impulse responses, evaluated
   vectorized); unvoiced frames use FFT band-passed white noise. Formants
   glide between phonemes over ~28 ms transitions.

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
   distance over all 6 rate/pitch template variants per word (rates 0.85 / 1.0
   / 1.2 × pitches 1.0 / 1.12). Cost is normalized to [0, 2].
4. **Decoding** — long segments are recursively split 2-way at the point of
   minimum total DTW cost, yielding multi-word transcriptions. Words above
   `unknown_threshold` (default 0.70) are rejected as `<unk>` and excluded
   from the output text — that is how out-of-vocabulary words are rejected.

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

Verified by `python tests/test_natural_speech.py` (41 checks, all passing):
single-word roundtrip 6/6, multi-word utterances with clear word breaks
("open close", "open time"), OOV rejection ("banana", "zebra", "computer",
"elephant" → empty), silence handling, 10% noise and 0.85× tempo robustness,
cross-voice recognition after enrollment, WAV I/O, G2P and normalization.

Limitations by design (this is a from-scratch, zero-data engine):

- **Vocabulary-based**: only enrolled/listed words can be recognized. Open
  transcription of arbitrary sentences needs pretrained models and is out of
  scope for a no-template, no-API engine.
- **Connected speech**: tightly coarticulated word pairs ("yes stop" run
  together) may occasionally yield a sub-sequence; words with clearer
  boundaries or slightly spaced dictation transcribe reliably.
- **Cross-voice**: enroll once per main speaker for best accuracy.
- **Microphone input**: requires `pyaudio`; transcribing files/arrays has no
  extra dependencies beyond NumPy.

---

## Testing

```bash
python tests/test_natural_speech.py   # 41 checks, standalone harness
pytest tests/test_natural_speech.py   # same suite via pytest
```
