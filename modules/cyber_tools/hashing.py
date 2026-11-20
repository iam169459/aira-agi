"""Digest, password-strength, hash-ID, and file-integrity utilities."""
import hashlib
import hmac as _hmac
import math
import os
import re
import secrets

COMMON_PASSWORD_PATTERNS = [
    (re.compile(r"^[0-9]+$"), "digits only"),
    (re.compile(r"^[a-z]+$"), "lowercase only"),
    (re.compile(r"^[A-Z]+$"), "uppercase only"),
    (re.compile(r"^(.)\1+$"), "single repeated character"),
    (re.compile(r"(19|20)\d{2}"), "contains a year"),
    (re.compile(r"(012|123|234|345|456|567|678|789|abc|qwer|asdf|zxcv)",
                re.IGNORECASE), "keyboard or sequence pattern"),
    (re.compile(r"password|passwort|letmein|welcome|admin|iloveyou|dragon",
                re.IGNORECASE), "common password word"),
]

_CHARSETS = [
    (re.compile(r"[a-z]"), 26),
    (re.compile(r"[A-Z]"), 26),
    (re.compile(r"[0-9]"), 10),
    (re.compile(r"[^a-zA-Z0-9]"), 33),
]


def digest_bytes(data, algo="sha256"):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.new(algo, data).hexdigest()


def digest_file(path, algo="sha256", chunk_size=1 << 20):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def hmac_digest(key, message, algo="sha256"):
    if isinstance(message, str):
        message = message.encode("utf-8")
    return _hmac.new(key, message, algo).hexdigest()


_HEX_LENGTHS = {32: ("md5", 2), 40: ("sha1", 2), 56: ("sha224", 2),
                64: ("sha256", 2), 96: ("sha384", 2), 128: ("sha512", 2)}


def hash_identifier(value):
    """Guess the hash family of a digest string by its structure."""
    value = value.strip()
    is_hex = bool(re.fullmatch(r"[0-9a-fA-F]+", value))
    guesses = []
    if is_hex and len(value) in _HEX_LENGTHS:
        guesses.append(_HEX_LENGTHS[len(value)])
    if value.startswith(("$2a$", "$2b$", "$2y$")):
        guesses.append(("bcrypt", 3))
    if value.startswith("$6$"):
        guesses.append(("sha512crypt", 3))
    if value.startswith("$5$"):
        guesses.append(("sha256crypt", 3))
    if value.startswith("$argon2"):
        guesses.append(("argon2", 3))
    if re.fullmatch(r"[A-Za-z0-9+/]{22,}={0,2}", value):
        guesses.append(("base64-encoded digest", 1))
    if not guesses:
        return {"guesses": [], "best_guess": None,
                "length": len(value),
                "charset": "hex" if is_hex else "other"}
    best = max(guesses, key=lambda g: g[1])
    return {
        "best_guess": best[0],
        "guesses": [g[0] for g in sorted(guesses, key=lambda g: -g[1])],
        "length": len(value),
        "charset": "hex" if is_hex else "other",
    }


def password_strength(password):
    """Heuristic strength estimate: entropy bits, pattern penalties, 0-5 score."""
    if not password:
        return {"score": 0, "entropy_bits": 0, "verdict": "empty",
                "patterns": [], "charsets": 0}
    pool = sum(size for rx, size in _CHARSETS if rx.search(password))
    entropy = len(password) * math.log2(pool) if pool else 0.0
    patterns = [label for rx, label in COMMON_PASSWORD_PATTERNS
                if rx.search(password)]
    score = entropy / 14.0
    score -= 1.5 * len(patterns)
    if len(patterns) >= 2:
        score -= 0.5
    if len(password) < 8:
        score -= 1.5
    if len(password) >= 16 and not patterns:
        score += 0.5
    score = int(max(0, min(5, round(score))))
    verdicts = ["very weak", "weak", "fair", "good", "strong",
                "very strong"]
    return {
        "score": score,
        "entropy_bits": round(entropy, 1),
        "verdict": verdicts[score],
        "patterns": patterns,
        "charsets": sum(1 for rx, _ in _CHARSETS if rx.search(password)),
    }


def baseline_build(paths, algo="sha256"):
    """Map each file path to its digest (relative paths for portability)."""
    baseline = {}
    for root in paths:
        if os.path.isfile(root):
            baseline[os.path.basename(root)] = digest_file(root, algo)
            continue
        for dirpath, _dirs, files in os.walk(root):
            for name in sorted(files):
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, root)
                baseline[rel] = digest_file(full, algo)
    return baseline


def baseline_verify(baseline, root=None, algo="sha256"):
    """Diff the current filesystem against a baseline.

    Returns {'ok', 'changed', 'missing', 'added'}.
    """
    changed, missing, added = [], [], []
    for rel, expected in baseline.items():
        path = rel if root is None else os.path.join(root, rel)
        if not os.path.isfile(path):
            missing.append(rel)
        elif digest_file(path, algo) != expected:
            changed.append(rel)
    if root is not None:
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                rel = os.path.relpath(os.path.join(dirpath, name), root)
                if rel not in baseline:
                    added.append(rel)
    return {"ok": not (changed or missing or added),
            "changed": changed, "missing": missing, "added": added}


def generate_password(length=20, symbols=True, rng=None):
    if length < 8:
        raise ValueError("length must be at least 8")
    alphabet = ("abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    if symbols:
        alphabet += "!@#$%^&*()-_=+[]{};:,.?"
    rng = rng or secrets.SystemRandom()
    return "".join(rng.choice(alphabet) for _ in range(length))


def generate_passphrase(words=6, separators=True, wordlist=None):
    if wordlist is None:
        base = os.path.join(os.path.dirname(__file__), "wordlist.txt")
        with open(base, encoding="utf-8") as f:
            wordlist = [w for line in f for w in line.split() if w]
    rng = secrets.SystemRandom()
    picked = [rng.choice(wordlist) for _ in range(words)]
    return "-".join(picked) if separators else " ".join(picked)
