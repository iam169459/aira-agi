"""Symmetric encryption built only on stdlib primitives.

Construction: SHA-256 in counter mode as the keystream, encrypt-then-MAC
with HMAC-SHA256. Format: salt(16) || nonce(8) || ciphertext || tag(32).
Not a replacement for a vetted TLS/AEAD library, but a sound from-first-
principles design using primitives reviewed by the stdlib.
"""
import hashlib
import hmac
import os

SALT_SIZE = 16
NONCE_SIZE = 8
KEY_SIZE = 32
BLOCK = 32
TAG_SIZE = 32
_PBKDF2_ITERATIONS = 200_000


def derive_key(password, salt=None, iterations=_PBKDF2_ITERATIONS):
    """Derive a 32-byte key from a password via PBKDF2-HMAC-SHA256."""
    if isinstance(password, str):
        password = password.encode("utf-8")
    if salt is None:
        salt = os.urandom(SALT_SIZE)
    key = hashlib.pbkdf2_hmac("sha256", password, salt, iterations)
    return key, salt, iterations


def _bytes_to_int(b):
    return int.from_bytes(b, "big")


def _keystream(key, nonce, length):
    out = bytearray()
    counter = 0
    while len(out) < length:
        block = hashlib.sha256(
            key + _bytes_to_int(nonce).to_bytes(NONCE_SIZE, "big")
            + counter.to_bytes(8, "big")
        ).digest()
        out.extend(block)
        counter += 1
    return bytes(out[:length])


def _subkeys(master):
    enc = hashlib.sha256(b"cyber-tools-enc" + master).digest()
    mac = hashlib.sha256(b"cyber-tools-mac" + master).digest()
    return enc, mac


def encrypt(key, plaintext):
    if isinstance(plaintext, str):
        plaintext = plaintext.encode("utf-8")
    enc_key, mac_key = _subkeys(key)
    nonce = os.urandom(NONCE_SIZE)
    stream = _keystream(enc_key, nonce, len(plaintext))
    ciphertext = bytes(a ^ b for a, b in zip(plaintext, stream))
    tag = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()
    return nonce + ciphertext + tag


def decrypt(key, blob):
    if len(blob) < NONCE_SIZE + TAG_SIZE:
        raise ValueError("ciphertext too short")
    enc_key, mac_key = _subkeys(key)
    nonce, body = blob[:NONCE_SIZE], blob[NONCE_SIZE:-TAG_SIZE]
    tag = blob[-TAG_SIZE:]
    expected = hmac.new(mac_key, nonce + body, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expected):
        raise ValueError("authentication failed: ciphertext was tampered with")
    stream = _keystream(enc_key, nonce, len(body))
    return bytes(a ^ b for a, b in zip(body, stream))
