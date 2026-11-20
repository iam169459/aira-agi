"""Credential auditing: dictionary/rules hash cracking for authorized targets.

This audits *hashes you have legitimate access to* (your own box, an
engagement you are contracted for). It never contacts a network target.
"""
import hashlib
import os
import time

_BUNDLED_WORDLIST = os.path.join(os.path.dirname(__file__), "wordlist.txt")

_LEET = {"a": "@", "e": "3", "i": "1", "o": "0", "s": "$", "t": "7"}


def load_wordlist(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return [word for line in f for word in line.split() if word]


def mutations(word, level=1):
    """Deterministic candidate generator from one base word."""
    yield word
    yield word.capitalize()
    yield word.upper()
    yield word[::-1]
    if level >= 1:
        yield word + "1"
        yield word + "123"
        yield word + "!"
        for year in ("2024", "2023", "2025"):
            yield word + year
    if level >= 2:
        leet = "".join(_LEET.get(c, c) for c in word)
        yield leet
        for suffix in ("1234", "#", "@@", "20", "007"):
            yield word + suffix
        for year in ("2023", "2024", "2025"):
            yield word.capitalize() + year
            yield word.capitalize() + year + "!"
        yield word.capitalize() + "1"
        yield leet.capitalize()
        yield leet.capitalize() + "!"


def _hash_candidate(algo, password):
    return hashlib.new(algo, password.encode("utf-8")).hexdigest()


def _parse_stored(value):
    """Accept 'hexdigest' or 'salt$hexdigest' (PBKDF2 salted form)."""
    value = value.strip()
    if "$" in value:
        salt, digest = value.split("$", 1)
        return salt, digest
    return None, value


_ALGO_LENGTHS = {32: "md5", 40: "sha1", 64: "sha256", 128: "sha512"}


def _guess_algo(digest):
    return _ALGO_LENGTHS.get(len(digest), "sha256")


def _pbkdf2_hex(password, salt, iterations):
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations
    ).hex()


def crack_stored(stored, algo=None, words=None, rules_level=1,
                 pbkdf2_iterations=0):
    """Try to recover the plaintext of one stored hash.

    `stored` is a hex digest, or `salt$hexdigest` when
    `pbkdf2_iterations` > 0. Returns
    {'cracked', 'password', 'attempts', 'seconds'}.
    """
    if words is None:
        words = load_wordlist(_BUNDLED_WORDLIST)
    salt, digest = _parse_stored(stored)
    if algo is None:
        algo = _guess_algo(digest)
    if pbkdf2_iterations and algo != "sha256":
        raise ValueError("salted PBKDF2 form requires algo='sha256'")
    start = time.perf_counter()
    attempts = 0
    seen = set()
    for word in words or []:
        for candidate in mutations(word, rules_level):
            if candidate in seen:
                continue
            seen.add(candidate)
            attempts += 1
            if pbkdf2_iterations:
                candidate_hex = _pbkdf2_hex(
                    candidate, salt if salt else "", pbkdf2_iterations)
            else:
                candidate_hex = _hash_candidate(algo, candidate)
            if candidate_hex == digest:
                return {"cracked": True, "password": candidate,
                        "attempts": attempts,
                        "seconds": round(time.perf_counter() - start, 4)}
    return {"cracked": False, "password": None, "attempts": attempts,
            "seconds": round(time.perf_counter() - start, 4)}


def audit_credentials(database, words, rules_level=1, pbkdf2_iterations=0):
    """Audit a {user: stored_hash} mapping.

    Use only on hashes you legitimately possess. Returns
    {'users', 'cracked': {user: password}, 'resistant': [user],
    'attempts'}.
    """
    cracked, resistant, attempts = {}, [], 0
    for user, stored in database.items():
        result = crack_stored(stored, words=words,
                              rules_level=rules_level,
                              pbkdf2_iterations=pbkdf2_iterations)
        attempts += result["attempts"]
        if result["cracked"]:
            cracked[user] = result["password"]
        else:
            resistant.append(user)
    return {"users": len(database), "cracked": cracked,
            "resistant": resistant, "attempts": attempts}
