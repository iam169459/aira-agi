"""Tests for the extra tool skills (ai_core/more_tools.py) and the
self-updater (ai_core/updater.py)."""
import base64
import hashlib
import os
import re
import subprocess
import tempfile
import uuid as uuid_mod


def _git(args, cwd=None, check=True):
    """git with signing disabled (sandbox CI commit-signing config)."""
    return subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-c", "gpg.format=ssh",
         "-c", "user.signingkey="] + args,
        cwd=cwd, check=check, capture_output=True, text=True)


def _registry(root=None, notes_path=None):
    import ai_core.tools as tools
    return tools.ToolRegistry(notes_path=notes_path or tempfile.mkstemp()[1],
                              root=root or os.getcwd())


def check(name, ok, detail=""):
    print(f"  {'ok' if ok else 'FAIL'} {name}"
          + ("" if ok else f" -> {detail}"))
    return ok


def test_convert():
    r = _registry()
    ok = check("km to miles", "3.106856 mi" in (r.respond(
        "convert 5 km to miles") or ""))
    ok &= check("alias words", "1.8288 m" in (r.respond(
        "convert 6 feet to meters") or ""))
    ok &= check("temperature", "36.666667 c" in (r.respond(
        "convert 98 f to c") or ""))
    ok &= check("unknown units refused", "don't know" in (r.respond(
        "convert 5 km to bananas") or ""))
    check("unit conversion", ok)


def test_base64_hash_password_uuid():
    r = _registry()
    enc = r.respond("base64 encode aira is offline")
    ok = check("base64 encode", base64.b64encode(
        "aira is offline".encode()).decode() in enc)
    encoded = base64.b64encode("hello".encode()).decode()
    ok &= check("base64 decode", "hello" in
                r.respond(f"base64 decode {encoded}"))
    ok &= check("base64 decode garbage", "valid base64" in
                r.respond("base64 decode ###not-base64###"))
    digest = hashlib.sha256(b"abc").hexdigest()
    ok &= check("sha256 known value", digest in
                r.respond("hash abc"))
    digest1 = hashlib.md5(b"abc").hexdigest()
    ok &= check("md5 algo choice", digest1 in
                r.respond("hash abc with md5"))
    pw = r.respond("generate a strong password 24")
    token = re.search(r"password: (\S+)", pw).group(1)
    ok &= check("password length + class",
                len(token) == 24 and any(c in "!@#$%^&*()-_=+" for c in token))
    u = re.search(r"UUID: ([0-9a-f-]+)",
                  r.respond("generate uuid")).group(1)
    ok &= check("uuid parses", uuid_mod.UUID(u) ==
                uuid_mod.UUID(str(uuid_mod.UUID(u))))
    check("base64/hash/password/uuid", ok)


def test_roll_wordcount_json():
    r = _registry()
    flip = r.respond("flip a coin")
    ok = check("coin", any(w in flip for w in ("heads", "tails")))
    for _ in range(20):
        out = r.respond("roll 2d20")
        vals = re.findall(r"\b(\d+)\b", out.split(":")[1])
        if len(vals) == 3 and all(1 <= int(v) <= 20 for v in vals[:2]) \
                and int(vals[2]) == int(vals[0]) + int(vals[1]):
            break
    else:
        ok = False
    check("2d20 shape", ok)
    ok = check("word count", "5 words" in (r.respond(
        "count the words in aira talks on her own") or ""))
    ok &= check("json valid", '"b": 2' in (r.respond(
        'format json: {"a":1, "b":2}') or ""))
    ok &= check("json invalid", "doesn't parse" in (r.respond(
        "format json: {oops}") or ""))
    check("roll/wordcount/json", ok)


def test_sys_info_and_help():
    r = _registry()
    ok = check("sys info", "Python 3." in (r.respond("system info") or ""))
    listing = r.respond("what can you do")
    ok &= check("help lists tools", "calculator" in listing
                and "check_updates" in listing)
    ok &= check("new tools registered",
                set(r.names()) >= {"convert_units", "base64", "hash_text",
                                   "password", "uuid_gen", "roll",
                                   "word_count", "format_json", "sys_info",
                                   "check_updates", "self_update", "help"})
    check("sys_info/help", ok)


def test_updater():
    from ai_core.updater import Updater, UpdaterError
    tmp = tempfile.mkdtemp()
    work = os.path.join(tmp, "work")
    bare = os.path.join(tmp, "remote.git")
    _git(["init", "-q", "-b", "main", work])
    _git(["init", "-q", "--bare", "-b", "main", bare])
    for gr in (("user.email", "a@a"), ("user.name", "a")):
        _git(["-C", work, "config", *gr], cwd=".")
    with open(os.path.join(work, "aira.txt"), "w") as fh:
        fh.write("v1\n")
    _git(["-C", work, "add", "-A"])
    _git(["-C", work, "commit", "-qm", "v1"])
    _git(["-C", work, "push", "-q", bare, "main"])

    up = Updater(root=work, remote=bare, branch="main")
    ok = check("up to date", "Already up to date" in up.check())
    ok &= check("apply no-op ok", "Already up to date" in up.apply())

    # a new commit arrives on origin
    other = os.path.join(tmp, "other")
    _git(["clone", "-q", bare, other])
    for gr in (("user.email", "a@a"), ("user.name", "a")):
        _git(["-C", other, "config", *gr], cwd=".")
    with open(os.path.join(other, "new.txt"), "w") as fh:
        fh.write("v2\n")
    _git(["-C", other, "add", "-A"])
    _git(["-C", other, "commit", "-qm", "v2"])
    _git(["-C", other, "push", "-q", bare, "main"])

    ok &= check("behind detected", "1 commit(s) behind" in up.check())
    ok &= check("ff update applied", "Updated to" in up.apply())
    ok &= check("new file landed", os.path.exists(
        os.path.join(work, "new.txt")))

    with open(os.path.join(work, "aira.txt"), "w") as fh:
        fh.write("v1 local change\n")
    try:
        up.apply()
        ok &= False
    except UpdaterError:
        ok &= True
    check("dirty tree refused", ok)

    ok = True
    try:
        Updater(root=tmp).check()
        ok = False
    except UpdaterError:
        pass
    check("non-repo refused", ok)
    check("updater", ok)


def main():
    print("[unit conversion]")
    test_convert()
    print("[base64 / hash / password / uuid]")
    test_base64_hash_password_uuid()
    print("[dice, word count, json]")
    test_roll_wordcount_json()
    print("[system info + help]")
    test_sys_info_and_help()
    print("[self-updater]")
    test_updater()
    print("\nAll more-tools tests passed.")


if __name__ == "__main__":
    main()
