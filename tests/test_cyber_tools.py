#!/usr/bin/env python
"""Tests for the cybersecurity toolkit (defensive + authorized-target tools).

Run: python tests/test_cyber_tools.py
or:  pytest tests/test_cyber_tools.py

Network-facing checks are loopback-only; every external-target case
verifies that the authorization guard refuses to run.
"""
import http.server
import os
import socket
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.cyber_tools import (audit_headers, baseline_build,
                                 baseline_verify, brute_force_hits,
                                 decrypt, derive_key, digest_bytes,
                                 digest_file, encrypt, extract_iocs,
                                 generate_passphrase, generate_password,
                                 guard, hash_identifier, hmac_digest,
                                 password_strength, scan_ports, tls_expiry)

PASS = []
FAIL = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} {detail}")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_hashes():
    print("\n[1] Digests and known-answer vectors")
    check("sha256 known vector",
          digest_bytes("abc") ==
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    check("md5 known vector",
          digest_bytes("abc", "md5") == "900150983cd24fb0d6963f7d28e17f72")
    check("hmac known vector",
          hmac_digest(b"key",
                      "The quick brown fox jumps over the lazy dog") ==
          "f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("verify me")
        path = f.name
    check("file digest matches bytes",
          digest_file(path) == digest_bytes("verify me"))
    os.unlink(path)


def test_hash_identifier():
    print("\n[2] Hash identification")
    r = hash_identifier("ba7816bf8f01cfea414140de5dae2223"
                        "b00361a396177a9cb410ff61f20015ad")
    check("sha256 detected", r["best_guess"] == "sha256")
    r = hash_identifier("$2b$12$abcdefghijklmnopqrstuv")
    check("bcrypt detected", r["best_guess"] == "bcrypt")
    r = hash_identifier("900150983cd24fb0d6963f7d28e17f72")
    check("md5 detected", r["best_guess"] == "md5")
    r = hash_identifier("not a hash at all!")
    check("non-hash rejected", r["best_guess"] is None)


def test_password_strength():
    print("\n[3] Password strength scoring")
    weak = password_strength("password123")
    check("common password scored low", weak["score"] <= 1,
          f"score {weak['score']}")
    check("pattern flagged", any("common password" in p
                                 for p in weak["patterns"]))
    strong = password_strength("u9#Vgq2!LmZt$7@Rk4Px")
    check("random password scored high", strong["score"] >= 4,
          f"score {strong['score']}")
    check("password has no patterns", strong["patterns"] == [])
    check("empty password handled", password_strength("")["score"] == 0)
    pw = generate_password(24)
    check("generated password length", len(pw) == 24)
    check("generated password strong",
          password_strength(pw)["score"] >= 3)
    phrase = generate_passphrase(5, wordlist=["alpha", "beta", "gamma"])
    check("passphrase words", len(phrase.split("-")) == 5
          and all(w in ("alpha", "beta", "gamma") for w in phrase.split("-")))


def test_baseline_integrity():
    print("\n[4] File integrity baseline")
    with tempfile.TemporaryDirectory() as root:
        a = os.path.join(root, "a.txt")
        b = os.path.join(root, "sub", "b.txt")
        os.makedirs(os.path.dirname(b))
        open(a, "w").write("hello")
        open(b, "w").write("world")
        baseline = baseline_build([root])
        ok = baseline_verify(baseline, root)
        check("clean baseline verifies", ok["ok"] and not ok["changed"])
        open(a, "w").write("tampered")
        open(os.path.join(root, "c.txt"), "w").write("new")
        os.unlink(b)
        diff = baseline_verify(baseline, root)
        check("modification detected", "a.txt" in diff["changed"])
        check("deletion detected", "sub/b.txt" in diff["missing"])
        check("addition detected", "c.txt" in diff["added"])
        check("diff not ok", diff["ok"] is False)


def test_crypto():
    print("\n[5] Encryption roundtrip and tamper detection")
    key, salt, iterations = derive_key("hunter2 but longer")
    msg = b"attack at dawn" * 100
    blob = encrypt(key, msg)
    check("decrypt roundtrip", decrypt(key, blob) == msg)
    check("key derivation salted", len(salt) == 16 and iterations == 200000)
    key2, _, _ = derive_key("hunter2 but longer", salt)
    check("derive_key deterministic with salt", key2 == key)
    key3, _, _ = derive_key("wrong password", salt)
    try:
        decrypt(key3, blob)
        check("wrong key rejected", False, "no exception")
    except ValueError:
        check("wrong key rejected", True)
    tampered = blob[:-1] + bytes([blob[-1] ^ 0x01])
    try:
        decrypt(key, tampered)
        check("tamper rejected", False, "no exception")
    except ValueError:
        check("tamper rejected", True)
    blob2 = encrypt(key, msg)
    check("unique nonce per encryption", blob != blob2)
    check("plaintext not recoverable by inspection", msg[:4] not in blob[:64])


def test_portscan_guard():
    print("\n[6] Port scanner: guard + loopback scan")
    try:
        scan_ports("8.8.8.8", ports=[80], timeout=0.2)
        check("guard blocks public IPs", False, "no exception")
    except PermissionError:
        check("guard blocks public IPs", True)
    open_port, closed_port = _free_port(), _free_port()
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", open_port))
    server.listen(1)
    try:
        results = scan_ports("127.0.0.1", ports=[open_port, closed_port],
                             timeout=0.4, grab_banner=False)
        states = {r["port"]: r["state"] for r in results}
        check("open port detected", states.get(open_port) == "open")
        check("closed port detected", states.get(closed_port) == "closed")
    finally:
        server.close()


def test_logscan():
    print("\n[7] Log analysis: IOCs and brute force")
    sample = [
        "Jan 12 03:41:22 srv sshd[902]: Failed password for admin from "
        "203.0.113.5 port 41022 ssh2",
        "Jan 12 03:41:24 srv sshd[903]: Failed password for invalid user "
        "oracle from 203.0.113.5 port 41026 ssh2",
        "Jan 12 03:41:26 srv sshd[904]: Failed password for root from "
        "198.51.100.9 port 55012 ssh2",
        "Jan 12 03:42:01 srv sshd[905]: Invalid user test from 203.0.113.5",
        "contact admin@example.org, see https://alerts.example.net/id/9 "
        "md5 44d88612fea8a8f36de82e1278abb02f",
    ]
    hits = brute_force_hits(sample, threshold=2)
    counts = {e["ip"]: e["count"] for e in hits["by_source"]}
    check("failed-auth total", hits["total"] == 4, str(counts))
    check("per-source counts", counts.get("203.0.113.5") == 3
          and counts.get("198.51.100.9") == 1)
    check("suspect flagged by threshold", hits["suspects"] == ["203.0.113.5"])
    iocs = extract_iocs("\n".join(sample))
    check("ioc ips", set(iocs["ips"]) == {"203.0.113.5", "198.51.100.9"})
    check("ioc email", iocs["emails"] == ["admin@example.org"])
    check("ioc url", iocs["urls"] == ["https://alerts.example.net/id/9"])
    check("ioc hash", iocs["hashes"] == ["44d88612fea8a8f36de82e1278abb02f"])
    iocs2 = extract_iocs("from 127.0.0.1 and 10.0.0.3", drop_private_ips=True)
    check("loopback/private dropped",
          iocs2["ips"] == ["10.0.0.3"] or iocs2["ips"] == [])


def test_webaudit():
    print("\n[8] Web header audit on a local server")
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Strict-Transport-Security",
                             "max-age=31536000")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy",
                             "strict-origin-when-cross-origin")
            if self.path == "/cookie":
                self.send_header("Set-Cookie", "sid=1; Path=/")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        report = audit_headers(f"http://127.0.0.1:{port}/",
                               authorized=False)
        check("audit reached local server", report["ok"])
        check("present headers counted",
              report["present"].get("strict-transport-security") is not None)
        check("missing headers flagged",
              "content-security-policy" in report["missing"])
        check("score from present headers", report["score"] == 3)
        report = audit_headers(f"http://127.0.0.1:{port}/cookie")
        check("insecure cookie flagged",
              any("HttpOnly" in f for f in report["findings"]))
        try:
            tls_expiry("http://127.0.0.1")
            check("tls_expiry rejects http", False, "no exception")
        except ValueError:
            check("tls_expiry rejects http", True)
    finally:
        server.shutdown()
        server.server_close()


def test_recon_guard():
    print("\n[9] Recon tools are authorization-gated")
    try:
        guard("8.8.8.8", False, "test")
        check("guard blocks public IP", False, "no exception")
    except PermissionError:
        check("guard blocks public IP", True)
    check("guard allows loopback", guard("127.0.0.1", False, "test") is None)
    check("guard allows authorized", guard("any.host", True, "test") is None)


def main():
    test_hashes()
    test_hash_identifier()
    test_password_strength()
    test_baseline_integrity()
    test_crypto()
    test_portscan_guard()
    test_logscan()
    test_webaudit()
    test_recon_guard()
    print(f"\n{'='*60}\nResults: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name, detail in FAIL:
            print(f"  FAILED: {name} {detail}")
        sys.exit(1)
    print("All cyber_tools tests passed.")


if __name__ == "__main__":
    main()
