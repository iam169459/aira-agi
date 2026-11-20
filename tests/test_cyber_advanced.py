#!/usr/bin/env python
"""Tests for the advanced cyber_tools: crack/wall/probe/repair/sentinel/engage.

Run: python tests/test_cyber_advanced.py
or:  pytest tests/test_cyber_advanced.py

Active-probing tests run against throwaway loopback servers only; every
public-target case verifies the authorization guard refuses first. No
DoS/flooding capability exists or is tested.
"""
import http.server
import hashlib
import os
import socket
import sys
import tempfile
import threading
import time
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.cyber_tools import (Sentinel, audit_credentials, crack_stored,
                                 injection_probe, listening_ports,
                                 load_wordlist, mutations, plan_repairs,
                                 run_engagement, secure_permissions,
                                 tls_version_check, wall_integrity)

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


def test_crack():
    print("\n[1] Offline credential auditing")
    md5_of = lambda s: hashlib.md5(s.encode()).hexdigest()
    r = crack_stored(md5_of("rockyou"), words=["secret", "rockyou", "a"],
                     rules_level=1)
    check("md5 cracked", r["cracked"] and r["password"] == "rockyou",
          str(r))
    r = crack_stored(md5_of("Rockyou2024!"), words=["rockyou"],
                     rules_level=2)
    check("rule mutations crack variants",
          r["cracked"] and r["password"] == "Rockyou2024!", str(r))
    r = crack_stored(md5_of("unbreakable-phrase-xyz"),
                     words=["rockyou", "letmein"], rules_level=2)
    check("strong hash resistant", not r["cracked"])
    salt = "abc12345"
    salted = salt + "$" + hashlib.pbkdf2_hmac(
        "sha256", b"Passw0rd!", salt.encode(), 5000).hex()
    r = crack_stored(salted, words=["passw0rd!", "passw0rd"], algo="sha256",
                     pbkdf2_iterations=5000)
    check("salted PBKDF2 cracked", r["cracked"]
          and r["password"].lower() == "passw0rd!", str(r))
    db = {"alice": md5_of("hunter2"), "bob": md5_of("iloveyou2024"),
          "carol": md5_of("Xq7#vLm2!pR9")}
    audit = audit_credentials(db, words=["hunter2", "iloveyou"],
                              rules_level=2)
    check("audit cracked weak users",
          set(audit["cracked"]) == {"alice", "bob"}, str(audit["cracked"]))
    check("audit marks resistant user", audit["resistant"] == ["carol"])
    check("mutations deterministic",
          len(list(mutations("abc", level=1))) == 10)
    bundled = load_wordlist(os.path.join(
        os.path.dirname(__file__), "..", "modules", "cyber_tools",
        "wordlist.txt"))
    check("bundled wordlist loads", len(bundled) >= 100)


def test_wall():
    print("\n[2] Wall integrity: crack detection")
    try:
        wall_integrity("8.8.8.8", ports=[80])
        check("wall guard blocks public IPs", False, "no exception")
    except PermissionError:
        check("wall guard blocks public IPs", True)
    risky_port = _free_port()
    risky_port = risky_port if risky_port > 1023 else risky_port + 2000
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 5900))  # VNC-class "risky" service
    server.listen(1)
    clean_port = _free_port()
    try:
        report = wall_integrity("127.0.0.1", ports=[5900, clean_port],
                                timeout=0.3, skip_tls=True)
        check("wall score returned",
              isinstance(report["wall_score"], int)
              and 0 <= report["wall_score"] <= 100)
        kinds = {c["kind"] for c in report["cracks"]}
        check("risky service detected as crack",
              "dangerous-service" in kinds and 5900 in report["open"],
              str(report["cracks"]))
        server.close()
        report2 = wall_integrity("127.0.0.1", ports=[5900, clean_port],
                                 timeout=0.3, skip_tls=True)
        check("clean wall has no cracks", report2["cracks"] == [])
        check("clean wall scores high", report2["wall_score"] >= 90)
    finally:
        server.close()
        _ = risky_port
    try:
        tls_version_check("8.8.8.8", 443)
        check("tls check guard blocks public", False, "no exception")
    except PermissionError:
        check("tls check guard blocks public", True)


def test_webprobe():
    print("\n[3] Injection probes vs a loopback app")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            parsed = urllib.parse.parse_qs(query, keep_blank_values=True)
            target = parsed.get("user", [""])[0]
            echo = f"<div>Hi, {target}!</div>"
            body = echo.encode()
            if "'" in target:
                body += (b"<br>Error: you have an error in your SQL syntax; "
                         b"check the manual near '''")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        report = injection_probe(f"http://127.0.0.1:{port}/",
                                 params=["user"], timeout=2)
        find_kinds = {(f["param"], f["kind"]) for f in report["findings"]}
        check("sql error signature detected",
              ("user", "sql-error") in find_kinds,
              str(report["findings"]))
        check("reflected xss detected", ("user", "xss") in find_kinds)
        check("vulnerable flag set", report["vulnerable"] is True)
        plan = plan_repairs(probe_report=report)
        check("repair plan generated for probes",
              any("Parameterize SQL" in p["title"] for p in plan))
    finally:
        server.shutdown()
        server.server_close()


def test_repair():
    print("\n[4] Repair engine")
    plan = plan_repairs(wall_report={
        "cracks": [
            {"kind": "dangerous-service",
             "detail": "port 5900 open: vnc exposed — weak auth",
             "severity": "high"},
            {"kind": "weak-tls",
             "detail": "port 443 accepts TLSv1.1 (downgrade risk)",
             "severity": "high"},
        ]}, credential_report={"cracked": {"root": "hunter2"}},
        probe_report={"findings": [{"param": "id", "kind": "sql-error"}]})
    titles = [p["title"] for p in plan]
    check("port-closure repair planned",
          any("port 5900" in t for t in titles), str(titles))
    check("tls repair planned", any("Disable TLSv1.1" in t for t in titles))
    check("credential rotation planned",
          any("'root'" in t for t in titles))
    check("sql repair planned", any("Parameterize SQL" in t for t in titles))
    check("plan items have ids",
          all(p["id"].startswith("R-") for p in plan) and len(plan) >= 4)
    with tempfile.TemporaryDirectory() as root:
        weak = os.path.join(root, "secrets.env")
        open(weak, "w").write("KEY=1")
        os.chmod(weak, 0o666)
        preview = secure_permissions([weak], confirm=False)
        check("dry-run does not apply", preview["applied"] is False
              and os.stat(weak).st_mode & 0o022)
        applied = secure_permissions([weak], confirm=True)
        check("world-writable bit removed",
              applied["applied"] and not os.stat(weak).st_mode & 0o022)
        check("change recorded with modes",
              applied["changed"][0]["to"] == "-rw-r--r--")


def test_sentinel():
    print("\n[5] Sentinel: live crack detection")
    with tempfile.TemporaryDirectory() as root:
        f1 = os.path.join(root, "critical.cfg")
        open(f1, "w").write("original")
        baseline = listening_ports()
        sentinel = Sentinel(watch_paths=[root], port_baseline=baseline,
                            brute_threshold=3)
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        rogue_port = listener.getsockname()[1]
        breach = sentinel.check()
        check("rogue listener detected", rogue_port in breach["new_ports"])
        open(f1, "w").write("tampered by attacker")
        log = os.path.join(root, "auth.log")
        with open(log, "w") as f:
            for i in range(5):
                f.write("Failed password for admin from 203.0.113.7 "
                        f"port {i} ssh2\n")
        sentinel.auth_log = log
        breach = sentinel.check()
        check("file tamper detected", "critical.cfg" in breach["files"])
        check("rogue port still flagged", rogue_port in breach["new_ports"])
        check("brute force detected",
              breach["brute_force"] == ["203.0.113.7"], str(breach))
        check("breach not clean", breach["clean"] is False)
        sentinel.rescan_files()
        sentinel.port_baseline = listening_ports()
        breach = sentinel.check()
        check("files re-baselined", breach["files"] == [] and
              breach["new_ports"] == [])
        ports = listening_ports()
        check("listening_ports returns a set of ints",
              isinstance(ports, set)
              and all(isinstance(p, int) for p in ports))


def test_engagement():
    print("\n[6] Engagement console: attack + repair together")
    try:
        run_engagement("8.8.8.8")
        check("engagement guard blocks public", False, "no exception")
    except PermissionError:
        check("engagement guard blocks public", True)
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 3389))  # RDP-class risky port
    server.listen(1)
    try:
        report = run_engagement("127.0.0.1", timeout=0.3)
        check("engagement ran stages",
              {"recon", "ports", "wall", "plan", "summary"} <= set(report))
        check("risky-port crack found",
              any("3389" in c["detail"] for c in report["cracks"]))
        check("repair plan proposed", len(report["plan"]) >= 1)
        check("summary composed", report["summary"].count("crack") >= 1)
    finally:
        server.close()


def main():
    test_crack()
    test_wall()
    test_webprobe()
    test_repair()
    test_sentinel()
    test_engagement()
    print(f"\n{'='*60}\nResults: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name, detail in FAIL:
            print(f"  FAILED: {name} {detail}")
        sys.exit(1)
    print("All advanced cyber_tools tests passed.")


if __name__ == "__main__":
    main()
