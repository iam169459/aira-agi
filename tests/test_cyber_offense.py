#!/usr/bin/env python
"""Tests for the consent-gated cyber-offense pack (modules.cyber_tools.offense)

Run: python tests/test_cyber_offense.py
or:  pytest tests/test_cyber_offense.py

The offense pack is STRICTER than the defensive toolkit: loopback does
NOT bypass the gate. Active-phase tests run against a throwaway loopback
HTTP server only, with authorized=True; every gate case verifies refusal.
"""
import http.server
import os
import socket
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.cyber_tools import (ConsentRequired, ConsentStore,
                                 DEFAULT_SUBDOMAIN_WORDS, KILL_CHAIN_PHASES,
                                 attack_surface, kill_chain, recon_sweep,
                                 run_offense, strict_gate, web_probe_suite)

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


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/admin"):
            body = b"forbidden"
            self.send_response(403)
        elif self.path.startswith("/backup"):
            body = b"secret-backup-file"
            self.send_response(200)
        elif self.path.startswith("/?q="):
            body = (b"you have an error in your sql syntax near ''")
            self.send_response(200)
        else:
            body = b"<html><body>ok</body></html>"
            self.send_response(200)
            self.send_header("Server", "aira-test/1.0")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_strict_gate():
    print("\n[1] Strict consent gate (no loopback bypass)")
    try:
        strict_gate("127.0.0.1", False, "unit-test")
        check("loopback refused without consent", False, "no exception")
    except ConsentRequired:
        check("loopback refused without consent", True)
    except PermissionError:
        check("loopback refused without consent", True,
              "PermissionError raised instead of ConsentRequired")

    try:
        strict_gate("example.com", False, "unit-test")
        check("public host refused without consent", False, "no exception")
    except ConsentRequired:
        check("public host refused without consent", True)

    check("authorized=True passes gate",
          strict_gate("example.com", True, "unit-test") is None)


def test_consent_profile_path():
    print("\n[2] Consent profile authorizes a profile-covered target")
    with tempfile.TemporaryDirectory() as tmp:
        store = ConsentStore(os.path.join(tmp, "consent.json"))
        store.grant("unittest", ["127.0.0.1"], scope="full", ttl_days=1)
        check("profile covers loopback", store.covered("unittest",
                                                       "127.0.0.1"))
        check("profile clears the strict gate",
              strict_gate("127.0.0.1", False, "unit-test",
                          store=store, profile="unittest") is None)
        empty = ConsentStore(os.path.join(tmp, "none.json"))
        try:
            strict_gate("127.0.0.1", False, "unit-test", store=empty,
                        profile="missing")
            check("missing profile refused", False, "no exception")
        except ConsentRequired:
            check("missing profile refused", True)


def test_phases_on_loopback():
    print("\n[3] Active phases against a throwaway loopback server")
    port = _free_port()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        target = f"127.0.0.1"
        surface = attack_surface(target, authorized=True,
                                 ports=(port,), timeout=0.3)
        check("surface phase opens port", port in surface["open"],
              str(surface["ports"]))
        check("surface reports services",
              isinstance(surface["banners"], dict)
              and "services" in surface["banners"],
              str(surface["banners"])[:80])

        probe = web_probe_suite(target, authorized=True, http_port=port,
                                timeout=2.0)
        check("probe phase fingerprinted server",
              (probe["fingerprint"] or {}).get("ok") is True,
              str(probe["fingerprint"]))
        check("probe injection result present (dict)",
              isinstance(probe["injection"], dict) or
              probe["injection"] is None)

        try:
            attack_surface("192.0.2.99", ports=(port,), timeout=0.3)
            check("phase refuses without consent", False, "no exception")
        except ConsentRequired:
            check("phase refuses without consent", True)
    finally:
        server.shutdown()


def test_run_offense_and_chain():
    print("\n[4] Full offense run + kill-chain narrative")
    port = _free_port()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        report = run_offense("127.0.0.1", authorized=True,
                             phases=("recon", "surface", "probe"),
                             timeout=0.3)
        check("offense run has phase map",
              set(report["phases"]) == {"recon", "surface", "probe"},
              str(set(report["phases"])))
        check("findings normalized (list of dicts with severity)",
              isinstance(report["findings"], list) and all(
                  isinstance(f, dict) and "severity" in f
                  for f in report["findings"]))
        check("counts aggregate severities",
              isinstance(report["counts"], dict))
        chain = kill_chain(report)
        check("kill chain mentions surface", "SURFACE" in chain, chain)
        check("kill chain ends in posture",
              "POSTURE" in chain, chain[:200])
        prof = run_offense("127.0.0.1", store=_tmp_store(), profile="u2",
                           phases=("surface",), timeout=0.3)
        check("consent-profile run works (no authorized flag)",
              prof["target"] == "127.0.0.1"
              and "surface" in prof["phases"])
        check("subdomain wordlist constant sane",
              "www" in DEFAULT_SUBDOMAIN_WORDS)
        check("phase constants match run order",
              KILL_CHAIN_PHASES == ("recon", "surface", "probe",
                                    "posture"))
    finally:
        server.shutdown()


_STORE = {}


def _tmp_store():
    import tempfile
    if "store" not in _STORE:
        st = ConsentStore(os.path.join(tempfile.mkdtemp(),
                                       "consent.json"))
        st.grant("u2", ["127.0.0.1"], scope="full", ttl_days=1)
        _STORE["store"] = st
    return _STORE["store"]


def main():
    test_strict_gate()
    test_consent_profile_path()
    test_phases_on_loopback()
    test_run_offense_and_chain()
    print(f"\n{'='*60}\nResults: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name, detail in FAIL:
            print(f"  FAILED: {name} {detail}")
        sys.exit(1)
    print("All cyber_offense tests passed.")


if __name__ == "__main__":
    main()
