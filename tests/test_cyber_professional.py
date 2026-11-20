#!/usr/bin/env python
"""Tests for the professional pentest layer: fingerprint, banner, ciphers, reports.

Run: python tests/test_cyber_professional.py
or:  pytest tests/test_cyber_professional.py

Active probe tests run against throwaway loopback servers only; every
public-target case verifies the authorization guard refuses first.
"""
import http.server
import os
import socket
import subprocess
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.cyber_tools import (cipher_audit, build_report, dir_probe,
                                 fingerprint_http, render_html,
                                 render_markdown, run_engagement,
                                 save_report, service_banner)
from modules.cyber_tools.reporting import collect_findings
from modules.cyber_tools.clitool import _severity_counts

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


def test_fingerprint():
    print("\n[1] Web stack fingerprinting")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = (b"<html><head><meta name='generator' "
                    b"content='WordPress 5.8'></head>"
                    b"<body><div>wp-content/themes</div></body></html>")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Server", "nginx/1.14.0")
            self.send_header("X-Powered-By", "PHP/7.2.1")
            self.send_header("Set-Cookie", "PHPSESSID=abc123; Path=/")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        try:
            fingerprint_http("http://8.8.8.8/")
            check("fingerprint guard blocks public", False, "no exception")
        except PermissionError:
            check("fingerprint guard blocks public", True)
        result = fingerprint_http(f"http://127.0.0.1:{port}/",
                                  timeout=3.0, authorized=True)
        check("fingerprint ok", result["ok"] and result["status"] == 200,
              str(result))
        check("server header captured",
              result["server"] == "nginx/1.14.0")
        check("tech includes WordPress",
              "WordPress" in result["tech"], str(result["tech"]))
        check("tech includes PHP via powered-by",
              "PHP" in result["tech"], str(result["tech"]))
        check("cookie hint PHPSESSID",
              "PHPSESSID" in result["cookies"])
        check("generator meta extracted",
              result["generator"] == "WordPress 5.8")
        engagement = {"target": f"127.0.0.1:{port}", "wall": {},
                      "http_fingerprint": result}
        findings = collect_findings(engagement)
        outdated = [f for f in findings if "Outdated" in f["title"]]
        check("outdated nginx flagged high",
              outdated and outdated[0]["severity"] == "high",
              str(findings))
    finally:
        server.shutdown()
        server.server_close()


def test_dir_probe():
    print("\n[2] Light directory probing")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.lstrip("/").split("?")[0]
            table = {"admin": (200, b"<html>admin panel</html>"),
                     ".git/config": (200, b"[core]"),
                     "private": (403, b"forbidden"),
                     "old": (301, "http://127.0.0.1/new")}
            if path in table:
                code, body = table[path]
                if code == 301:
                    self.send_response(301)
                    self.send_header("Location", body)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
            else:
                code, body = 404, b"not found"
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        try:
            dir_probe("http://8.8.8.8/", words=["admin"])
            check("dir probe guard blocks public", False, "no exception")
        except PermissionError:
            check("dir probe guard blocks public", True)
        result = dir_probe(f"http://127.0.0.1:{port}/",
                           words=["admin", ".git/config", "private",
                                  "old", "missing-page"],
                           timeout=2.0, authorized=True)
        statuses = {e["path"]: e["status"] for e in result["found"]}
        check("dir probe 200 found", statuses.get("admin") == 200,
              str(result))
        check("dir probe git config found",
              statuses.get(".git/config") == 200)
        check("dir probe 403 flagged",
              statuses.get("private") == 403)
        check("dir probe redirect flagged",
              statuses.get("old") in (301, 302), str(result))
        check("quarantine 404 discarded", "missing-page" not in statuses)
        check("checked counter", result["checked"] == 5,
              str(result["checked"]))
        engagement = {"target": f"127.0.0.1:{port}",
                      "wall": {}, "directories": result}
        findings = collect_findings(engagement)
        git = [f for f in findings if ".git" in f["title"]]
        check("exposed .git ranked high",
              git and git[0]["severity"] == "high", str(findings))
        admin = [f for f in findings if f["title"].endswith("/admin")]
        check("plain admin dir ranked low",
              admin and admin[0]["severity"] == "low")
    finally:
        server.shutdown()
        server.server_close()


def test_service_banner():
    print("\n[3] Service banner analysis")

    def _banner_listener(payload):
        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(4)

        def serve():
            while True:
                try:
                    conn, _ = srv.accept()
                    if payload:
                        conn.sendall(payload)
                    conn.close()
                except OSError:
                    return
        threading.Thread(target=serve, daemon=True).start()
        return srv, srv.getsockname()[1]

    ssh_srv, ssh_port = _banner_listener(
        b"SSH-2.0-OpenSSH_6.9p1 Ubuntu\r\n")
    quiet_srv, quiet_port = _banner_listener(None)
    try:
        try:
            service_banner("8.8.8.8", ports=[22])
            check("banner guard blocks public", False, "no exception")
        except PermissionError:
            check("banner guard blocks public", True)
        result = service_banner("127.0.0.1", ports=[ssh_port, quiet_port],
                                timeout=2.0, authorized=True)
        by_port = {s["port"]: s for s in result["services"]}
        check("ssh banner read",
              ssh_port in {p[0] for p in
                           [(s["port"], None) for s in result["services"]]},
              str(result))
        ssh = by_port.get(ssh_port)
        check("ssh service identified",
              ssh and "SSH" in (ssh["service"] or ""), str(ssh))
        check("ssh version parsed", ssh and ssh["version"] == "6.9")
        check("openssh 6.9 flagged outdated",
              ssh and ssh["outdated"] is True)
        quiet = by_port.get(quiet_port)
        check("silent port kept with no banner",
              quiet is not None and quiet["banner"] is None, str(quiet))
    finally:
        ssh_srv.close()
        quiet_srv.close()


def test_cipher_audit():
    print("\n[4] TLS cipher audit")

    try:
        cipher_audit("8.8.8.8", port=443)
        check("cipher guard blocks public", False, "no exception")
    except PermissionError:
        check("cipher guard blocks public", True)
    port = _free_port()
    result = cipher_audit("127.0.0.1", port=port, timeout=0.5,
                          authorized=True)
    check("closed port yields empty audit",
          result["accepted"] == {"weak": [], "medium": [], "strong": []}
          and result["weak_accepted"] is False, str(result))


def test_reporting():
    print("\n[5] Professional report generation")
    engagement = {
        "target": "demo.local",
        "recon": {"host": "demo.local", "addresses": ["10.0.0.5"]},
        "wall": {"wall_score": 42, "open": [22, 80, 5900],
                 "cracks": [
                     {"kind": "dangerous-service",
                      "detail": "port 5900 open: vnc \u2014 weak auth",
                      "severity": "high"},
                     {"kind": "weak-web-headers",
                      "detail": "5 security headers missing on port 80",
                      "severity": "medium"}]},
        "probe": {"findings": [
            {"param": "user", "kind": "sql-error",
             "url": "http://demo.local/", "evidence": "SQL syntax"}],
            "vulnerable": True},
        "http_fingerprint": {"ok": True, "server": "nginx/1.14.0",
                             "tech": ["nginx", "PHP", "WordPress"],
                             "powered_by": "PHP/7.2.1"},
        "directories": {"found": [
            {"path": ".git/config", "status": 200,
             "note": "exists \u2014 content served"},
            {"path": "admin", "status": 200,
             "note": "exists \u2014 content served"}]},
        "ciphers": {"weak_accepted": True,
                    "accepted": {"weak": ["RC4-SHA"], "medium": [],
                                 "strong": []}},
        "plan": [{"id": "R-001", "severity": "high",
                  "title": "Close perimeter services",
                  "steps": ["firewall vnc"], "auto_applicable": False}],
        "summary": "3 open port(s), 2 crack(s)"
    }
    findings = collect_findings(engagement)
    severities = {f["id"]: f["severity"] for f in findings}
    check("injection ranked critical",
          any(s == "critical" for s in severities.values()),
          str(severities))
    check("risky port high", any(
        "Dangerous service" in f["title"] and f["severity"] == "high"
        for f in findings))
    check("weak cipher high", any(
        "Weak TLS cipher" in f["title"] for f in findings))
    check("fingerprint info finding", any(
        f["severity"] == "info" and "fingerprint" in f["source"]
        for f in findings))
    check("findings sorted critical first",
          [f["severity"] for f in findings] == sorted(
              [f["severity"] for f in findings],
              key=lambda s: ["critical", "high", "medium", "low",
                             "info"].index(s)),
          str([f["severity"] for f in findings]))

    report = build_report(engagement, meta={
        "title": "Demo engagement", "tester": "unit-test",
        "authorization": "lab target"})
    check("risk rating critical", report["risk_rating"] == "CRITICAL",
          report["risk_rating"])
    check("posture mentions compromise",
          "compromised" in report["posture"], report["posture"])
    check("counts add up",
          report["counts"]["critical"] == 1 and
          report["counts"]["high"] >= 3, str(report["counts"]))
    check("authorization recorded",
          report["meta"]["authorization"] == "lab target")

    md = render_markdown(report)
    check("markdown has title + summary", "# Penetration Test Report"
          in md and "## Executive summary" in md)
    check("markdown tables severities", "CRITICAL 1" in md)
    check("markdown evidence present", "sql-error" in md
          and "user" in md)
    check("markdown appendix json", "## Appendix" in md and '"recon"' in md)
    evidence_hack = dict(report["findings"][0])
    evidence_hack["evidence"] = "<script>alert(1)</script>"
    report["findings"][0] = evidence_hack
    page = render_html(report)
    check("html escapes evidence",
          "&lt;script&gt;" in page and "<script>alert" not in page)

    with tempfile.TemporaryDirectory() as tmp:
        md_path = os.path.join(tmp, "report.md")
        html_path = os.path.join(tmp, "report.html")
        save_report(report, md_path)
        save_report(report, html_path)
        check("markdown report saved",
              os.path.getsize(md_path) > 500)
        check("html report saved",
              os.path.getsize(html_path) > 500)
        try:
            save_report(report, os.path.join(tmp, "report.txt"))
            check("bad extension rejected", False, "no exception")
        except ValueError:
            check("bad extension rejected", True)


def test_engagement_pipeline():
    print("\n[6] Full professional pipeline (engage -> report)")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"<html><head><title>App</title></head>body</html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Server", "Werkzeug/2.0.0 Python/3.12.0")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    try:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 8080),
                                                 Handler)
    except OSError:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 8000),
                                                 Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        try:
            run_engagement("8.8.8.8")
            check("pipeline guard blocks public", False, "no exception")
        except PermissionError:
            check("pipeline guard blocks public", True)
        engagement = run_engagement("127.0.0.1", timeout=0.4)
        check("engagement has new professional stages",
              {"http_fingerprint", "directories"} <= set(engagement),
              str(set(engagement)))
        fp = engagement["http_fingerprint"]
        check("fingerprint ran in engagement",
              fp and fp.get("ok") and any("Werkzeug" in t
                                          for t in fp["tech"]),
              str(fp))
        check("dirprobe ran in engagement",
              engagement["directories"] and
              engagement["directories"]["checked"] > 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "htb_style_report.md")
            doc = build_report(engagement,
                               meta={"title": "Authorized lab engagement",
                                     "authorization": "operator-owned lab"})
            save_report(doc, path)
            content = open(path, encoding="utf-8").read()
            check("deliverable written with scope line",
                  "Authorized lab engagement" in content and
                  "open port" in content.lower())
    finally:
        server.shutdown()
        server.server_close()


def test_cli():
    print("\n[7] CLI surface for the new commands")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cli = os.path.join(root, "modules", "cyber_tools", "clitool.py")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"ok"
            self.send_response(200)
            self.send_header("Server", "nginx/1.18.0")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        out = subprocess.run(
            [sys.executable, cli, "fingerprint",
             f"http://127.0.0.1:{port}/", "--authorized"],
            capture_output=True, text=True, timeout=60)
        check("cli fingerprint", "tech" in out.stdout and
              "nginx" in out.stdout, out.stdout + out.stderr)
        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "cli_report.md")
            out = subprocess.run(
                [sys.executable, cli, "dirs",
                 f"http://127.0.0.1:{port}/", "--words",
                 os.path.join(root, "modules", "cyber_tools",
                              "wordlist.txt"), "--authorized"],
                capture_output=True, text=True, timeout=120)
            check("cli dirs runs against bundled wordlist",
                  "path(s) checked" in out.stdout, out.stdout + out.stderr)
            out = subprocess.run(
                [sys.executable, cli, "report", "127.0.0.1",
                 "--out", report_path],
                capture_output=True, text=True, timeout=180)
            check("cli report generates deliverable",
                  "risk rating" in out.stdout and
                  os.path.exists(report_path), out.stdout + out.stderr)
    finally:
        server.shutdown()
        server.server_close()


def main():
    test_fingerprint()
    test_dir_probe()
    test_service_banner()
    test_cipher_audit()
    test_reporting()
    test_engagement_pipeline()
    test_cli()
    print(f"\n{'='*60}\nResults: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name, detail in FAIL:
            print(f"  FAILED: {name} {detail}")
        sys.exit(1)
    print("All professional pentest tests passed.")


if __name__ == "__main__":
    main()
