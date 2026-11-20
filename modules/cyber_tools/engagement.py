"""Engagement console: attack a target AND produce its repair, together.

`run_engagement` chains the whole pipeline — recon, port scan, wall
integrity, web fingerprinting, light directory enumeration, cipher
audit, injection probing — and compiles a repair plan from the findings;
`build_report` then renders the professional deliverable. Requires
authorized=True for anything non-loopback.
"""
from .fingerprint import cipher_audit, dir_probe, fingerprint_http
from .guard import guard
from .portscan import scan_ports
from .recon import dns_lookup
from .repair import plan_repairs
from .wall import wall_integrity
from .webpwn import injection_probe


def run_engagement(target, authorized=False, probe_web=True,
                   timeout=0.5):
    """Full authorized assessment: hack + repair in one pass.

    Returns {'target', 'recon', 'ports', 'wall', 'probe', 'plan',
    'wall_score_before', 'summary'}.
    """
    guard(target, authorized, "engagement")
    report = {"target": target}
    try:
        report["recon"] = dns_lookup(target)
    except OSError:
        report["recon"] = {"host": target, "addresses": []}

    report["ports"] = scan_ports(target, timeout=timeout,
                                 authorized=True)
    report["wall"] = wall_integrity(target, skip_tls=True,
                                    timeout=timeout, authorized=True)
    report["wall_score_before"] = report["wall"]["wall_score"]

    probe_report = None
    fingerprint = None
    directories = None
    ciphers = None
    open_numbers = set(report["wall"]["open"])
    if probe_web:
        http_port = next((p for p in (80, 8080, 8000)
                          if p in open_numbers), None)
        if http_port:
            base = f"http://{target}:{http_port}/"
            try:
                fingerprint = fingerprint_http(base, authorized=True,
                                               timeout=3.0)
            except (OSError, PermissionError):
                fingerprint = None
            if fingerprint and fingerprint.get("ok"):
                try:
                    directories = dir_probe(base, authorized=True,
                                            timeout=2.0)
                except (OSError, ValueError, PermissionError):
                    directories = None
            try:
                probe_report = injection_probe(
                    base, authorized=True,
                    extra_params=["cmd", "file", "redirect"],
                    timeout=3.0)
            except OSError:
                probe_report = None
        if 443 in open_numbers:
            try:
                ciphers = cipher_audit(target, 443, authorized=True)
            except (OSError, PermissionError):
                ciphers = None
    report["probe"] = probe_report
    report["http_fingerprint"] = fingerprint
    report["directories"] = directories
    report["ciphers"] = ciphers
    report["cracks"] = report["wall"]["cracks"]
    report["plan"] = plan_repairs(wall_report=report["wall"],
                                  probe_report=probe_report)
    report["summary"] = (
        f"{len(report['wall']['open'])} open port(s), "
        f"{len(report['wall']['cracks'])} crack(s), wall score "
        f"{report['wall_score_before']}/100, {len(report['plan'])} "
        "repair step(s) proposed")
    if fingerprint and fingerprint.get("ok"):
        stack = ", ".join(fingerprint["tech"]) or "unknown"
        report["summary"] += f", web stack: {stack}"
    return report
