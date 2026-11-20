"""Engagement console: attack a target AND produce its repair, together.

`run_engagement` chains the whole pipeline — recon, port scan, wall
integrity, web probing — and immediately compiles a repair plan from the
findings. Requires authorized=True for anything non-loopback.
"""
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
    http_port = next((p for p in (80, 8080, 8000)
                      if p in report["wall"]["open"]), None)
    if probe_web and http_port:
        scheme = "https" if http_port in (443, 8443) else "http"
        try:
            probe_report = injection_probe(
                f"{scheme}://{target}:{http_port}/", authorized=True,
                extra_params=["cmd", "file", "redirect"], timeout=3.0)
        except OSError:
            probe_report = None
    report["probe"] = probe_report
    report["cracks"] = report["wall"]["cracks"]
    report["plan"] = plan_repairs(wall_report=report["wall"],
                                  probe_report=probe_report)
    report["summary"] = (
        f"{len(report['wall']['open'])} open port(s), "
        f"{len(report['wall']['cracks'])} crack(s), wall score "
        f"{report['wall_score_before']}/100, {len(report['plan'])} "
        "repair step(s) proposed")
    return report
