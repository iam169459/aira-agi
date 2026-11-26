"""Red-team offense pack for Aira AGI - strictly consent-gated attack
simulation. Stdlib only, no external dependencies.

This module chains the existing audit primitives (recon, port scan, service
fingerprinting, web probes) into kill-chain-ordered offense workflows and
then turns the results into defense findings via the reporting layer.

Consent gate - STRICTER than the defensive toolkit:

``guard()`` in the defensive tools auto-allows loopback/private targets.
The offense pack does NOT. Every entry point here requires explicit
``authorized=True`` on the call (you asserting, right now, that you own or
are otherwise authorized to test the target) OR a valid consent profile
from ``consent.ConsentStore``. No profile, no assertion, no scan - even on
localhost.

Non-destructive by design: TCP connects, GET requests and detection
strings only; no exploit payloads, no data destruction, no persistence,
no exfiltration. Findings feed ``plan_repairs``/``build_report`` so an
offense run ends in defense work.
"""
from .consent import ConsentStore
from .fingerprint import (dir_probe, fingerprint_http, service_banner)
from .knowledge import suggest_for_finding
from .recon import dns_lookup, reverse_dns, subdomain_sweep
from .reporting import collect_findings
from .wall import tls_version_check
from .webpwn import injection_probe
from .portscan import scan_ports

KILL_CHAIN_PHASES = ("recon", "surface", "probe", "posture")

DEFAULT_SUBDOMAIN_WORDS = ("www", "mail", "dev", "staging", "test",
                           "vpn", "api", "admin", "portal", "intranet")


class ConsentRequired(PermissionError):
    """Raised when an offense action runs without explicit consent.

    Distinct from PermissionError so strict testers can confirm the pack
    never reports "local, skip the gate" like the defensive tools do.
    """


def strict_gate(target, authorized, action, store=None, profile=None):
    """Consent gate for offense actions.

    Allowed only when ``authorized=True`` (explicit per-call assertion) or
    when ``store`` + ``profile`` name a valid, unexpired consent profile
    that covers ``target``. Loopback does NOT bypass this gate.
    """
    if target is None:
        raise ValueError(f"{action}: a target is required")
    if authorized:
        return
    if store is not None and profile is not None:
        if isinstance(store, ConsentStore) and store.covered(profile,
                                                             target):
            return
    raise ConsentRequired(
        f"{action}: consent required for {target!r}. The offense pack "
        "never trusts 'it is only localhost' - pass authorized=True to "
        "confirm you own or are permitted to test this target, or grant "
        "a consent profile first (ConsentStore.grant(...)).")


def recon_sweep(target, authorized=False, store=None, profile=None,
                timeout=2.0):
    """Phase 1 - reconnaissance: DNS, reverse DNS, subdomain sweep.

    Returns {'target', 'dns', 'reverse', 'subdomains'}.
    """
    strict_gate(target, authorized, "recon sweep", store, profile)
    result = {"target": target, "phase": "recon"}
    try:
        result["dns"] = dns_lookup(target)
    except OSError:
        result["dns"] = {"host": target, "addresses": []}
    first = (result["dns"].get("addresses") or [None])[0]
    try:
        result["reverse"] = reverse_dns(first) if first else None
    except OSError:
        result["reverse"] = None
    result["subdomains"] = subdomain_sweep(
        target, words=DEFAULT_SUBDOMAIN_WORDS, timeout=timeout,
        authorized=True)
    return result


def attack_surface(target, authorized=False, store=None, profile=None,
                   ports=None, timeout=0.5):
    """Phase 2 - exposure mapping: open ports, service banners, TLS.

    Returns {'target', 'phase', 'ports', 'open', 'banners', 'tls'}:
    'ports' is the raw scan list, 'open' the sorted open port numbers,
    'banners' the service_banner result and 'tls' the TLS check (None
    when 443 is closed).
    """
    strict_gate(target, authorized, "attack surface", store, profile)
    result = {"target": target, "phase": "surface"}
    result["ports"] = scan_ports(target, ports=ports, timeout=timeout,
                                 authorized=True)
    open_numbers = sorted(scan["port"] for scan in result["ports"]
                          if scan.get("state") == "open")
    result["open"] = open_numbers
    banners = {}
    if open_numbers:
        try:
            banners = service_banner(target, ports=open_numbers,
                                     timeout=min(timeout * 4, 2.0),
                                     authorized=True) or {}
        except (OSError, PermissionError, ValueError):
            banners = {}
    result["banners"] = banners
    result["tls"] = None
    if 443 in open_numbers:
        try:
            result["tls"] = tls_version_check(target, port=443,
                                              timeout=min(timeout * 4, 3.0),
                                              authorized=True)
        except (OSError, PermissionError, ValueError):
            result["tls"] = None
    return result


def web_probe_suite(target, authorized=False, store=None, profile=None,
                    http_port=None, timeout=3.0):
    """Phase 3 - web probing: fingerprint, directory listing, injection
    detection probes (GET-only, detection strings, non-destructive).

    Returns {'target', 'phase', 'fingerprint', 'directories', 'injection'}.
    """
    strict_gate(target, authorized, "web probe suite", store, profile)
    result = {"target": target, "phase": "probe",
              "fingerprint": None, "directories": None,
              "injection": None}
    if http_port is None:
        open_ports = scan_ports(target, ports=(80, 443, 8000, 8080),
                                timeout=timeout * 0.4, authorized=True)
        open_numbers = [scan["port"] for scan in open_ports
                        if scan.get("state") == "open"]
        http_port = next((p for p in (80, 8080, 8000, 443)
                          if p in open_numbers), None)
    if http_port is None:
        result["error"] = "no web port found"
        return result
    scheme = "https" if http_port == 443 else "http"
    base = f"{scheme}://{target}:{http_port}/"
    try:
        result["fingerprint"] = fingerprint_http(base, timeout=timeout,
                                                 authorized=True)
    except (OSError, PermissionError, ValueError):
        result["fingerprint"] = None
    try:
        result["directories"] = dir_probe(base, timeout=timeout * 0.6,
                                          authorized=True)
    except (OSError, PermissionError, ValueError):
        result["directories"] = None
    try:
        result["injection"] = injection_probe(base, timeout=timeout,
                                              authorized=True)
    except (OSError, PermissionError, ValueError):
        result["injection"] = None
    return result


def run_offense(target, authorized=False, store=None, profile=None,
                phases=KILL_CHAIN_PHASES, timeout=0.5):
    """Full consent-gated offense run: kill-chain ordered phases turned
    into severity-rated defense findings.

    Returns {'target', 'phases': {...}, 'findings': [...], 'counts',
    'chain'} where 'chain' is a short human-readable kill-chain narrative.
    """
    strict_gate(target, authorized, "offense run", store, profile)
    report = {"target": target, "phases": {}}
    target_url = None

    if "recon" in phases:
        report["phases"]["recon"] = recon_sweep(
            target, authorized=True, timeout=max(timeout * 2, 1.0))
    if "surface" in phases:
        report["phases"]["surface"] = attack_surface(
            target, authorized=True, timeout=timeout)
        open_numbers = report["phases"]["surface"]["open"]
        web_port = next((p for p in (80, 8080, 8000, 443)
                         if p in open_numbers), None)
        if web_port:
            target_url = f"{target}:{web_port}"
    if "probe" in phases:
        report["phases"]["probe"] = web_probe_suite(
            target, authorized=True, http_port=(
                int(target_url.rsplit(":", 1)[1]) if target_url else None),
            timeout=max(timeout * 2, 1.0))

    # posture phase: roll every phase finding into the reporting format
    engagement = report["phases"].get("probe") or {}
    if "surface" in report["phases"]:
        engagement = dict(engagement)
        engagement.setdefault("ports", report["phases"]["surface"]["ports"])
        engagement.setdefault("banners",
                              report["phases"]["surface"]["banners"])
        engagement.setdefault("tls", report["phases"]["surface"]["tls"])
    findings = collect_findings(engagement) if engagement else []
    report["findings"] = findings
    counts = {}
    for finding in report["findings"]:
        counts[finding["severity"]] = counts.get(finding["severity"], 0) + 1
    report["counts"] = counts
    report["chain"] = kill_chain(report)
    return report


def kill_chain(report):
    """Human-readable kill-chain narrative + mitigation pointers for an
    offense report (uses the TECHNIQUES knowledge base)."""
    lines = []
    phase = report.get("phases", {})
    if phase.get("recon"):
        recon = phase["recon"]
        dns = recon.get("dns") or {}
        subs = recon.get("subdomains") or []
        lines.append(
            "RECON: resolved {} to {} address(es){}".format(
                dns.get("host", "?"), len(dns.get("addresses") or []),
                f", {len(subs)} subdomain hit(s)" if subs else ""))
    if phase.get("surface"):
        surface = phase["surface"]
        open_ports = surface.get("open", [])
        lines.append(
            f"SURFACE: {len(open_ports)} open port(s) {open_ports[:10]}")
        for service in (surface.get("banners") or {}).get("services", []):
            text = service.get("banner")
            if text:
                lines.append(
                    f"BANNER on port {service.get('port')}: "
                    f"{str(text)[:60]}")
    if phase.get("probe"):
        probe = phase["probe"]
        if probe.get("injection") and probe["injection"].get("vulnerable"):
            hits = probe["injection"].get("findings") or []
            for hit in hits[:5]:
                lines.append(
                    f"PROBE: {hit.get('kind', '?')} on parameter "
                    f"{hit.get('param', '?')} - {suggest_for_finding(hit['kind']) or 'review input handling'}")
        elif probe.get("error") is None and probe.get("fingerprint"):
            lines.append("PROBE: web stack fingerprinted, no injection "
                         "responses detected")
    if not lines:
        lines.append("No phases ran (empty phase list).")
    lines.append("POSTURE: findings above feed plan_repairs()/"
                 "build_report() for the defense side.")
    return "\n".join(lines)


__all__ = ["ConsentRequired", "strict_gate", "recon_sweep",
           "attack_surface", "web_probe_suite", "run_offense",
           "kill_chain", "KILL_CHAIN_PHASES", "DEFAULT_SUBDOMAIN_WORDS"]
