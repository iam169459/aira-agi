"""Repair engine: turn findings/cracks into concrete fixes.

Sanitize-first philosophy: the engine proposes a plan, applies only the
actions explicitly flagged auto-applicable, and never touches a network
target — repairs are local configuration or generated snippets.
"""
import os
import stat

from .hashing import generate_password


def plan_repairs(wall_report=None, credential_report=None,
                 probe_report=None):
    """Build a remediation plan from audit outputs.

    Each item: {'id', 'severity', 'title', 'steps': [...],
    'auto_applicable': bool}.
    """
    plan = []
    seq = 0

    def add(severity, title, steps, auto=False):
        nonlocal seq
        seq += 1
        plan.append({"id": f"R-{seq:03d}", "severity": severity,
                     "title": title, "steps": steps,
                     "auto_applicable": auto})

    if wall_report is not None:
        for crack in wall_report.get("cracks", []):
            detail = crack["detail"]
            if crack["kind"] == "dangerous-service":
                port = next(
                    (seg for seg in detail.replace(":", " ").split()
                     if seg.isdigit() and 0 < int(seg) < 65536), "?")
                add("high", f"Close exposed service on port {port}", [
                    f"identify the listener: ss -tlnp | grep ':{port}'",
                    "stop or reconfigure the service",
                    f"deny external traffic: sudo ufw deny {port}/tcp",
                    "re-run wall_integrity to confirm the crack is gone",
                ])
            elif crack["kind"] == "weak-tls":
                version = "TLSv1.1" if "TLSv1.1" in detail else "TLSv1"
                add("high", f"Disable {version}", [
                    "nginx: ssl_protocols TLSv1.2 TLSv1.3;",
                    "apache: SSLProtocol -all +TLSv1.2 +TLSv1.3",
                    "reload the web server and re-run tls_version_check",
                ])
            elif crack["kind"] == "weak-web-headers":
                add("medium", "Add missing security headers", [
                    "deploy the headers from webaudit.HEADER_ADVICE",
                    'strict-transport-security: "max-age=31536000; '
                    'includeSubDomains"',
                    'content-security-policy: "default-src \'self\'"',
                    'x-content-type-options: "nosniff"',
                ])
            else:
                add(crack.get("severity", "medium"), detail,
                    ["investigate and re-scan"])

    if credential_report is not None:
        for user in credential_report.get("cracked", {}):
            replacement = generate_password(20)
            add("high", f"Rotate compromised credential for '{user}'", [
                f"replace the credential for {user} immediately",
                f"suggested replacement (keep secret): {replacement}",
                "re-hash with PBKDF2-HMAC-SHA256, >=100k iterations",
            ])

    if probe_report is not None:
        params = sorted({f["param"]
                         for f in probe_report.get("findings", [])})
        kinds = {f["kind"] for f in probe_report.get("findings", [])}
        if params:
            add("high",
                "Parameterize SQL and encode output for: "
                + ", ".join(params), [
                    "use prepared statements / bound parameters",
                    "contextually escape output (HTML entities)",
                    "add Content-Security-Policy as a second layer",
                ])
        if "xss" in kinds:
            add("medium", "Tighten reflected-input handling", [
                "validate and reject unexpected parameter values",
                "apply auto-escaping templates",
            ])

    return plan


def secure_permissions(paths, confirm=False):
    """Un-world-writable local files.

    Only acts when confirm=True; otherwise reports what would change.
    """
    changed, skipped = [], []
    for path in paths:
        try:
            mode = os.stat(path).st_mode
        except OSError as exc:
            skipped.append((path, str(exc)))
            continue
        if not stat.S_ISDIR(mode) and mode & (stat.S_IWGRP | stat.S_IWOTH):
            new_mode = mode & ~(stat.S_IWGRP | stat.S_IWOTH)
            if confirm:
                os.chmod(path, new_mode)
            changed.append({"path": path,
                            "from": stat.filemode(mode),
                            "to": stat.filemode(new_mode)})
        else:
            skipped.append((path, "already secure"))
    return {"changed": changed, "skipped": skipped, "applied": confirm}
