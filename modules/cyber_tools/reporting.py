"""Professional pentest reporting: findings, severity, executive summary.

Turns the raw output of `engagement.run_engagement` (plus the
`fingerprint` module's enumeration results) into a severity-ranked
findings register and renders it as Markdown or HTML — the deliverable
a professional engagement ends with.
"""
import datetime
import html
import json

from .fingerprint import _outdated

SEVERITY_ORDER = ["critical", "high", "medium", "low", "info"]

_SEVERITY_LABEL = {
    "critical": "CRITICAL",
    "high": "HIGH",
    "medium": "MEDIUM",
    "low": "LOW",
    "info": "INFO",
}

_CRACK_SEVERITY = {
    "dangerous-service": "high",
    "weak-tls": "high",
    "no-tls": "medium",
    "weak-web-headers": "medium",
}

CRACK_REMEDIATION = {
    "dangerous-service": (
        "Close the service on the perimeter or restrict it to trusted "
        "networks with a firewall; never expose databases, RDP, VNC or "
        "message queues directly to the internet."),
    "weak-tls": (
        "Disable the legacy protocol versions in the TLS stack and keep "
        "TLS 1.2/1.3 only."),
    "no-tls": (
        "The TLS endpoint answered but none of the standard protocol "
        "versions completed a handshake — verify the service "
        "configuration and certificate."),
    "weak-web-headers": (
        "Add the missing security headers (CSP, X-Content-Type-Options, "
        "X-Frame-Options, Referrer-Policy, Permissions-Policy, HSTS)."),
}

_HIGH_RISK_PATHS = (".git", ".env", ".htaccess", ".ssh",
                    "wp-config.php.bak", "web.config.bak")

_TITLE_BY_KIND = {
    "dangerous-service": "Dangerous service exposed on the perimeter",
    "weak-tls": "Legacy TLS protocol version accepted",
    "no-tls": "TLS endpoint rejected all standard protocol versions",
    "weak-web-headers": "Required HTTP security headers are missing",
}


def collect_findings(engagement):
    """Normalize every finding in an engagement dict.

    Each finding: {'id', 'title', 'severity', 'evidence', 'remediation',
    'source'}. Sorted critical-first, info-last.
    """
    findings = []

    def add(title, severity, evidence, remediation, source):
        findings.append({
            "id": f"F-{len(findings) + 1:03d}",
            "title": title,
            "severity": severity if severity in SEVERITY_ORDER else "info",
            "evidence": evidence,
            "remediation": remediation,
            "source": source,
        })

    wall = engagement.get("wall") or {}
    for crack in wall.get("cracks", []):
        add(_TITLE_BY_KIND.get(crack.get("kind"), crack.get("kind", "crack")),
            crack.get("severity", "medium"),
            crack.get("detail", ""),
            CRACK_REMEDIATION.get(crack.get("kind"),
                                  "Review and remediate the finding."),
            "wall_integrity")

    probe = engagement.get("probe")
    if probe and probe.get("findings"):
        for item in probe["findings"]:
            kind = item.get("kind", "")
            title = ("SQL injection (error-based fingerprint)"
                     if "sql" in kind else "Reflected XSS")
            evidence = (f"parameter '{item.get('param', '?')}' at "
                        f"{item.get('url', probe.get('url', 'unknown'))} "
                        f"triggered {kind} via probe "
                        f"{item.get('probe', '')!r}")
            add(title, "critical", evidence,
                "Parameterize database queries and HTML-escape all "
                "reflected output; add a Content-Security-Policy.",
                "injection_probe")

    fp = engagement.get("http_fingerprint")
    if fp and fp.get("ok"):
        server = fp.get("server") or ""
        powered = fp.get("powered_by") or ""
        for label, value in (("server", server), ("x-powered-by", powered)):
            if label == "server" and value:
                parts = value.split("/", 1)
                version = parts[1] if len(parts) > 1 else None
                if version and _outdated(parts[0], version):
                    add(f"Outdated software fingerprinted: {value}",
                        "high",
                        f"'{label}' header reports {value}",
                        f"Upgrade {parts[0].strip()} to a maintained "
                        "release; patch known CVEs.",
                        "fingerprint_http")
        tech = fp.get("tech") or []
        if tech:
            add("Web technology stack fingerprinted", "info",
                ", ".join(tech),
                "Information disclosure only: minimize version banners "
                "and framework markers on production servers.",
                "fingerprint_http")

    dirs = engagement.get("directories")
    if dirs:
        for entry in dirs.get("found", []):
            path = entry.get("path", "")
            risky = any(secret in path.lower()
                        for secret in _HIGH_RISK_PATHS)
            add(f"Sensitive path exposed: /{path}",
                "high" if risky else "low",
                f"/{path} returned HTTP {entry.get('status')} "
                f"({entry.get('note', '')})",
                (f"Restrict or remove {path!r} from the public surface."
                 if risky else
                 f"Review whether /{path} should be publicly reachable."),
                "dir_probe")

    ciphers = engagement.get("ciphers")
    if ciphers and ciphers.get("weak_accepted"):
        add("Weak TLS cipher suites accepted", "high",
            "handshake completed with: " + ", ".join(ciphers["accepted"]["weak"]),
            "Disable export/RC4/3DES/CBC cipher suites; serve only "
            "AEAD ciphers (AES-GCM, ChaCha20-Poly1305).",
            "cipher_audit")

    order = {s: i for i, s in enumerate(SEVERITY_ORDER)}
    findings.sort(key=lambda f: (order[f["severity"]], f["id"]))
    return findings


def _risk_rating(counts):
    for severity in SEVERITY_ORDER:
        if counts.get(severity):
            return severity
    return "clean"


def build_report(engagement, meta=None):
    """Assemble the full report structure from an engagement dict."""
    meta = dict(meta or {})
    findings = collect_findings(engagement)
    counts = {s: 0 for s in SEVERITY_ORDER}
    for finding in findings:
        counts[finding["severity"]] += 1
    wall = engagement.get("wall") or {}
    critical = counts["critical"] + counts["high"]
    if counts["critical"]:
        posture = ("CRITICAL — exploitable injection paths were confirmed; "
                   "treat the target as compromised until remediated.")
    elif counts["high"]:
        posture = (f"HIGH — {critical} high-severity weakness(es) give an "
                   "attacker a viable foothold; remediate before the next "
                   "release cycle.")
    elif counts["medium"]:
        posture = ("MODERATE — no direct exploitation path found, but the "
                   "perimeter leaks information that lowers attack cost "
                   "for a targeted adversary.")
    elif counts["low"] or counts["info"]:
        posture = ("GOOD — hardening gaps are informational; no "
                   "actionable foothold was identified.")
    else:
        posture = "CLEAN — no weaknesses surfaced with the probes used."
    report = {
        "target": engagement.get("target", "unknown"),
        "generated": datetime.datetime.now(datetime.timezone.utc)
                              .strftime("%Y-%m-%d %H:%M UTC"),
        "meta": {
            "title": meta.get("title",
                              f"Security engagement — {engagement.get('target', 'unknown')}"),
            "tester": meta.get("tester", "Aira AGI cyber toolkit"),
            "authorization": meta.get(
                "authorization",
                "Target confirmed owned/operated by the requestor "
                "(authorized engagement)."),
            "scope": meta.get("scope",
                              engagement.get("target", "unknown")),
        },
        "wall_score": wall.get("wall_score"),
        "open_ports": wall.get("open", []),
        "counts": counts,
        "risk_rating": _risk_rating(counts).upper(),
        "posture": posture,
        "summary": engagement.get("summary", ""),
        "findings": findings,
        "plan": engagement.get("plan", []),
        "recon": engagement.get("recon"),
        "fingerprint": engagement.get("http_fingerprint"),
        "directories": engagement.get("directories"),
        "ciphers": engagement.get("ciphers"),
    }
    return report


def _severity_table(counts):
    return " | ".join(
        f"{_SEVERITY_LABEL[s].upper()} {counts.get(s, 0)}"
        for s in SEVERITY_ORDER)


def render_markdown(report):
    """Render the report structure as a Markdown document."""
    lines = [f"# Penetration Test Report — {report['target']}", ""]
    lines.append(f"**{report['meta']['title']}**")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Date | {report['generated']} |")
    lines.append(f"| Tester | {report['meta']['tester']} |")
    lines.append(f"| Scope | {report['meta']['scope']} |")
    lines.append(f"| Authorization | {report['meta']['authorization']} |")
    lines.append(f"| Overall risk | **{report['risk_rating']}** |")
    lines.append("")
    lines.append("## Executive summary")
    lines.append("")
    lines.append(report["posture"])
    lines.append("")
    lines.append(f"Findings: {_severity_table(report['counts'])}"
                 f" — wall integrity {report.get('wall_score')}/100.")
    for port in report.get("open_ports", []):
        lines.append(f"- open port: {port}")
    lines.append("")
    lines.append("## Findings")
    lines.append("")
    if not report["findings"]:
        lines.append("_No weaknesses surfaced with the probes used._")
    for finding in report["findings"]:
        lines.append(
            f"### [{_SEVERITY_LABEL[finding['severity']]}] "
            f"{finding['id']} — {finding['title']}")
        lines.append("")
        lines.append(f"- **Evidence:** {finding['evidence']}")
        lines.append(f"- **Remediation:** {finding['remediation']}")
        lines.append(f"- **Source:** `{finding['source']}`")
        lines.append("")
    lines.append("## Remediation plan")
    lines.append("")
    for item in report["plan"]:
        lines.append(f"### {item['id']} — {item['title']} "
                     f"({item.get('severity', '')})")
        for step in item["steps"]:
            lines.append(f"- {step}")
    lines.append("")
    lines.append("## Appendix — raw tool output")
    lines.append("")
    lines.append("```json")
    raw = {k: report.get(k) for k in
           ("recon", "fingerprint", "directories", "ciphers")}
    lines.append(json.dumps(raw, indent=2, default=str))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


_SEV_COLOR = {
    "CRITICAL": "#b91c1c",
    "HIGH": "#c2410c",
    "MEDIUM": "#a16207",
    "LOW": "#4d7c0f",
    "INFO": "#475569",
}


def render_html(report):
    """Render the report structure as a self-contained HTML document."""
    esc = html.escape

    def sev_badge(severity):
        color = _SEV_COLOR.get(severity.upper(), "#475569")
        return (f'<span style="background:{color};color:#fff;'
                f'padding:2px 8px;border-radius:4px;font-size:12px;'
                f'font-weight:600">{esc(severity)}</span>')

    parts = ["<!DOCTYPE html><html><head><meta charset=\"utf-8\">",
             "<title>", esc(report["meta"]["title"]), "</title>",
             "<style>body{font-family:system-ui,sans-serif;max-width:"
             "860px;margin:2rem auto;padding:0 1rem;color:#111}"
             "table{border-collapse:collapse;width:100%}td,th{border:"
             "1px solid #ddd;padding:6px 10px;text-align:left}"
             "pre{background:#f4f4f5;padding:12px;overflow-x:auto}"
             ".card{border:1px solid #e5e7eb;border-radius:8px;"
             "padding:12px 16px;margin:12px 0}</style></head><body>"]
    parts.append(f"<h1>{esc(report['meta']['title'])}</h1>")
    parts.append(f"<p>Risk rating: <strong>{esc(report['risk_rating'])}"
                 f"</strong> &middot; wall integrity "
                 f"{esc(str(report.get('wall_score')))}/100 &middot; "
                 f"{esc(report['generated'])}</p>")
    parts.append("<table>")
    parts.append(f"<tr><th>Date</th><td>{esc(report['generated'])}</td></tr>")
    for row in ("tester", "scope", "authorization"):
        parts.append(f"<tr><th>{row.title()}</th><td>"
                     f"{esc(str(report['meta'][row]))}</td></tr>")
    parts.append("</table>")
    parts.append("<h2>Executive summary</h2>")
    parts.append(f"<p>{esc(report['posture'])}</p>")
    parts.append("<p>" + esc(_severity_table(report["counts"])) + "</p>")
    parts.append("<h2>Findings</h2>")
    if not report["findings"]:
        parts.append("<p>No weaknesses surfaced with the probes used.</p>")
    for finding in report["findings"]:
        parts.append('<div class="card">')
        parts.append(
            f"<h3>{sev_badge(_SEVERITY_LABEL[finding['severity']])} "
            f"{esc(finding['id'])} — {esc(finding['title'])}</h3>")
        parts.append(f"<p><strong>Evidence:</strong> "
                     f"{esc(finding['evidence'])}</p>")
        parts.append(f"<p><strong>Remediation:</strong> "
                     f"{esc(finding['remediation'])}</p>")
        parts.append(f"<p><code>{esc(finding['source'])}</code></p>")
        parts.append("</div>")
    parts.append("<h2>Remediation plan</h2>")
    for item in report["plan"]:
        parts.append('<div class="card">')
        parts.append(f"<h3>{esc(item['id'])} — {esc(item['title'])}</h3>")
        parts.append("<ul>")
        for step in item["steps"]:
            parts.append(f"<li>{esc(step)}</li>")
        parts.append("</ul></div>")
    parts.append("<h2>Appendix — raw tool output</h2><pre>")
    raw = {k: report.get(k) for k in
           ("recon", "fingerprint", "directories", "ciphers")}
    parts.append(esc(json.dumps(raw, indent=2, default=str)))
    parts.append("</pre></body></html>")
    return "".join(parts)


def save_report(report, path):
    """Write the report to `path`; format follows the extension."""
    if path.lower().endswith(".html") or path.lower().endswith(".htm"):
        rendered = render_html(report)
    elif path.lower().endswith(".md") or path.lower().endswith(".markdown"):
        rendered = render_markdown(report)
    else:
        raise ValueError("save_report: path must end in .md or .html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(rendered)
    return path
