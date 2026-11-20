"""Authorized-hacking knowledge base for Aira.

Curated technique records: what each vulnerability class is, how to
detect it non-destructively (mapped to the toolkit tools that already
exist), and how to fix it. This is teaching + assessment knowledge —
no exploit payloads, by design. Every entry is framed for targets
you own or are contracted to test.
"""
import re

TECHNIQUES = [
    {
        "id": "T-SQLI",
        "name": "SQL injection",
        "class": "web",
        "severity": "critical",
        "owasp": "A03:2021 Injection",
        "summary": "Untrusted input reaches a database query. Tested "
                   "non-destructively with SQL-error probes only; never "
                   "extract, modify, or drop data.",
        "detect": "webpwn.injection_probe on parameters you own; watch "
                  "for DB error strings in responses.",
        "remediate": ["prepared statements / parameterized queries only",
                      "least-privilege DB accounts",
                      "input validation + overflow-safe ORMs"],
        "tools": ["webpwn", "engagement"],
    },
    {
        "id": "T-XSS",
        "name": "Reflected cross-site scripting",
        "class": "web",
        "severity": "high",
        "owasp": "A03:2021 Injection",
        "summary": "Input echoed into HTML without encoding. Probed with "
                   "benign marker strings only — no script payloads.",
        "detect": "webpwn.injection_probe marker reflection check.",
        "remediate": ["context-aware output encoding (HTML/JS/URL)",
                      "Content-Security-Policy response header",
                      "X-XSS-Protection: 1; mode=block historically"],
        "tools": ["webpwn", "repair"],
    },
    {
        "id": "T-AUTHN",
        "name": "Weak or bypassable authentication",
        "class": "web",
        "severity": "critical",
        "owasp": "A07:2021 Identification and Authentication Failures",
        "summary": "Default/reused/weak credentials, missing lockout, "
                   "predictable session tokens. Off-line credential "
                   "auditing only (crack.py), never against live logins.",
        "detect": "crack.audit_credentials on hashes you hold; wall "
                  "score for login surfaces.",
        "remediate": ["MFA, Argon2id/bcrypt storage",
                      "account lockout + delays",
                      "CSPRNG session tokens with rotation"],
        "tools": ["crack", "wall"],
    },
    {
        "id": "T-COOKIE",
        "name": "Insecure session-cookie flags",
        "class": "web",
        "severity": "medium",
        "owasp": "A05:2021 Security Misconfiguration",
        "summary": "Session cookies without Secure/HttpOnly/SameSite "
                   "are exposed over plain HTTP or readable by script.",
        "detect": "webaudit.audit_headers cookie flag inspection.",
        "remediate": ["Set-Cookie: Secure; HttpOnly; SameSite=Lax",
                      "short lifetimes, rotate on privilege change"],
        "tools": ["webaudit", "repair"],
    },
    {
        "id": "T-HSTS",
        "name": "Missing transport security headers",
        "class": "web",
        "severity": "medium",
        "owasp": "A05:2021 Security Misconfiguration",
        "summary": "No HSTS / CSP / XCTO / frame options: downgrade and "
                   "clickjacking exposure.",
        "detect": "webaudit.audit_headers (the 6 canonical headers).",
        "remediate": ["Strict-Transport-Security: max-age=63072000; "
                      "includeSubDomains; preload",
                      "frame-ancestors CSP + X-Content-Type-Options"],
        "tools": ["webaudit", "repair"],
    },
    {
        "id": "T-TLS",
        "name": "TLS downgrade and weak ciphers",
        "class": "network",
        "severity": "high",
        "owasp": "A02:2021 Cryptographic Failures",
        "summary": "TLSv1.0/1.1 or weak/medium cipher suites still "
                   "negotiable on public endpoints.",
        "detect": "wall.tls_version_check + fingerprint.cipher_audit; "
                  "decryptable traffic outside CCAs should not exist.",
        "remediate": ["TLSv1.2+ only, prefer 1.3",
                      "disable EXPORT/NULL/anon suites",
                      "ACME auto-renewal (expiry < 30d is a tripwire)"],
        "tools": ["wall", "fingerprint"],
    },
    {
        "id": "T-BANNER",
        "name": "Outdated service versions",
        "class": "network",
        "severity": "medium",
        "owasp": "A06:2021 Vulnerable and Outdated Components",
        "summary": "Banners advertise old OpenSSH/Apache/nginx/PHP; "
                   "known CVE ranges apply.",
        "detect": "fingerprint.service_banner version-age table.",
        "remediate": ["patch to current stable",
                      "suppress version banners as defense-in-depth"],
        "tools": ["fingerprint", "reporting"],
    },
    {
        "id": "T-DIRLIST",
        "name": "Exposed directories and listings",
        "class": "web",
        "severity": "medium",
        "owasp": "A05:2021 Security Misconfiguration",
        "summary": "/backup, /.git, /admin-style paths readable; index "
                   "listings leak trees.",
        "detect": "fingerprint.dir_probe with redirect catching.",
        "remediate": ["deny rules in reverse proxy config",
                      "remove VCS/build artifacts from webroot"],
        "tools": ["fingerprint", "webpwn"],
    },
    {
        "id": "T-PORTS",
        "name": "Dangerous service exposure",
        "class": "network",
        "severity": "high",
        "owasp": "A05:2021 Security Misconfiguration",
        "summary": "telnet/VNC/RDP/DB engines reachable from untrusted "
                   "networks (the cyber-wall crack list).",
        "detect": "wall.wall_integrity RISKY_PORTS sweep + banner grab.",
        "remediate": ["bind to loopback/private only",
                      "firewall allow-lists; VPN for admin planes"],
        "tools": ["wall", "portscan", "repair"],
    },
    {
        "id": "T-BRUTE",
        "name": "Brute-force / credential stuffing",
        "class": "host",
        "severity": "high",
        "owasp": "A07:2021 Identification and Authentication Failures",
        "summary": "Detect ongoing attacks in your own logs — the "
                   "defensive side of access abuse.",
        "detect": "logscan.brute_force_hits + IOC extraction; "
                  "sentinel live watch on auth logs.",
        "remediate": ["rate limiting, lockout, MFA",
                      "fail2ban-style bans; alert on spikes"],
        "tools": ["logscan", "sentinel"],
    },
    {
        "id": "T-IDOR",
        "name": "Insecure direct object references",
        "class": "web",
        "severity": "high",
        "owasp": "A01:2021 Broken Access Control",
        "summary": "Sequential or guessable IDs without ownership "
                   "checks. Test only with accounts you control.",
        "detect": "authorized probe of two own accounts; diff access.",
        "remediate": ["server-side ownership checks per request",
                      "unpredictable UUIDs (defense-in-depth)"],
        "tools": ["webpwn", "engagement"],
    },
    {
        "id": "T-SSRF",
        "name": "Server-side request forgery",
        "class": "web",
        "severity": "high",
        "owasp": "A10:2021 SSRF",
        "summary": "URL-fetch parameters reach internal services. "
                   "Detection idea only — point fetches at attacker-"
                   "uncontrolled benign endpoints you host.",
        "detect": "probe URL params with a host you own; watch callbacks.",
        "remediate": ["allow-list destinations, block RFC1918 at the "
                      "fetch layer", "metadata-service protections"],
        "tools": ["webpwn"],
    },
    {
        "id": "T-TRAVERSAL",
        "name": "Path traversal",
        "class": "web",
        "severity": "high",
        "owasp": "A01:2021 Broken Access Control",
        "summary": "File-serve paths escape the webroot. Probed with "
                   "benign known-file reads only.",
        "detect": "urlencode /../../ patterns against a known harmless "
                  "file you placed.",
        "remediate": ["normalize + prefix-check resolved paths",
                      "serve from immutable directories"],
        "tools": ["webpwn"],
    },
    {
        "id": "T-CSRF",
        "name": "Cross-site request forgery",
        "class": "web",
        "severity": "medium",
        "owasp": "A01:2021 Broken Access Control",
        "summary": "State-changing endpoints lack per-session tokens.",
        "detect": "attempt save/action without token while authorized; "
                  "expect 4xx.",
        "remediate": ["synchronizer tokens, SameSite cookies",
                      "require re-auth for sensitive actions"],
        "tools": ["webaudit"],
    },
    {
        "id": "T-SECRETS",
        "name": "Secrets in code and dependencies",
        "class": "code",
        "severity": "critical",
        "owasp": "A02:2021 Cryptographic Failures",
        "summary": "API keys, tokens, DB passwords hardcoded in source "
                   "or required on client side.",
        "detect": "dev_tools.codeaudit secret rules = local integrity "
                  "scan; logscan IOC regexes for log leakage.",
        "remediate": ["move to env/secret manager, rotate leaked keys",
                      "git-history scrubbing after rotation"],
        "tools": ["codeaudit", "logscan"],
    },
    {
        "id": "T-TAMPER",
        "name": "File tampering and persistence",
        "class": "host",
        "severity": "high",
        "owasp": "A08:2021 Software and Data Integrity Failures",
        "summary": "Unauthorized modification of served code or "
                   "configuration — the defensive watch.",
        "detect": "sentinel hash watch + integrity baselines "
                  "(hashing.baseline_build/verify).",
        "remediate": ["FIM baselines + alerting",
                      "immutable deployments, signed artifacts"],
        "tools": ["sentinel", "hashing"],
    },
]

# Phased assessment methodology — the same flow run_engagement automates.
METHODOLOGY = [
    {"phase": "1. reconnaissance",
     "goal": "know the surface before touching anything",
     "actions": ["recon.dns_lookup / reverse_dns / subdomain_sweep",
                 "fingerprint.fingerprint_http (passive header read)"],
     "rules": "passive/allowed only; no interaction beyond normal use"},
    {"phase": "2. enumeration",
     "goal": "map services and versions",
     "actions": ["portscan.scan_ports (conservative presets)",
                 "fingerprint.service_banner",
                 "fingerprint.dir_probe"],
     "rules": "loopback/private by default; authorized=True or a "
              "consent profile for anything else"},
    {"phase": "3. wall assessment",
     "goal": "crack detection: what would an attacker see?",
     "actions": ["wall.wall_integrity", "fingerprint.cipher_audit",
                 "wall.tls_version_check", "webaudit.audit_headers"],
     "rules": "no flooding; sequential low-rate scans only"},
    {"phase": "4. active probing",
     "goal": "confirm or refute findings with benign payloads",
     "actions": ["webpwn.injection_probe (markers, no real payloads)",
                 "crack.audit_credentials (offline hashes only)"],
     "rules": "explicit authorization; non-destructive payloads only"},
    {"phase": "5. reporting + repair",
     "goal": "turn findings into fixed code",
     "actions": ["reporting.build_report / save_report",
                 "repair.plan_repairs (ready-to-apply snippets)",
                 "dev_tools.codeaudit for the code-side findings"],
     "rules": "retest after each fix; keep the report"},
]


def lookup(query):
    """Score-and-rank technique search over name/id/class/keywords."""
    words = [w for w in re.split(r"\W+", (query or "").lower()) if w]
    if not words:
        return []
    scored = []
    for tech in TECHNIQUES:
        hay = " ".join([tech["id"].lower(), tech["name"].lower(),
                        tech["class"],
                        tech["owasp"].lower(), tech["summary"].lower(),
                        " ".join(tech["remediate"]).lower()]).lower()
        score = sum(hay.count(w) for w in words)
        if tech["id"].lower().startswith(query.lower()) and len(query) >= 2:
            score += 5
        if score:
            scored.append((score, tech))
    scored.sort(key=lambda pair: -pair[0])
    return [tech for _, tech in scored]


def show(technique_id):
    """Full record for one technique (case-insensitive, T- prefix ok)."""
    wanted = technique_id.lower().replace("t-", "")
    for tech in TECHNIQUES:
        if tech["id"].lower().replace("t-", "") == wanted:
            return tech
    return None


def suggest_for_finding(kind):
    """Map a toolkit finding kind (crack/audit/repair vocabulary) to
    the relevant technique record."""
    mapping = {
        "risky_port": "T-PORTS", "tls_version": "T-TLS",
        "header": "T-HSTS", "headers": "T-HSTS", "cookie": "T-COOKIE",
        "sqli": "T-SQLI", "xss": "T-XSS", "sql": "T-SQLI",
        "brute": "T-BRUTE", "auth": "T-AUTHN", "version": "T-BANNER",
        "dir": "T-DIRLIST", "secret": "T-SECRETS", "tamper": "T-TAMPER",
        "csrf": "T-CSRF", "ssrf": "T-SSRF", "traversal": "T-TRAVERSAL",
        "idor": "T-IDOR",
    }
    tech_id = mapping.get((kind or "").lower())
    return show(tech_id) if tech_id else None
