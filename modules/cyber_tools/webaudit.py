"""HTTP security-header audit and TLS certificate expiry check."""
import socket
import ssl
import urllib.request
from urllib.parse import urlparse

from .guard import guard

SECURITY_HEADERS = {
    "strict-transport-security": "HSTS forces browsers onto HTTPS",
    "content-security-policy": "CSP restricts injection-prone resources",
    "x-content-type-options": "prevents MIME-type sniffing",
    "x-frame-options": "blocks clickjacking framing",
    "referrer-policy": "limits referrer leakage",
    "permissions-policy": "restricts powerful browser APIs",
}

HEADER_ADVICE = {
    "strict-transport-security": 'e.g. "max-age=31536000; includeSubDomains"',
    "content-security-policy": 'e.g. "default-src \'self\'"',
    "x-content-type-options": 'add "nosniff"',
    "x-frame-options": 'use "DENY" or "SAMEORIGIN" (or CSP frame-ancestors)',
    "referrer-policy": 'use "strict-origin-when-cross-origin"',
    "permissions-policy": "explicitly deny unused features",
}


def audit_headers(url, timeout=5.0, authorized=False):
    """Fetch a URL and grade its security-response headers.

    Returns {'ok', 'score', 'present', 'missing', 'findings'}.
    An 'ok' of False means the fetch itself failed; findings say why.
    """
    guard(urlparse(url).hostname, authorized, "web audit")
    result = {"url": url, "ok": False, "score": 0, "present": {},
              "missing": [], "findings": []}
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "aira-cyber-tools/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            headers = {k.lower(): v for k, v in resp.getheaders()}
            result["status"] = resp.status
    except (OSError, ValueError) as exc:
        result["findings"].append(f"fetch failed: {exc}")
        return result
    result["ok"] = True
    for name, why in SECURITY_HEADERS.items():
        value = headers.get(name)
        if value:
            result["present"][name] = value
            result["score"] += 1
        else:
            result["missing"].append(name)
            result["findings"].append(f"missing {name}: {why} "
                                      f"({HEADER_ADVICE[name]})")
    cookie = headers.get("set-cookie", "")
    if cookie and "httponly" not in cookie.lower():
        result["findings"].append(
            "cookie lacks HttpOnly (readable from JavaScript)")
        result["score"] -= 1
    if cookie and "secure" not in cookie.lower():
        result["findings"].append("cookie lacks the Secure flag")
        result["score"] -= 1
    csp = headers.get("content-security-policy", "")
    if csp and "unsafe-inline" in csp:
        result["findings"].append(
            "CSP contains unsafe-inline, which weakens XSS protection")
        result["score"] -= 1
    result["score"] = max(0, result["score"])
    return result


def tls_expiry(url, timeout=5.0):
    """Return TLS certificate metadata for an https URL."""
    parsed = urlparse(url if "://" in url else f"https://{url}")
    if parsed.scheme != "https":
        raise ValueError("tls_expiry expects an https URL")
    host = parsed.hostname
    port = parsed.port or 443
    context = ssl.create_default_context()
    with socket.create_connection((host, port), timeout=timeout) as sock:
        with context.wrap_socket(sock, server_hostname=host) as tls:
            cert = tls.getpeercert()
    return {
        "subject": cert.get("subject"),
        "not_before": cert.get("notBefore"),
        "not_after": cert.get("notAfter"),
        "subject_alt_names": [v for k, v in cert.get("subjectAltName", ())
                              if k == "DNS"],
    }
