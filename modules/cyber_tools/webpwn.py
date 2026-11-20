"""Injection probing against web endpoints you are authorized to test.

Non-destructive by design: GET requests only, no data extraction, no
payloads beyond detection strings. Results are evidence for a repair plan.
"""
import urllib.parse
import urllib.request

from .guard import guard

SQL_ERROR_SIGNATURES = [
    "you have an error in your sql syntax",
    "unclosed quotation mark",
    "unterminated quoted string",
    "syntax error at or near",
    "sqlite3.operationalerror",
    "warning: mysql",
    "pg_query()",
    "ora-01756",
    "quoted string not properly terminated",
]

REFLECTION_PROBES = {
    "sql-error": ["'", '"'],
    "xss": [
        "<script>aira-probe</script>",
        "\"><svg onload=aira-probe()>",
        "<img src=x onerror=aira-probe()>",
    ],
}

DEFAULT_PARAMS = ["q", "search", "id", "user", "name", "page", "query"]


def _fetch(url, timeout):
    req = urllib.request.Request(
        url, headers={"User-Agent": "aira-cyber-tools/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8", "replace")


def injection_probe(url, params=None, timeout=5.0, authorized=False,
                    extra_params=None):
    """Probe GET parameters for reflected XSS and SQL error disclosure.

    Returns {'url', 'probes', 'findings': [{'param','kind','probe',
    'evidence'}], 'vulnerable': bool}.
    """
    guard(urllib.parse.urlparse(url).hostname, authorized,
          "injection probe")
    params = params or DEFAULT_PARAMS + (extra_params or [])
    findings = []
    probes_sent = 0
    for param in params:
        for kind, payload_list in REFLECTION_PROBES.items():
            for payload in payload_list:
                query = {p: "aira-baseline" for p in params}
                query[param] = payload
                try:
                    status, body = _fetch(
                        url + ("&" if "?" in url else "?")
                        + urllib.parse.urlencode(query), timeout)
                except OSError:
                    continue
                probes_sent += 1
                lowered = body.lower()
                if kind == "sql-error":
                    evidence = next(
                        (sig for sig in SQL_ERROR_SIGNATURES
                         if sig in lowered), None)
                else:
                    reflected = "aira-probe" in body
                    unfiltered = any(
                        tag in lowered for tag in
                        ("<script>", "onerror=", "onload="))
                    evidence = (reflected and unfiltered) or None
                if evidence:
                    findings.append({
                        "param": param, "kind": kind, "probe": payload,
                        "evidence": str(evidence)[:120],
                    })
    findings.sort(key=lambda f: (f["param"], f["kind"]))
    return {"url": url, "probes": probes_sent, "findings": findings,
            "vulnerable": bool(findings)}
