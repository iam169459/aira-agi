"""Cyber-wall integrity: how badly is a system's perimeter cracked?

Combines port exposure, dangerous-service detection, TLS protocol
strength, and web-header hygiene into a single 0-100 wall score with a
list of "cracks" — the entry points an attacker would use. Loopback and
private targets only unless authorized=True.
"""
import socket
import ssl

from .guard import guard
from .portscan import scan_ports, COMMON_PORTS
from .webaudit import audit_headers

# Services that should essentially never face a perimeter
RISKY_PORTS = {
    21: "ftp — cleartext credentials",
    23: "telnet — cleartext credentials and session",
    25: "smtp open relay risk",
    161: "snmp — weak default communities",
    445: "smb — historical worm target",
    389: "ldap — anonymous binds risk",
    1433: "mssql exposed to network",
    1521: "oracle exposed to network",
    3306: "mysql exposed to network",
    3389: "rdp exposed to network — brute-force target",
    5432: "postgres exposed to network",
    5900: "vnc exposed to network — weak auth",
    6379: "redis exposed to network — unauth by default",
    9200: "elasticsearch exposed to network",
    27017: "mongodb exposed to network — noauth by default",
}

_TLS_VERSIONS = [
    ("TLSv1", ssl.TLSVersion.TLSv1),
    ("TLSv1.1", ssl.TLSVersion.TLSv1_1),
    ("TLSv1.2", ssl.TLSVersion.TLSv1_2),
    ("TLSv1.3", ssl.TLSVersion.TLSv1_3),
]

_SSL_ERRORS = (ssl.SSLError, ConnectionResetError, TimeoutError, OSError)


def tls_version_check(host, port=443, timeout=3.0, authorized=False):
    """Detect which TLS protocol versions the endpoint still accepts.

    Accepting TLSv1 / TLSv1.1 is a wall crack (downgrade attacks).
    """
    guard(host, authorized, "tls version check")
    supported, rejected = [], []
    for name, version in _TLS_VERSIONS:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.minimum_version = version
        context.maximum_version = version
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        try:
            with socket.create_connection((host, port),
                                          timeout=timeout) as sock:
                with context.wrap_socket(sock, server_hostname=host):
                    supported.append(name)
        except _SSL_ERRORS:
            rejected.append(name)
    return {"host": host, "port": port, "supported": supported,
            "rejected": rejected,
            "weak_versions": [v for v in supported
                              if v in ("TLSv1", "TLSv1.1")]}


def wall_integrity(host, ports=None, timeout=0.5, skip_tls=False,
                   authorized=False):
    """Score the perimeter at `host`.

    Returns {'host', 'wall_score' (0-100), 'cracks': [...],
    'open': [...], 'tls': {...}|None, 'web': {...}|None}.
    """
    guard(host, authorized, "wall integrity check")
    cracks = []
    results = scan_ports(host, ports=ports, timeout=timeout,
                         authorized=True)
    open_ports = [r for r in results if r["state"] == "open"]
    for entry in open_ports:
        if entry["port"] in RISKY_PORTS:
            cracks.append({
                "kind": "dangerous-service",
                "detail": f"port {entry['port']} open: "
                          f"{RISKY_PORTS[entry['port']]}",
                "severity": "high",
            })
    score = 100
    score -= 8 * len(open_ports)
    score -= 15 * len(cracks)

    tls_result = None
    open_numbers = {r["port"] for r in open_ports}
    if 443 in open_numbers and not skip_tls:
        tls_result = tls_version_check(host, 443, authorized=True)
        for weak in tls_result["weak_versions"]:
            cracks.append({
                "kind": "weak-tls",
                "detail": f"port 443 accepts {weak} (downgrade risk)",
                "severity": "high",
            })
            score -= 15
        if not tls_result["supported"]:
            cracks.append({"kind": "no-tls",
                           "detail": "port 443 answered but accepted no "
                                     "TLS version probed",
                           "severity": "medium"})
            score -= 5

    web_result = None
    http_port = next((p for p in (80, 8080) if p in open_numbers), None)
    if http_port is not None:
        web_result = audit_headers(
            f"http://{host}:{http_port}/", authorized=True)
        missing = len(web_result["missing"])
        if missing:
            cracks.append({
                "kind": "weak-web-headers",
                "detail": f"{missing} security headers missing on port "
                          f"{http_port}",
                "severity": "medium",
            })
            score -= 3 * missing
        score -= max(0, 5 - web_result["score"])

    return {
        "host": host,
        "wall_score": max(0, min(100, score)),
        "cracks": cracks,
        "open": [r["port"] for r in open_ports],
        "tls": tls_result,
        "web": web_result,
    }
