"""Concurrent TCP port scanner (connect scan) with banner grabbing."""
import socket
from concurrent.futures import ThreadPoolExecutor

from .guard import guard

COMMON_PORTS = {
    20: "ftp-data", 21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp",
    53: "dns", 67: "dhcp", 80: "http", 110: "pop3", 123: "ntp",
    143: "imap", 161: "snmp", 389: "ldap", 443: "https", 445: "smb",
    465: "smtps", 587: "submission", 993: "imaps", 995: "pop3s",
    1433: "mssql", 1521: "oracle", 3306: "mysql", 3389: "rdp",
    5432: "postgres", 5900: "vnc", 6379: "redis", 8080: "http-alt",
    8443: "https-alt", 9200: "elasticsearch", 27017: "mongodb",
}

PORT_PRESETS = {
    "quick": [21, 22, 23, 25, 53, 80, 110, 143, 443, 445, 3306, 3389,
              5432, 6379, 8080, 8443, 9200, 27017],
    "web": [80, 443, 8000, 8080, 8443, 8888, 3000, 5000],
    "db": [1433, 1521, 3306, 5432, 6379, 9200, 27017],
}


def _grab_banner(sock):
    try:
        sock.settimeout(1.0)
        data = sock.recv(256)
        return data.decode("utf-8", "replace").strip()[:200]
    except OSError:
        return ""


def _probe(host, port, timeout, grab_banner):
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            banner = _grab_banner(sock) if grab_banner else ""
            return {"port": port, "state": "open",
                    "service": COMMON_PORTS.get(port, "unknown"),
                    "banner": banner}
    except (ConnectionRefusedError, TimeoutError):
        return {"port": port, "state": "closed", "service": None,
                "banner": ""}
    except OSError as exc:
        return {"port": port, "state": "filtered",
                "service": None, "banner": str(exc)[:120]}


def scan_ports(host, ports=None, timeout=0.5, workers=128,
               grab_banner=True, authorized=False):
    """TCP connect scan. Loopback/private targets only unless authorized=True."""
    guard(host, authorized, "port scan")
    if ports is None:
        ports = sorted(COMMON_PORTS)
    elif isinstance(ports, str):
        ports = PORT_PRESETS[ports]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(
            lambda p: _probe(host, p, timeout, grab_banner), ports))
    return sorted(results, key=lambda r: r["port"])
