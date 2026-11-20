"""Log analysis: IOC extraction and brute-force detection."""
import ipaddress
import re

_IP_RE = re.compile(
    r"(?<![\d.])((?:\d{1,3}\.){3}\d{1,3})(?![\d.])")
_EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_URL_RE = re.compile(r"https?://[^\s\"'<>]+")
_HEXHASH_RE = re.compile(r"\b(?:[0-9a-fA-F]{32}|[0-9a-fA-F]{40}"
                         r"|[0-9a-fA-F]{64}|[0-9a-fA-F]{128})\b")

_FAILED_PATTERNS = [
    re.compile(r"Failed password for (?:invalid user )?(?P<user>\S+) "
               r"from (?P<ip>\S+)"),
    re.compile(r"Invalid user (?P<user>\S+) from (?P<ip>\S+)"),
    re.compile(r"authentication failure.*rhost=(?P<ip>\S+)"),
    re.compile(r"Failed login for (?P<user>\S+) from (?P<ip>\S+)"),
]

_TIMESTAMPS = re.compile(
    r"^(?P<ts>[A-Z][a-z]{2}\s+\d{1,2}\s\d{2}:\d{2}:\d{2})\s")


def extract_iocs(text, drop_private_ips=False):
    """Pull indicators of compromise out of arbitrary log/text content."""
    ips = []
    for match in _IP_RE.findall(text):
        try:
            addr = ipaddress.ip_address(match)
        except ValueError:
            continue
        if addr.is_loopback or addr.is_multicast:
            continue
        if drop_private_ips and addr.is_private:
            continue
        ips.append(str(addr))
    emails = _EMAIL_RE.findall(text)
    urls = _URL_RE.findall(text)
    hashes = [h.lower() for h in _HEXHASH_RE.findall(text)]
    return {
        "ips": sorted(set(ips)),
        "emails": sorted(set(emails)),
        "urls": sorted(set(urls)),
        "hashes": sorted(set(hashes)),
    }


def brute_force_hits(lines, threshold=10, per_source=True):
    """Count failed authentications per source IP.

    `lines` may be a path or an iterable of log lines. Returns
    {'total', 'by_source': [{ip, count, users}], 'suspects': [ip, ...]}
    where suspects exceeded `threshold`.
    """
    if isinstance(lines, str) and "\n" not in lines and "\t" not in lines:
        with open(lines, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    by_source = {}
    total = 0
    for line in lines:
        for pattern in _FAILED_PATTERNS:
            match = pattern.search(line)
            if match is None:
                continue
            ip = match.group("ip")
            user = match.group("user")
            entry = by_source.setdefault(
                ip, {"ip": ip, "count": 0, "users": set()})
            entry["count"] += 1
            entry["users"].add(user)
            total += 1
            break
    results = [{"ip": e["ip"], "count": e["count"],
                "users": sorted(e["users"])}
               for e in by_source.values()]
    results.sort(key=lambda e: -e["count"])
    return {
        "total": total,
        "by_source": results,
        "suspects": [e["ip"] for e in results if e["count"] >= threshold],
    }
