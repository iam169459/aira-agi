"""Passive DNS recon: lookups, reverse DNS, and subdomain enumeration."""
import socket

from .guard import guard


def dns_lookup(host):
    """Resolve host to its address records."""
    infos = socket.getaddrinfo(host, None)
    seen = []
    for family, _type, _proto, _canon, addr in infos:
        ip = addr[0]
        if ip not in seen:
            seen.append(ip)
    return {"host": host, "addresses": seen}


def reverse_dns(ip):
    try:
        name = socket.gethostbyaddr(ip)[0]
    except (OSError, IndexError):
        return {"ip": ip, "name": None}
    return {"ip": ip, "name": name}


def subdomain_sweep(domain, words=None, timeout=2.0, authorized=False):
    """Enumerate subdomains by resolving candidate names from a wordlist.

    Loopback/private resolutions only unless authorized=True. Returns a
    list of {'subdomain', 'addresses'} for names that resolved.
    """
    guard(domain, authorized, "subdomain sweep")
    if words is None:
        words = ["www", "mail", "api", "dev", "staging", "test", "vpn",
                 "admin", "portal", "ns1", "ns2", "smtp", "git", "jenkins"]
    socket.setdefaulttimeout(timeout)
    found = []
    dns_lookup(domain)
    for word in words:
        candidate = f"{word}.{domain}"
        try:
            result = dns_lookup(candidate)
        except OSError:
            continue
        found.append({"subdomain": candidate,
                      "addresses": result["addresses"]})
    return found
