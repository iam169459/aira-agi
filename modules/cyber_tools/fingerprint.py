"""Professional enumeration: fingerprint what a target actually runs.

CTF/pentest methodology, stdlib only — web technology fingerprinting,
lightweight directory probing, service banner analysis with version-age
checks, and a TLS cipher-suite audit. Everything is guard-gated, GET/HEAD
only, and non-destructive: no exploit payloads, no brute force, no DoS.
"""
import re
import socket
import ssl
import urllib.error
import urllib.request
from urllib.parse import urljoin, urlparse

from .guard import guard

USER_AGENT = "aira-cyber-tools/2.0"

_SERVER_HINTS = {
    "nginx": "nginx",
    "apache": "Apache",
    "caddy": "Caddy",
    "litespeed": "LiteSpeed",
    "iis": "Microsoft IIS",
    "gunicorn": "Gunicorn",
    "uvicorn": "Uvicorn",
    "werkzeug": "Werkzeug (Flask dev server)",
}

_POWERED_HINTS = {
    "php": "PHP",
    "asp.net": "ASP.NET",
    "express": "Node.js / Express",
    "next.js": "Next.js",
    "jsp": "Java / JSP",
}

_BODY_HINTS = [
    ("wp-content", "WordPress"),
    ("wp-json", "WordPress"),
    ("/_next/", "Next.js"),
    ("__NEXT_DATA__", "Next.js"),
    ("laravel_session", "Laravel"),
    ("csrfmiddlewaretoken", "Django"),
    ("X-Runtime: ", "Ruby (Rack)"),
    ("drupal", "Drupal"),
    ("joomla", "Joomla"),
]

_COOKIE_HINTS = {
    "phpsessid": "PHP",
    "jsessionid": "Java",
    "asp.net_sessionid": "ASP.NET",
    "connect.sid": "Express / Node.js",
    "csrftoken": "Django",
    "laravel_session": "Laravel",
    "rack.session": "Ruby / Rack",
    "si": "Sails.js",
}

DEFAULT_DIR_WORDS = [
    "admin", "administrator", "api", "backup", "backups", "cgi-bin",
    "config", "console", "dashboard", "db", "debug", "dev", "docs",
    ".env", ".git", ".git/config", ".htaccess", ".ssh", "images",
    "img", "index.php", "install", "jmx-console", "login", "manager",
    "phpinfo.php", "phpmyadmin", "private", "robots.txt", "server-status",
    "setup", "sitemap.xml", "src", "static", "swagger", "swagger.json",
    "temp", "test", "tests", "tmp", "uploads", "user", "wp-admin",
    "wp-login.php", "wp-config.php.bak", ".well-known/security.txt",
    "actuator", "actuator/health", "graphql", "metrics", "admin.php",
]

INTERESTING_STATUSES = {
    200: "exists — content served",
    201: "exists — resource created",
    301: "redirected (path exists)",
    302: "redirected (path exists)",
    401: "exists — requires authentication",
    403: "exists — forbidden (denied but present)",
    500: "exists — server error reveals handling",
}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirect)

_BANNER_PATTERNS = re.compile(
    r"(?i)(openssh|apache|nginx|proftpd|vsftpd|postfix|exim|dovecot|"
    r"mysql|postgresql|redis|memcached|pure-ftpd)[ _/-]*v?(\d+(?:\.\d+)+)")

_MIN_VERSIONS = {
    "openssh": (7, 0),
    "apache": (2, 4),
    "nginx": (1, 18),
    "proftpd": (1, 3, 6),
    "vsftpd": (3, 0),
    "php": (7, 4),
    "mysql": (5, 7),
    "postgresql": (9, 6),
    "redis": (4, 0),
    "memcached": (1, 5),
}

_SSH_BANNER = re.compile(r"SSH-(\d)\.(\d)-(\S+)")


def _version_tuple(text):
    return tuple(int(p) for p in re.findall(r"\d+", text)[:4])


def _outdated(product, version):
    floor = _MIN_VERSIONS.get(product.lower())
    if floor is None or version is None:
        return False
    try:
        return _version_tuple(version) < floor
    except ValueError:
        return False


def _fetch(url, timeout, opener=None):
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT,
                      "Accept": "*/*"})
    open_fn = opener.open if opener else urllib.request.urlopen
    with open_fn(req, timeout=timeout) as resp:
        body = resp.read(120000).decode("utf-8", "replace")
        return resp.status, {k.lower(): v for k, v in resp.getheaders()}, body


def fingerprint_http(url, timeout=5.0, authorized=False):
    """Identify the web stack behind a URL.

    Returns {'url', 'ok', 'status', 'server', 'powered_by', 'tech',
    'cookies', 'generator', 'hints'} — everything is passive, from a
    single GET and response headers.
    """
    guard(urlparse(url).hostname, authorized, "web fingerprint")
    result = {"url": url, "ok": False, "status": None, "server": None,
              "powered_by": None, "tech": [], "cookies": [],
              "generator": None, "hints": []}
    try:
        status, headers, body = _fetch(url, timeout)
    except (OSError, ValueError) as exc:
        result["error"] = f"fetch failed: {exc}"
        return result
    result["ok"] = True
    result["status"] = status
    server = (headers.get("server") or "").strip()
    result["server"] = server or None
    powered = (headers.get("x-powered-by") or "").strip()
    result["powered_by"] = powered or None
    tech = []
    lowered = server.lower()
    for key, name in _SERVER_HINTS.items():
        if key in lowered and name not in tech:
            tech.append(name)
    lowered = powered.lower()
    for key, name in _POWERED_HINTS.items():
        if key in lowered and name not in tech:
            tech.append(name)
    for key, name in _BODY_HINTS:
        if key.lower() in body.lower() and name not in tech:
            tech.append(name)
            result["hints"].append(f"body marker {key!r} -> {name}")
    for cookie in headers.get("set-cookie", "").split(","):
        name = cookie.split("=", 1)[0].strip()
        if name:
            result["cookies"].append(name)
        hint = _COOKIE_HINTS.get(name.lower())
        if hint and hint not in tech:
            tech.append(hint)
            result["hints"].append(f"cookie {name!r} -> {hint}")
    match = re.search(
        r'<meta[^>]+name=["\']generator["\'][^>]+content=["\']([^"\']+)',
        body, re.I)
    if match:
        result["generator"] = match.group(1)
        tech.append(match.group(1))
    result["tech"] = tech
    return result


def dir_probe(base_url, words=None, timeout=3.0, extensions=(),
              authorized=False):
    """Lightweight directory/path discovery (dirb-style, restrained).

    Plain GET requests for each word, no brute force, no recursion.
    Returns {'base_url', 'checked', 'found': [{'path', 'status', 'note'}],
    'errors'} — 'found' only lists responses whose status is interesting
    (200/201/401/403/500); ordinary 404s are discarded.
    """
    guard(urlparse(base_url).hostname, authorized, "directory probe")
    allowed = ("http", "https")
    if urlparse(base_url).scheme not in allowed:
        raise ValueError(f"dir_probe: base_url scheme must be one of {allowed}")
    words = list(words) if words is not None else list(DEFAULT_DIR_WORDS)
    result = {"base_url": base_url, "checked": 0, "found": [],
              "errors": 0}
    for word in words:
        for ext in ("",) + tuple(extensions):
            path = word + ext
            url = urljoin(base_url if base_url.endswith("/")
                          else base_url + "/", path)
            try:
                status, _, _ = _fetch(url, timeout,
                                      opener=_NO_REDIRECT_OPENER)
            except urllib.error.HTTPError as exc:
                status = exc.code
            except (OSError, ValueError):
                result["errors"] += 1
                continue
            result["checked"] += 1
            note = INTERESTING_STATUSES.get(status)
            if note:
                result["found"].append(
                    {"path": path, "status": status, "note": note})
    return result


def service_banner(host, ports=None, timeout=2.0, authorized=False):
    """Grab service banners and flag outdated software versions.

    Returns {'host', 'services': [{'port', 'banner', 'service', 'version',
    'outdated'}]}. Empty banners are kept so the caller can still see
    which ports answered. Loopback/private only unless authorized=True.
    """
    guard(host, authorized, "service banner grab")
    ports = ports or [21, 22, 23, 25, 80, 110, 143, 443, 465, 587, 993,
                      995, 1433, 3306, 3389, 5432, 5900, 6379, 8080,
                      9200, 27017]
    result = {"host": host, "services": []}
    for port in ports:
        entry = {"port": port, "banner": None, "service": None,
                 "version": None, "outdated": False}
        try:
            with socket.create_connection((host, port),
                                          timeout=timeout) as sock:
                sock.settimeout(timeout)
                try:
                    entry["banner"] = sock.recv(256).decode(
                        "utf-8", "replace").strip() or None
                except (TimeoutError, OSError):
                    entry["banner"] = None
        except OSError:
            continue
        banner = entry["banner"] or ""
        match = _SSH_BANNER.search(banner)
        product = None
        for candidate, version in _BANNER_PATTERNS.findall(banner):
            product = candidate.lower()
            if product == "pure-ftpd":
                product = None
                continue
            entry["service"] = product
            entry["version"] = version
            break
        if match:
            entry["service"] = (f"SSH ({product or match.group(3)})")
        entry["outdated"] = _outdated(product or entry["service"] or "",
                                      entry["version"])
        result["services"].append(entry)
    return result


_CIPHER_CLASSES = [
    ("weak", "EXP-RC4-MD5:EXP-DES-CBC-SHA:NULL-MD5:NULL-SHA:"
             "aNULL:eNULL:RC4-MD5:RC4-SHA:DES-CBC3-SHA:EDH-RSA-DES-CBC-SHA"),
    ("medium", "AES128-SHA:AES256-SHA:AES128-SHA256:AES256-SHA256:"
               "CAMELLIA128-SHA:SEED-SHA"),
    ("strong", "ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:"
               "ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305"),
]

_TLS13_CIPHERS = [
    "TLS_AES_128_GCM_SHA256",
    "TLS_AES_256_GCM_SHA384",
    "TLS_CHACHA20_POLY1305_SHA256",
]

_SSL_ERRORS = (ssl.SSLError, ConnectionResetError, TimeoutError, OSError)


def cipher_audit(host, port=443, timeout=3.0, authorized=False):
    """Grade the TLS cipher suites an endpoint still accepts.

    Each class is probed by offering exactly that class's ciphers and
    seeing whether the handshake succeeds. TLS 1.3 suites (always strong
    in modern OpenSSL) are reported separately. Returns
    {'host', 'port', 'accepted': {'weak', 'medium', 'strong'},
    'tls13': [...], 'weak_accepted': bool}.
    """
    guard(host, authorized, "cipher audit")
    accepted = {"weak": [], "medium": [], "strong": []}
    for label, spec in _CIPHER_CLASSES:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        context.maximum_version = ssl.TLSVersion.TLSv1_2
        try:
            context.set_ciphers(spec)
        except ssl.SSLError:
            continue
        try:
            with socket.create_connection((host, port),
                                          timeout=timeout) as sock:
                with context.wrap_socket(sock, server_hostname=host) as tls:
                    accepted[label].append(tls.cipher()[0])
        except _SSL_ERRORS:
            continue
    tls13 = []
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=host) as tls:
                tls13.append(tls.cipher()[0])
    except _SSL_ERRORS:
        pass
    result = {"host": host, "port": port, "accepted": accepted,
              "tls13": tls13, "weak_accepted": bool(accepted["weak"])}
    if tls13:
        result["tls13"] = [c for c in _TLS13_CIPHERS
                           if any(c in got for got in tls13)] or tls13
    return result
