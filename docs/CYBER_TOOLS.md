# Cybersecurity Toolkit (`modules/cyber_tools`)

Aira ships a stdlib-only cybersecurity toolkit — no external security
libraries, no external APIs. It covers both **defensive auditing** and
**authorized-target testing**, with authorization guardrails built in.

## Safety model

- Every active tool (port scan, subdomain sweep, web audit) refuses public
  targets unless you pass `authorized=True`.
- Loopback (127.0.0.1) and RFC1918/private addresses always work, so local
  hardening scans never need extra flags.
- The toolkit never exploits anything; it enumerates, verifies, and reports.

## Modules

| File | What it does |
|------|--------------|
| `guard.py` | `guard(target, authorized, action)` — blocks unauthorized active scans |
| `hashing.py` | SHA-1/2 digests, HMAC, hash-family identification, password strength, file-integrity baselines, password/passphrase generators |
| `crypto.py` | From-first-principles symmetric encryption: SHA-256-CTR keystream + HMAC-SHA256 (encrypt-then-MAC), PBKDF2-HMAC-SHA256 key derivation (200k iterations) |
| `portscan.py` | Concurrent TCP connect scan with banner grabbing, common-port/service map, `quick`/`web`/`db` presets |
| `webaudit.py` | HTTP security-header audit (CSP, HSTS, X-Frame-Options, cookie flags…) with a score and findings, TLS certificate expiry |
| `logscan.py` | IOC extraction (IPs, emails, URLs, code hashes), failed-auth/brute-force detection from syslog-style logs |
| `recon.py` | DNS lookup, reverse DNS, subdomain enumeration from a wordlist |

## Quick start

```python
from modules.cyber_tools import (
    password_strength, derive_key, encrypt, decrypt,
    scan_ports, audit_headers, brute_force_hits, baseline_build,
)

result = password_strength("hunter2")
# {'score': 0, 'entropy_bits': 22.5, 'verdict': 'very weak', 'patterns': [...]}

brute_force_hits("/var/log/auth.log", threshold=10)

key, salt, iterations = derive_key("my long passphrase")
blob = encrypt(key, "secret message")
assert decrypt(key, blob) == b"secret message"
```

```python
# Active testing — ONLY on systems you own or have permission to test
results = scan_ports("localhost", ports="quick")
report = audit_headers("http://localhost:8080")
```

## CLI

```bash
python -m modules.cyber_tools.clitool scan 127.0.0.1 --preset quick
python -m modules.cyber_tools.clitool audit http://localhost:8080
python -m modules.cyber_tools.clitool strength "some password"
python -m modules.cyber_tools.clitool generate --passphrase
```

File-integrity workflow:

```python
from modules.cyber_tools import baseline_build, baseline_verify
import json

baseline = baseline_build(["modules"])          # snapshot
json.dump(baseline, open("baseline.json", "w"))
later = json.load(open("baseline.json"))
print(baseline_verify(later, root="modules"))   # changed/missing/added
```

## Tests

```bash
python tests/test_cyber_tools.py    # or: pytest tests/test_cyber_tools.py
```

Known-answer digest vectors, hash-ID, strength scoring, integrity baselines,
crypto roundtrip/tamper rejection, loopback port scan, log IOC and brute-force
parsing, a live local HTTP header audit, and guard rejections for public
targets. No external network is contacted by the tests.
