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
| `crack.py` | **Authorized** credential auditing: mutations-based dictionary cracking (md5/sha1/sha256, salted PBKDF2 form `salt$hash`), userdb weak-credential reports. Offline only — it never contacts a network target |
| `wall.py` | **Crack detection for the cyber wall** — dangerous-service exposure (telnet/VNC/RDP/DB ports…), TLS version downgrade detection, header hygiene, single 0–100 wall score |
| `webpwn.py` | Injection probes (reflected XSS, SQL-error fingerprints) on endpoints you are authorized to test. GET-only, non-destructive, no data extraction |
| `repair.py` | Repair engine: converts findings into a concrete remediation plan (firewall rules, TLS config, header snippets, credential rotation), plus optional dry-run/confirm file-permission hardening |
| `sentinel.py` | Continuous watch: file-tamper checksums, rogue-listener drift (`/proc/net/tcp`), brute-force surges → breach events |
| `engagement.py` | **Attack + repair in one pass**: `run_engagement` chains recon → port scan → wall integrity → fingerprint → dir probe → cipher audit → web probes → repair plan |
| `fingerprint.py` | Professional enumeration: web-stack fingerprinting (server/powered-by/cookies/generator), restrained dirb-style directory probing, service banner + version-age analysis (outdated OpenSSH/Apache/PHP…), TLS cipher-suite audit (weak/medium/strong classes) |
| `reporting.py` | **Professional pentest deliverables**: severity-ranked findings register (critical→info), executive summary, risk rating, remediation plan, raw-output appendix — rendered as Markdown or HTML via `build_report` + `save_report` |

## What is deliberately NOT included

No DoS/DDoS/flooding, no exploitation payloads, no data exfiltration. Those
are indiscriminate attack tools with no authorized-value testing use and are
_out of scope by design_. The attack-side tools here stop at detection and
evidence collection, then hand off to the repair engine.

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
python -m modules.cyber_tools.clitool wall 127.0.0.1 --preset quick
python -m modules.cyber_tools.clitool probe "http://localhost:8080/search?q=1"
python -m modules.cyber_tools.clitool crack 5f4dcc3b5aa765d61d8327deb882cf99
python -m modules.cyber_tools.clitool engage 127.0.0.1
python -m modules.cyber_tools.clitool sentinel --auth-log /var/log/auth.log --duration 60
```

### Attack + repair together

```python
from modules.cyber_tools import run_engagement, Sentinel, plan_repairs

report = run_engagement("127.0.0.1")   # attack stages + wall score + plan
print(report["summary"])               # ports, cracks, score, repair steps
for item in report["plan"]:
    print(item["id"], item["title"], item["steps"])

sentinel = Sentinel(watch_paths=["/etc"], auth_log="/var/log/auth.log")
sentinel.watch(interval=5, on_event=lambda b: print("BREACH:", b))
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
parsing, live local HTTP header audit, guard rejections; plus hash cracking
with rule mutations, wall/crack classification against real loopback
listeners, injection probing against a throwaway HTTP server, repair-plan
generation, sentinel tamper/rogue-listener detection, and engagement guard
checks. No external network is contacted and no DoS/ext./exploitation
capability exists.
