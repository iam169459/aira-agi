# Developer Skills (`modules/dev_tools`)

Aira's development skill set: static code auditing, dependency hygiene,
and import comparison. Python stdlib only. This is the code-side half of
the security story — technique `T-SECRETS` in the hacking knowledge base
maps here.

## Code audit (`codeaudit.py`)

Severity-ranked findings for Python source, each with a concrete fix:

| Rule | Severity | Catches |
|------|----------|---------|
| `EVAL` | high | bare `eval()` / `exec()` (torch's `model.eval()` is not flagged) |
| `OSSYSTEM` | high | `os.system(...)` shell execution |
| `SHELLTRUE` | high | `subprocess.run(..., shell=True)` |
| `TLSVERIFY` | high | `verify=False` in HTTP clients |
| `SECRET` | high | hardcoded api keys / passwords / tokens |
| `SQLFORMAT` | high | SQL built with `%`-format / `.format()` / f-strings |
| `RANDTOKEN` | high | `random.*` used to build tokens/secrets |
| `PICKLE` | medium | `pickle.load` / `yaml.load` code execution |
| `WEAKHASH` | medium | `hashlib.md5/sha1` near password work |
| `MUTABLEDEFAULT` | medium | mutable default arguments |
| `BAREEXCEPT` / `WILDIMPORT` | low | error swallows, namespace pollution |
| `TOLONGFUNC` | low | functions spanning 80+ lines |

```python
from modules.dev_tools import audit_source, audit_tree, summarize

findings = audit_tree(".")             # whole project
for f in findings:                     # severity-sorted
    print(f["severity"], f["file"], f["line"], f["rule"], f["fix"])
```

## Dependency hygiene (`scan_dependencies`, `compare_imports`)

- unpinned / wildcard / floating-range pins, duplicate entries
- curated known-vulnerable ranges (e.g. `requests<2.20.0`) with the
  advisory and the fix pin
- `compare_imports(root, req)` — code imports vs `requirements.txt`:
  used-but-undeclared and declared-but-unused

## CLI

```bash
python -m modules.dev_tools.clitool audit .            # or a single file
python -m modules.dev_tools.clitool deps requirements.txt
python -m modules.dev_tools.clitool imports . --requirements requirements.txt
python -m modules.dev_tools.clitool report . --out dev_report.md
```

The `report` command writes a Markdown file combining code + dependency
findings with per-finding fix hints.
