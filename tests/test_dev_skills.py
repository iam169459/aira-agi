"""Dev-skills + hacking-knowledge suite."""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PASSED = 0
FAILED = 0


def check(name, cond, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ok  {name}")
    else:
        FAILED += 1
        print(f" FAIL {name} {detail}")


BAD_CODE = '''
import pickle, hashlib, os, random, subprocess, requests
API_KEY = "supersecretkey1234"
def risky(x):
    result = eval(x)
    os.system("echo " + x)
    subprocess.run(cmd, shell=True)
    requests.get(url, verify=False)
    q = "SELECT * FROM u WHERE name='%s'" % x
    token = random.randint(1000, 9999)
    try:
        pass
    except:
        pass
    return result
def weakpw(p): return hashlib.md5(p.encode())
def loads(p): return pickle.load(open(p, "r"))
'''

CLEAN_CODE = "import os\n\n\ndef ok(x):\n    return x + 1\n"


def test_code_audit():
    from modules.dev_tools import codeaudit

    rules = {f["rule"] for f in codeaudit.audit_source(BAD_CODE)}
    need = {"SECRET", "PICKLE", "EVAL", "OSSYSTEM", "SHELLTRUE",
            "TLSVERIFY", "SQLFORMAT", "RANDTOKEN", "BAREEXCEPT",
            "WEAKHASH"}
    check("all dangerous patterns caught", need <= rules,
          need - rules)
    findings = codeaudit.audit_source(BAD_CODE)
    for finding in findings:
        ok = (finding["severity"] in codeaudit._SEVERITY_RANK
              and finding["message"] and finding["fix"]
              and finding["line"] >= 1)
        check(f"finding complete: {finding['rule']}", ok, finding)
    check("clean code clean",
          codeaudit.audit_source(CLEAN_CODE) == [])
    check("torch model.eval() not flagged",
          "EVAL" not in {f["rule"] for f in codeaudit.audit_source(
              "import torch\nm = torch.nn.Linear(2, 2)\n\n\ndef use(m):\n"
              "    return m.eval()\n")})
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(BAD_CODE)
        path = fh.name
    check("audit_file works", len(codeaudit.audit_file(path)) >= 10)
    os.unlink(path)
    counts = codeaudit.summarize(
        codeaudit.audit_source(BAD_CODE))
    check("summary counts match",
          sum(counts.values()) >= 10, counts)


def test_depscan():
    from modules.dev_tools import codeaudit

    req = ("numpy==2.1.0\nrequests<2.20.0\ndjango\nflask>=2.0\n"
           "numpy==2.1.0\n# comment\n-e .\n")
    with tempfile.NamedTemporaryFile("w", suffix=".txt",
                                     delete=False) as fh:
        fh.write(req)
        path = fh.name
    deps, findings = codeaudit.scan_dependencies(path)
    os.unlink(path)
    check("parses deps", len(deps) == 5, [d.name for d in deps])
    msgs = " | ".join(f["message"] for f in findings)
    check("known-vuln pin flagged", "requests" in msgs, msgs)
    check("unpinned flagged", "unpinned" in msgs, msgs)
    check("duplicate flagged", "times" in msgs, msgs)
    pinned = [d for d in deps if d.pinned()]
    check("pinned detection", len(pinned) == 4, [d.spec for d in deps])


def test_import_compare():
    from modules.dev_tools import codeaudit

    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "app.py"), "w") as fh:
            fh.write("import numpy\nimport yaml\nimport json\n")
        with open(os.path.join(tmp, "requirements.txt"), "w") as fh:
            fh.write("numpy\nflask\n")
        result = codeaudit.compare_imports(tmp,
                                           os.path.join(tmp,
                                                        "requirements.txt"))
        check("used counts", result["used_count"] == 3, result)
        check("unused declared", result["declared_but_unimported"]
              == ["flask"], result)
        check("undeclared used", result["missing_from_requirements"]
              == ["yaml"], result)


def test_knowledge():
    from modules.cyber_tools import knowledge as know

    check("technique count", len(know.TECHNIQUES) >= 15)
    check("methodology phases", len(know.METHODOLOGY) == 5)
    hits = know.lookup("sql injection")
    check("lookup sqli first", hits[0]["id"] == "T-SQLI",
          [t["id"] for t in hits])
    check("lookup tls",
          know.lookup("weak ciphers")[0]["id"] == "T-TLS")
    check("lookup empty", know.lookup("") == [])
    record = know.show("tls")
    check("show by short id", record and record["id"] == "T-TLS")
    check("show full id", know.show("T-SECRETS")["severity"]
          == "critical")
    check("show miss", know.show("nope") is None)
    for tech in know.TECHNIQUES:
        ok = (tech["id"].startswith("T-") and tech["summary"]
              and tech["detect"] and tech["remediate"]
              and tech["severity"] in ("critical", "high", "medium", "low"))
        check(f"record complete: {tech['id']}", ok, tech)
    check("no exploit payload instruction",
          all("payload" not in t["detect"].lower() or "marker"
              in t["detect"].lower() or "no real payloads"
              in t["detect"].lower() or "benign" in t["detect"].lower()
              for t in know.TECHNIQUES if "payload" in t["detect"].lower()))
    check("suggest risky_port", know.suggest_for_finding(
        "risky_port")["id"] == "T-PORTS")
    check("suggest header", know.suggest_for_finding(
        "header")["id"] == "T-HSTS")
    check("suggest unknown", know.suggest_for_finding("zzz") is None)


def test_exports_and_cli():
    from modules.cyber_tools import (__version__ as cyber_version,
                                     TECHNIQUES, technique_lookup)
    from modules.dev_tools import __version__ as dev_version

    check("cyber toolkit 2.3.0", cyber_version == "2.3.0", cyber_version)
    check("dev tools 1.0.0", dev_version == "1.0.0", dev_version)
    check("re-exported knowledge", len(TECHNIQUES) >= 15
          and callable(technique_lookup))

    from modules.dev_tools import clitool as devcli
    check("dev cli present", callable(devcli.main))


def main():
    print("[code audit]")
    test_code_audit()
    print("[depscan]")
    test_depscan()
    print("[import compare]")
    test_import_compare()
    print("[hacking knowledge]")
    test_knowledge()
    print("[exports + cli]")
    test_exports_and_cli()
    print(f"\nResults: {PASSED} passed, {FAILED} failed")
    if FAILED:
        sys.exit(1)
    print("All dev-skills + knowledge tests passed.")


if __name__ == "__main__":
    main()
