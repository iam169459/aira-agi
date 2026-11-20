"""Airas developer-skills package: code auditing and project hygiene.

Stdlib-only static analysis that pairs with the cyber toolkit: T-SECRETS
and T-TAMPER techniques map here. Every finding carries an `id`, a
severity, the offending line, and a concrete fix hint.
"""
import ast
import os
import re


class Dependency:
    """One parsed requirement."""
    def __init__(self, name, spec, line, reqfile):
        self.name = name
        self.spec = spec
        self.line = line
        self.reqfile = reqfile

    def pinned(self):
        return bool(re.search(r"[=<>!~]", self.spec))


class _Visitor(ast.NodeVisitor):
    """AST pass for the rules that need structure, not text."""

    def __init__(self, filename):
        self.filename = filename
        self.findings = []
        self.func_lengths = []
        self._scope = None

    def _add(self, rule, node, message="", severity=None):
        rule_def = RULES[rule]
        self.findings.append({
            "rule": rule, "severity": severity or rule_def[0],
            "line": getattr(node, "lineno", 0),
            "message": message or rule_def[1], "fix": rule_def[2],
            "file": self.filename})

    # Each rule: (severity, message, fix hint)
    def visit_Call(self, node):
        func = node.func
        name = ""
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name in ("eval", "exec") and isinstance(func, ast.Name):
            self._add("EVAL", node,
                      message=f"use of {name}() at line {node.lineno}")
        elif name == "system" and getattr(func, "value", None) \
                is not None and getattr(func.value, "id", "") == "os":
            self._add("OSSYSTEM", node)
        elif name in ("md5", "sha1") and getattr(func, "value", None) \
                is not None and getattr(func.value, "id", "") == "hashlib":
            self._add("WEAKHASH", node,
                      message=f"hashlib.{name} at line {node.lineno} — "
                              "fine for digests, not for passwords")
        elif name == "load" and getattr(func, "value", None) is not None \
                and getattr(func.value, "id", "") in ("pickle", "yaml"):
            self._add("PICKLE", node,
                      message="pickle.load / yaml.load may execute "
                              "arbitrary code")
        self.generic_visit(node)

    def visit_keyword(self, node):
        if node.arg == "shell" and isinstance(node.value, ast.Constant) \
                and node.value.value is True:
            self._add("SHELLTRUE", node)
        elif node.arg == "verify" and isinstance(node.value, ast.Constant) \
                and node.value.value is False:
            self._add("TLSVERIFY", node)
        elif node.arg == "debug" and isinstance(node.value, ast.Constant) \
                and node.value.value is True:
            self._add("DEBUGTRUE", node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        end = getattr(node, "end_lineno", node.lineno)
        self.func_lengths.append((node.name, end - node.lineno + 1))
        args = node.args
        for default in list(args.defaults) + [
                d for d in getattr(args, "kw_defaults", []) if d]:
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                self._add("MUTABLEDEFAULT", node,
                          message=f"{node.name}() has a mutable default at "
                                  f"line {node.lineno}")
        for deco in node.decorator_list:
            if isinstance(deco, ast.Attribute) and deco.attr == "setter":
                pass
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        if node.type is None:
            body = node.body
            lone_pass = len(body) == 1 and isinstance(body[0], ast.Pass)
            self._add("BAREEXCEPT", node,
                      message="bare except (swallows all errors)"
                      + (" with just pass" if lone_pass else ""))
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module and "*" in [a.name for a in node.names]:
            self._add("WILDIMPORT", node)
        self.generic_visit(node)


RULES = {
    "EVAL": ("high", "eval/exec executes arbitrary code",
             "parse with ast.literal_eval or restructure to avoid "
             "dynamic execution"),
    "OSSYSTEM": ("high", "os.system runs a shell",
                 "use subprocess.run with a list and shell=False"),
    "SHELLTRUE": ("high", "subprocess with shell=True",
                  "pass the command as a list, shell=False"),
    "WEAKHASH": ("medium", "md5/sha1 in hashlib call",
                 "sha256+ for digests; PBKDF2/Argon2 for passwords"),
    "PICKLE": ("medium", "pickle/yaml load executes code",
               "yaml.safe_load, or unpickle only data you produced"),
    "TLSVERIFY": ("high", "HTTP client created with verify=False",
                  "keep verification on; pin a CA bundle instead"),
    "DEBUGTRUE": ("medium", "framework debug mode enabled in code",
                  "read debug from an environment variable"),
    "BAREEXCEPT": ("low", "bare except hides real bugs",
                   "except Exception, or catch the specific error"),
    "MUTABLEDEFAULT": ("medium", "mutable default argument",
                       "default to None and create inside the function"),
    "WILDIMPORT": ("low", "wildcard import pollutes the namespace",
                   "import the module or explicit names"),
    "SECRET": ("high", "possible hardcoded secret",
               "move it to an environment variable / secret manager "
               "and rotate it"),
    "SQLFORMAT": ("high", "SQL built with string formatting",
                  "parameterize the query (cursor.execute(sql, args))"),
    "TOLONGFUNC": ("low", "function is very long",
                   "split into helpers, one job per function"),
    "RANDTOKEN": ("high", "random module used where a token is built",
                  "use secrets.token_hex/token_urlsafe"),
}

_RULE_SOURCES = {
    "SECRET": re.compile(
        r"(?i)\b(?:api[_-]?key|secret|password|passwd|token|access[_-]?key)"
        r"\b\s*[:=]\s*[\"'][^\"']{8,}[\"']"),
    "SQLFORMAT": re.compile(
        r"(?i)\b(?:select|insert|update|delete)\b[^#\n]*"
        r"(?:%s|\.format\(|f[\"]{1})"),
    "RANDTOKEN": re.compile(
        r"(?i)^(?=.*\b(?:token|secret|key|password|passwd|salt|otp)\b)"
        r"(?=.*\b(?:random\.\w+|numpy\.random\b)).+$"),
}


def audit_source(source, filename="<string>"):
    """Run every rule over a block of code. Returns finding dicts."""
    visitor = _Visitor(filename)
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return [{"rule": "SYNTAX", "severity": "medium", "line": exc.lineno,
                 "message": f"syntax error: {exc.msg}", "fix": "see line",
                 "file": filename}]
    visitor.visit(tree)
    findings = list(visitor.findings)
    lines = source.splitlines()
    for idx, line in enumerate(lines, start=1):
        for rule, rx in _RULE_SOURCES.items():
            if rx.search(line):
                findings.append({
                    "rule": rule, "severity": RULES[rule][0],
                    "line": idx, "message": RULES[rule][1],
                    "fix": RULES[rule][2], "file": filename})
    for name, length in visitor.func_lengths:
        if length > 80:
            findings.append({
                "rule": "TOLONGFUNC", "severity": "low",
                "line": 0, "message": f"{name}() spans {length} lines",
                "fix": RULES["TOLONGFUNC"][2], "file": filename})
    findings.sort(key=lambda f: (-_SEVERITY_RANK[f["severity"]],
                                 f["line"]))
    return findings


_SEVERITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1}


def audit_file(path):
    """Audit one source file."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        return audit_source(fh.read(), path)


def audit_tree(root, extensions=(".py",), skip_dirs=(
        "__pycache__", ".git", ".aira", "node_modules", "archive")):
    """Audit every matching file below `root`; returns findings."""
    findings = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for name in files:
            if not name.endswith(extensions):
                continue
            findings.extend(audit_file(os.path.join(base, name)))
    findings.sort(key=lambda f: (-_SEVERITY_RANK[f["severity"]],
                                 f["file"], f["line"]))
    return findings


def summarize(findings):
    counts = {}
    for finding in findings:
        counts[finding["severity"]] = counts.get(finding["severity"], 0) + 1
    return counts


# --------------------------------------------------------------------------
def parse_requirements(path):
    deps, raw = [], []
    try:
        with open(path, encoding="utf-8") as fh:
            raw = fh.readlines()
    except OSError as exc:
        raise SystemExit(str(exc))
    for idx, line in enumerate(raw, start=1):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        entry = line.split("#")[0].strip()
        if not entry:
            continue
        match = re.match(r"^([A-Za-z0-9_.\[\]-]+)\s*(.*)$", entry)
        if not match:
            continue
        deps.append(Dependency(match.group(1).lower(), match.group(2),
                               idx, path))
    return deps


KNOWN_VULN_RANGES = {
    "requests": ("<2.20.0", "CVE-2018-18074: Authorization header "
                                     "leak on redirects"),
    "django": ("<2.2.24", "multiple path traversal / SQLi advisories "
                          "in old 1.x/2.0 lines"),
    "pillow": ("<8.1.1", "heap overflows in image parsers (2020-2021 "
                         "advisory run)"),
    "paramiko": ("<2.4.2", "CVE-2018-1000805: authentication bypass"),
    "urllib3": ("<1.24.2", "CVE-2019-11324: cert verification failure"),
}


def scan_dependencies(path):
    """Findings for one requirements file."""
    deps = parse_requirements(path)
    findings, names = [], {}
    for dep in deps:
        if dep.name in names:
            findings.append({
                "severity": "high", "line": dep.line,
                "message": f"{dep.name} is declared {len(names)}+ times",
                "fix": "keep exactly one pin per package",
                "file": dep.reqfile})
        names.setdefault(dep.name, 0)
        names[dep.name] += 1
        if dep.spec in ("", "*"):
            findings.append({
                "severity": "medium", "line": dep.line,
                "message": f"{dep.name} is unpinned — builds are not "
                           "reproducible",
                "fix": f"pin a tested version, e.g. {dep.name}==<version>",
                "file": dep.reqfile})
        elif dep.spec.startswith(">=") and dep.spec.count(".") < 2:
            findings.append({
                "severity": "low", "line": dep.line,
                "message": f"{dep.spec} admits any future major version",
                "fix": "add an upper bound (<X) once tested",
                "file": dep.reqfile})
        vuln = KNOWN_VULN_RANGES.get(dep.name)
        if vuln:
            range_, note = vuln
            spec = dep.spec
            m = re.match(r"^[<>=!~]*([0-9][0-9.]*)", spec)
            if spec.startswith("<") and m and _key(m.group(1)) <= _key(
                    range_[1:]):
                findings.append({
                    "severity": "high", "line": dep.line,
                    "message": f"{dep.name}{spec}: {note}",
                    "fix": f"bump to >= {range_[1:]}",
                    "file": dep.reqfile})
            elif not spec:
                findings.append({
                    "severity": "medium", "line": dep.line,
                    "message": f"{dep.name} unpinned while a known range "
                               f"exists ({note})",
                    "fix": f"pin >= {range_[1:]}",
                    "file": dep.reqfile})
    return deps, findings


def _key(version):
    return tuple(int(part) for part in version.split(".")[:4]
                 if part.isdigit()) + (0,)


def imports_of_tree(root, skip_dirs=("__pycache__", ".git", ".aira",
                                     "archive", "tests", "examples")):
    """Top-level local/import names used by the project's own code."""
    used = set()
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(base, name)
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    tree = ast.parse(fh.read(), filename=path)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        used.add(alias.name.split(".")[0].lower()
                                 .replace("-", "_"))
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.level == 0:
                        used.add(node.module.split(".")[0].lower()
                                 .replace("-", "_"))
    return used


def compare_imports(root, req_path):
    """Code imports vs declared requirements: used-but-undeclared and
    declared-but-unused (external packages only)."""
    import sys
    used = imports_of_tree(root)
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    local = {entry for entry in os.listdir(root)
             if os.path.isdir(os.path.join(root, entry))
             and not entry.startswith(".")}
    declared = {dep.name.replace("-", "_"): dep for dep
                in parse_requirements(req_path)}
    external = used - stdlib - local - {"setuptools", "wheel", "pip"}
    missing = sorted(n for n in external
                     if n not in declared and n not in ("os", "sys"))
    unimported = sorted(name for name in declared
                        if name not in used
                        and name not in ("setuptools", "wheel", "pip"))
    return {"missing_from_requirements": missing,
            "declared_but_unimported": unimported,
            "used_count": len(used),
            "declared_count": len(declared)}
