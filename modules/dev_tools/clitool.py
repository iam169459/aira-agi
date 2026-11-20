"""Developer-skills CLI: code audit, dependency scan, import compare.

    python -m modules.dev_tools.clitool audit <file-or-dir>
    python -m modules.dev_tools.clitool deps requirements.txt
    python -m modules.dev_tools.clitool imports . --requirements requirements.txt
    python -m modules.dev_tools.clitool report .   # audit + deps -> markdown
"""
import argparse
import json
import sys

from . import codeaudit


def main():
    parser = argparse.ArgumentParser(
        prog="modules.dev_tools.clitool",
        description="Airas development skills: static audit + "
                    "dependency hygiene")
    subs = parser.add_subparsers(dest="cmd", required=True)

    p_audit = subs.add_parser("audit", help="static findings for a file "
                                            "or directory")
    p_audit.add_argument("path")
    p_audit.add_argument("--json", action="store_true",
                         help="emit machine-readable JSON")

    p_deps = subs.add_parser("deps", help="findings for a requirements "
                                          "file")
    p_deps.add_argument("reqfile")

    p_imp = subs.add_parser("imports", help="compare code imports vs "
                                            "requirements")
    p_imp.add_argument("root")
    p_imp.add_argument("--requirements", default="requirements.txt")

    p_rep = subs.add_parser("report", help="audit tree + deps -> "
                                           "markdown report")
    p_rep.add_argument("root")
    p_rep.add_argument("--requirements", default="requirements.txt")
    p_rep.add_argument("--out", default="dev_report.md")

    args = parser.parse_args()

    if args.cmd == "audit":
        if args.path.endswith(".py"):
            findings = codeaudit.audit_file(args.path)
        else:
            findings = codeaudit.audit_tree(args.path)
        if args.json:
            print(json.dumps(findings, indent=2))
        else:
            _print_findings(findings)
    elif args.cmd == "deps":
        _, findings = codeaudit.scan_dependencies(args.reqfile)
        _print_findings(findings, labels=False)
    elif args.cmd == "imports":
        result = codeaudit.compare_imports(args.root, args.requirements)
        print(json.dumps(result, indent=2))
    elif args.cmd == "report":
        _write_report(args)


def _print_findings(findings, labels=True):
    counts = codeaudit.summarize(findings)
    if counts:
        print(f"{len(findings)} finding(s): "
              + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    else:
        print("no findings — clean")
        return
    for finding in findings:
        where = (f"{finding['file']}:{finding['line']}"
                 if labels and finding.get("file") else f"L{finding['line']}")
        rule = finding.get("rule", "")
        print(f"  [{finding['severity']:>7}] {rule:>16} "
              f"{where} — {finding['message']}")
        print(f"            fix: {finding['fix']}")


def _write_report(args):
    findings = codeaudit.audit_tree(args.root)
    _, deps = codeaudit.scan_dependencies(args.requirements)
    counts = codeaudit.summarize(findings)
    lines = ["# Development report", "",
             f"Audit findings: {len(findings)} "
             f"({', '.join(f'{k}={v}' for k, v in sorted(counts.items()))})",
             f"Dependency findings: {len(deps)}", "",
             "## Code", ""]
    for finding in findings:
        lines.append(f"- [{finding['severity'].upper()}] "
                     f"`{finding['rule']}` {finding['file']}:"
                     f"{finding['line']} — {finding['message']} "
                     f"_fix: {finding['fix']}_")
    lines += ["", "## Dependencies", ""]
    for dep in deps:
        lines.append(f"- [{dep['severity'].upper()}] "
                     f"{dep['file']}:{dep['line']} — {dep['message']} "
                     f"_fix: {dep['fix']}_")
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"report written to {args.out}")


if __name__ == "__main__":
    main()
