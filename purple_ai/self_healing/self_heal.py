"""Self-healing code: detects syntax errors and attempts safe auto-correction."""
import ast
import traceback
import inspect
import textwrap


class SelfHealingEngine:
    """Detects syntax errors in Python source and attempts minimal repair."""

    def __init__(self):
        self.repairs_log = []

    def diagnose_code(self, source: str, error_trace: str = "") -> list:
        errors = []
        try:
            ast.parse(source)
        except SyntaxError as e:
            msg = f"Syntax error on line {e.lineno}: {e.msg}"
            errors.append({"line": e.lineno, "msg": msg, "type": "syntax"})
            # Attempt minimal repair suggestions
            lines = source.splitlines()
            if e.lineno > 0 and e.lineno <= len(lines):
                line_content = lines[e.lineno - 1]
                if line_content.strip() == "return":
                    errors[-1]["suggestion"] = "Add a return value or complete the statement."
                elif line_content.count("(") != line_content.count(")"):
                    errors[-1]["suggestion"] = "Mismatched parentheses; check open/close parens."
                else:
                    errors[-1]["suggestion"] = "Review indentation and colons."
        except Exception as e:
            errors.append({"line": 0, "msg": str(e), "type": "unknown"})
        return errors

    def try_repair_import(self, source: str, missing: str) -> str:
        # If missing import is needed, inject at top safely
        lines = source.splitlines()
        if f"import {missing}" in source or f"from {missing}" in source:
            return source
        new_lines = [f"import {missing}"] + lines if not lines[0].startswith("import") else lines
        self.repairs_log.append(f"Injected import {missing}")
        return "\n".join(new_lines)

    def fix_indent(self, source: str) -> str:
        # Rough de-indent/re-indent attempt for common mistakes
        cleaned = textwrap.dedent(source)
        self.repairs_log.append("Re-indented source with textwrap.dedent.")
        return cleaned
