"""Developer-skills package: code auditing + dependency hygiene.

Pairs with the cyber toolkit (T-SECRETS / T-TAMPER map here). Python
stdlib only.
"""
from .codeaudit import (audit_source, audit_file, audit_tree, summarize,
                        parse_requirements, scan_dependencies,
                        compare_imports, imports_of_tree, RULES)

__version__ = "1.0.0"

__all__ = [
    "audit_source", "audit_file", "audit_tree", "summarize",
    "parse_requirements", "scan_dependencies", "compare_imports",
    "imports_of_tree", "RULES", "__version__",
]
