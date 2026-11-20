"""
Cybersecurity toolkit for Aira AGI - defensive auditing and
authorized-target testing utilities. Stdlib only, no external dependencies.

Every active/operational tool (port scanning, subdomain sweeps, web audits,
injection probes, engagements) refuses to touch anything except loopback
unless it is given ``authorized=True`` together with an explicit target
you control.
"""
from .guard import require_authorization, guard
from .consent import ConsentStore, target_covered
from .knowledge import (TECHNIQUES, METHODOLOGY, lookup as technique_lookup,
                        show as technique_show,
                        suggest_for_finding)
from .hashing import (digest_bytes, digest_file, hash_identifier,
                      password_strength, hmac_digest, baseline_build,
                      baseline_verify)
from .crypto import derive_key, encrypt, decrypt
from .portscan import scan_ports, COMMON_PORTS, PORT_PRESETS
from .webaudit import audit_headers, tls_expiry
from .logscan import extract_iocs, brute_force_hits
from .recon import dns_lookup, reverse_dns, subdomain_sweep
from .hashing import generate_password, generate_passphrase
from .crack import (audit_credentials, crack_stored, load_wordlist,
                    mutations)
from .wall import RISKY_PORTS, tls_version_check, wall_integrity
from .webpwn import injection_probe
from .repair import plan_repairs, secure_permissions
from .sentinel import Sentinel, listening_ports
from .engagement import run_engagement
from .fingerprint import (fingerprint_http, dir_probe, service_banner,
                          cipher_audit, DEFAULT_DIR_WORDS)
from .reporting import (build_report, collect_findings, render_markdown,
                        render_html, save_report, SEVERITY_ORDER)

__version__ = "2.3.0"

__all__ = [
    "require_authorization", "guard", "ConsentStore", "target_covered",
    "TECHNIQUES", "METHODOLOGY", "technique_lookup", "technique_show",
    "suggest_for_finding",
    "digest_bytes", "digest_file", "hash_identifier", "password_strength",
    "hmac_digest", "baseline_build", "baseline_verify",
    "derive_key", "encrypt", "decrypt",
    "scan_ports", "COMMON_PORTS", "PORT_PRESETS",
    "audit_headers", "tls_expiry",
    "extract_iocs", "brute_force_hits",
    "dns_lookup", "reverse_dns", "subdomain_sweep",
    "generate_password", "generate_passphrase",
    "load_wordlist", "mutations", "crack_stored", "audit_credentials",
    "wall_integrity", "tls_version_check", "RISKY_PORTS",
    "injection_probe", "plan_repairs", "secure_permissions",
    "Sentinel", "listening_ports", "run_engagement",
    "fingerprint_http", "dir_probe", "service_banner", "cipher_audit",
    "DEFAULT_DIR_WORDS",
    "build_report", "collect_findings", "render_markdown", "render_html",
    "save_report", "SEVERITY_ORDER",
]
