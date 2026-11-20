"""
Cybersecurity toolkit for Aira AGI - defensive auditing and
authorized-target testing utilities. Stdlib only, no external dependencies.

Every active/operational tool (port scanning, subdomain sweeps, web audits)
refuses to touch anything except loopback unless it is given
``authorized=True`` together with an explicit target you control.
"""
from .guard import require_authorization, guard
from .hashing import (digest_bytes, digest_file, hash_identifier,
                      password_strength, hmac_digest, baseline_build,
                      baseline_verify)
from .crypto import derive_key, encrypt, decrypt
from .portscan import scan_ports, COMMON_PORTS, PORT_PRESETS
from .webaudit import audit_headers, tls_expiry
from .logscan import extract_iocs, brute_force_hits
from .recon import dns_lookup, reverse_dns, subdomain_sweep
from .hashing import generate_password, generate_passphrase

__version__ = "1.0.0"

__all__ = [
    "require_authorization", "guard",
    "digest_bytes", "digest_file", "hash_identifier", "password_strength",
    "hmac_digest", "baseline_build", "baseline_verify",
    "derive_key", "encrypt", "decrypt",
    "scan_ports", "COMMON_PORTS", "PORT_PRESETS",
    "audit_headers", "tls_expiry",
    "extract_iocs", "brute_force_hits",
    "dns_lookup", "reverse_dns", "subdomain_sweep",
    "generate_password", "generate_passphrase",
]
