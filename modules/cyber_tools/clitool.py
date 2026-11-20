"""Command-line interface for the Aira cybersecurity toolkit."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from modules import cyber_tools as ct


def main():
    parser = argparse.ArgumentParser(
        prog="cyber-tools",
        description="Aira AGI cybersecurity toolkit (defensive + "
                    "authorized-target testing)")
    subs = parser.add_subparsers(dest="cmd", required=True)

    p_scan = subs.add_parser("scan", help="TCP port scan")
    p_scan.add_argument("target")
    p_scan.add_argument("--preset", default=None,
                        choices=sorted(ct.PORT_PRESETS))
    p_scan.add_argument("--ports", default=None,
                        help="comma-separated, e.g. 22,80,443")
    p_scan.add_argument("--timeout", type=float, default=0.5)
    p_scan.add_argument("--authorized", action="store_true",
                        help="confirm you own / may test this target")
    p_scan.add_argument("--no-banners", action="store_true")

    p_audit = subs.add_parser("audit", help="HTTP security-header audit")
    p_audit.add_argument("url")
    p_audit.add_argument("--authorized", action="store_true")
    p_audit.add_argument("--timeout", type=float, default=5.0)

    p_str = subs.add_parser("strength", help="score a password")
    p_str.add_argument("password")
    p_str.add_argument("--show", action="store_true",
                       help="print the password alongside the verdict")

    p_hash = subs.add_parser("hash", help="identify a digest")
    p_hash.add_argument("value")

    p_gen = subs.add_parser("generate", help="generate a strong credential")
    p_gen.add_argument("--passphrase", action="store_true")
    p_gen.add_argument("--length", type=int, default=20)

    args = parser.parse_args()

    if args.cmd == "scan":
        ports = (args.ports.split(",") if args.ports else
                 args.preset if args.preset else None)
        ports = ([int(p) for p in ports] if isinstance(ports, list)
                 else ports)
        results = ct.scan_ports(args.target, ports=ports,
                                timeout=args.timeout,
                                grab_banner=not args.no_banners,
                                authorized=args.authorized)
        for r in results:
            if r["state"] == "open":
                banner = f"  {r['banner']}" if r["banner"] else ""
                print(f"open     {r['port']:>5}  {r['service']}{banner}")
            elif r["state"] == "filtered":
                print(f"filtered {r['port']:>5}")
        print(f"{sum(1 for r in results if r['state'] == 'open')} open / "
              f"{len(results)} scanned")
    elif args.cmd == "audit":
        report = ct.audit_headers(args.url, timeout=args.timeout,
                                  authorized=args.authorized)
        print(json.dumps(report, indent=2))
    elif args.cmd == "strength":
        result = ct.password_strength(args.password)
        key = args.password if args.show else (
            args.password and "<hidden>" or "<empty>")
        print(f"{key}: {result['verdict']} (score {result['score']}/5, "
              f"{result['entropy_bits']} bits)")
        for p in result["patterns"]:
            print(f"  pattern: {p}")
    elif args.cmd == "hash":
        print(json.dumps(ct.hash_identifier(args.value), indent=2))
    elif args.cmd == "generate":
        if args.passphrase:
            print(ct.generate_passphrase())
        else:
            print(ct.generate_password(args.length))


if __name__ == "__main__":
    main()
