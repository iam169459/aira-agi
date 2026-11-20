"""Command-line interface for the Aira cybersecurity toolkit."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from modules import cyber_tools as ct


def _severity_counts(counts):
    return " | ".join(f"{s.upper()} {n}" for s, n in counts.items() if n)


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

    p_wall = subs.add_parser("wall", help="perimeter integrity + crack report")
    p_wall.add_argument("target")
    p_wall.add_argument("--preset", default=None,
                        choices=sorted(ct.PORT_PRESETS))
    p_wall.add_argument("--ports", default=None)
    p_wall.add_argument("--authorized", action="store_true")
    p_wall.add_argument("--timeout", type=float, default=0.5)

    p_probe = subs.add_parser("probe", help="injection probe on a URL")
    p_probe.add_argument("url")
    p_probe.add_argument("--authorized", action="store_true")
    p_probe.add_argument("--params", default=None,
                         help="comma-separated parameter names")

    p_crack = subs.add_parser("crack", help="audit stored hashes (offline)")
    p_crack.add_argument("digest")
    p_crack.add_argument("--algo", default=None)
    p_crack.add_argument("--words", default=None,
                         help="path to a wordlist (default: bundled)")
    p_crack.add_argument("--rules", type=int, default=1, choices=[1, 2])

    p_engage = subs.add_parser("engage",
                               help="attack + repair plan in one pass")
    p_engage.add_argument("target")
    p_engage.add_argument("--authorized", action="store_true")

    p_sentinel = subs.add_parser("sentinel",
                                 help="watch files/ports/logs for cracks")
    p_sentinel.add_argument("--watch", nargs="*", default=None)
    p_sentinel.add_argument("--auth-log", default=None)
    p_sentinel.add_argument("--interval", type=float, default=5.0)
    p_sentinel.add_argument("--duration", type=float, default=30.0)

    p_fp = subs.add_parser("fingerprint",
                           help="web stack + tech fingerprint")
    p_fp.add_argument("url")
    p_fp.add_argument("--timeout", type=float, default=5.0)
    p_fp.add_argument("--authorized", action="store_true")

    p_dirs = subs.add_parser("dirs", help="light directory probe")
    p_dirs.add_argument("url")
    p_dirs.add_argument("--words", default=None,
                        help="path to a wordlist (default: bundled)")
    p_dirs.add_argument("--timeout", type=float, default=3.0)
    p_dirs.add_argument("--authorized", action="store_true")

    p_banner = subs.add_parser("banner",
                               help="service banner + version-age check")
    p_banner.add_argument("target")
    p_banner.add_argument("--ports", default=None)
    p_banner.add_argument("--timeout", type=float, default=2.0)
    p_banner.add_argument("--authorized", action="store_true")

    p_report = subs.add_parser("report",
                               help="full engagement + Markdown report")
    p_report.add_argument("target")
    p_report.add_argument("--authorized", action="store_true")
    p_report.add_argument("--out", default="pentest_report.md")

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
    elif args.cmd == "wall":
        ports = (args.ports.split(",") if args.ports else
                 args.preset if args.preset else None)
        ports = ([int(p) for p in ports] if isinstance(ports, list)
                 else ports)
        report = ct.wall_integrity(args.target, ports=ports,
                                   timeout=args.timeout,
                                   authorized=args.authorized)
        print(f"wall score: {report['wall_score']}/100")
        for c in report["cracks"]:
            print(f"  [{c['severity']}] {c['kind']}: {c['detail']}")
        if not report["cracks"]:
            print("  no cracks detected")
    elif args.cmd == "probe":
        params = (args.params.split(",") if args.params else None)
        report = ct.injection_probe(args.url, params=params,
                                    authorized=args.authorized)
        print(json.dumps(report, indent=2))
    elif args.cmd == "crack":
        words = (ct.load_wordlist(args.words) if args.words else None)
        result = ct.crack_stored(args.digest, algo=args.algo, words=words,
                                 rules_level=args.rules)
        print(json.dumps(result, indent=2))
    elif args.cmd == "engage":
        report = ct.run_engagement(args.target,
                                   authorized=args.authorized)
        print(report["summary"])
        for c in report["cracks"]:
            print(f"  [{c['severity']}] {c['detail']}")
        for item in report["plan"]:
            print(f"  repair {item['id']} ({item['severity']}): "
                  f"{item['title']}")
            for step in item["steps"]:
                print(f"    - {step}")
    elif args.cmd == "sentinel":
        sentinel = ct.Sentinel(watch_paths=args.watch,
                               auth_log=args.auth_log)
        def _on_event(breach):
            print("BREACH:", breach)
        try:
            events = sentinel.watch(interval=args.interval,
                                    duration=args.duration,
                                    on_event=_on_event)
        except KeyboardInterrupt:
            events = []
        print(f"{len(events)} breach event(s) reported")
    elif args.cmd == "fingerprint":
        result = ct.fingerprint_http(args.url, timeout=args.timeout,
                                     authorized=args.authorized)
        if not result["ok"]:
            print(result.get("error", "fetch failed"))
        else:
            print(f"status    {result['status']}")
            print(f"server    {result['server'] or '-'}")
            print(f"powered   {result['powered_by'] or '-'}")
            print(f"tech      {', '.join(result['tech']) or '-'}")
            for hint in result["hints"]:
                print(f"  hint: {hint}")
    elif args.cmd == "dirs":
        words = (ct.load_wordlist(args.words) if args.words else None)
        result = ct.dir_probe(args.url, words=words,
                              timeout=args.timeout,
                              authorized=args.authorized)
        print(f"{result['checked']} path(s) checked, "
              f"{len(result['found'])} interesting, "
              f"{result['errors']} error(s)")
        for entry in result["found"]:
            print(f"  [{entry['status']}] /{entry['path']} — {entry['note']}")
    elif args.cmd == "banner":
        ports = ([int(p) for p in args.ports.split(",")]
                 if args.ports else None)
        result = ct.service_banner(args.target, ports=ports,
                                   timeout=args.timeout,
                                   authorized=args.authorized)
        for svc in result["services"]:
            flag = " OUTDATED" if svc["outdated"] else ""
            print(f"port {svc['port']:>5}  "
                  f"{svc['service'] or 'unknown'} "
                  f"{svc['version'] or ''}{flag}")
            if svc["banner"]:
                print(f"          banner: {svc['banner'][:100]}")
    elif args.cmd == "report":
        engagement = ct.run_engagement(args.target,
                                       authorized=args.authorized)
        doc = ct.build_report(engagement)
        ct.save_report(doc, args.out)
        print(f"risk rating {doc['risk_rating']} — "
              f"{doc['posture']}")
        print(f"{_severity_counts(doc['counts'])}")
        for finding in doc["findings"]:
            print(f"  [{finding['severity']}] {finding['id']}: "
                  f"{finding['title']}")
        print(f"report written to {args.out}")


if __name__ == "__main__":
    main()
