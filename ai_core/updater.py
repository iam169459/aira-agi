"""Self-updater for Aira: check GitHub and fast-forward the local checkout.

Design rules, in line with the project's own-security posture:

- fetch + compare only; ``apply()`` uses ``git pull --ff-only`` so it can
  never create merge commits or overwrite divergent local work
- refuses to touch a dirty working tree (uncommitted changes) — Aira
  will not clobber your work, and her own staged rewrites
  (``ai_core/selfedit.py``) are unaffected
- never executes fetched code; loading updates happens on the next
  restart, by you
"""
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class UpdaterError(Exception):
    """Raised when an update cannot be checked or applied safely."""


def _run(args, cwd, timeout=120):
    try:
        proc = subprocess.run(
            args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise UpdaterError(f"git not available: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise UpdaterError(f"git {args[1]} timed out") from exc
    if proc.returncode != 0:
        detail = (proc.stderr.strip() or proc.stdout.strip()
                  or f"git {' '.join(args)} failed")
        raise UpdaterError(detail)
    return proc.stdout


class Updater:
    """Check and fast-forward a local git checkout from its remote."""

    def __init__(self, root=ROOT, remote="origin", branch="main"):
        self.root = os.path.abspath(root)
        self.remote = remote
        self.branch = branch

    def _require_repo(self):
        if not os.path.isdir(os.path.join(self.root, ".git")):
            raise UpdaterError(
                f"{self.root} is not a git checkout, so I can't self-update.")

    def status(self):
        self._require_repo()
        # FETCH_HEAD works for any remote (named or bare path) without
        # needing remote-tracking refs in the local config.
        _run(["git", "fetch", self.remote, self.branch], self.root)
        behind = int((_run(["git", "rev-list", "--count",
                            "HEAD..FETCH_HEAD"],
                           self.root) or "0").strip() or 0)
        ahead = int((_run(["git", "rev-list", "--count",
                           "FETCH_HEAD..HEAD"],
                          self.root) or "0").strip() or 0)
        dirty = bool(_run(["git", "status", "--porcelain"],
                          self.root).strip())
        head = _run(["git", "log", "-1", "--format=%h %s"],
                    self.root).strip()
        return {"behind": behind, "ahead": ahead, "dirty": dirty,
                "head": head}

    def check(self):
        s = self.status()
        if s["behind"] == 0:
            return f"Already up to date (at {s['head']})."
        msg = (f"{s['behind']} commit(s) behind "
               f"{self.remote}/{self.branch} (at {s['head']}).")
        if s["dirty"]:
            msg += (" Your working tree has uncommitted changes, so I will "
                    "refuse to apply until you commit or stash.")
        return msg

    def apply(self):
        s = self.status()
        if s["dirty"]:
            raise UpdaterError(
                "Working tree has uncommitted changes; commit or stash "
                "them first — I won't clobber your work.")
        if s["ahead"]:
            raise UpdaterError(
                "Local commits diverge from "
                f"{self.remote}/{self.branch}; push or rebase them first.")
        if s["behind"] == 0:
            return f"Already up to date (at {s['head']})."
        _run(["git", "merge", "--ff-only", "FETCH_HEAD"], self.root)
        head = _run(["git", "log", "-1", "--format=%h %s"],
                    self.root).strip()
        return (f"Updated to {head}. Restart Aira and I'll be running "
                "the new code.")


def _main():
    import argparse
    parser = argparse.ArgumentParser(
        description="check for updates or fast-forward this checkout")
    parser.add_argument("--apply", action="store_true",
                        help="fast-forward to the remote (default: report only)")
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="main")
    args = parser.parse_args()
    updater = Updater(root=args.root, remote=args.remote, branch=args.branch)
    try:
        print(updater.apply() if args.apply else updater.check())
    except UpdaterError as exc:
        print(f"Update failed: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
