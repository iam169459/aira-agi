"""Consent profiles: pre-approved authorization for your own targets.

A consent profile binds a set of targets (domains, IPs, or CIDR ranges)
to an owner assertion with an optional expiry. Once a target is covered
by a profile, every active tool accepts it through guard() with a single
--profile flag instead of asserting --authorized per command.

Profiles are stored as JSON (default .aira/consent.json) and never leave
the machine. Expiry defaults to 30 days so stale permissions do not
linger silently.
"""
import datetime
import ipaddress
import json
import os
import socket

from .guard import _targets_local

DEFAULT_STORE = os.path.join(".aira", "consent.json")

SCOPES = ("web", "ports", "full")


class ConsentStore:
    """Named authorization profiles persisted as JSON."""

    def __init__(self, path=None):
        self.path = path or os.environ.get("AIRA_CONSENT_STORE",
                                           DEFAULT_STORE)

    # -------------------------------------------------------------- profile
    def grant(self, name, targets, scope="full", ttl_days=30):
        """Create or replace a profile. Returns the stored profile."""
        if scope not in SCOPES:
            raise ValueError(
                f"scope must be one of {SCOPES}, got {scope!r}")
        targets = [str(t).strip().lower() for t in targets if str(t).strip()]
        if not targets:
            raise ValueError("at least one target is required")
        entries = {}
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as fh:
                    entries = json.load(fh)
            except (json.JSONDecodeError, OSError):
                entries = {}
        now = datetime.datetime.now(datetime.timezone.utc)
        entries[name] = {
            "targets": targets,
            "scope": scope,
            "granted_at": now.isoformat(timespec="seconds"),
            "expires_at": (now + datetime.timedelta(days=ttl_days))
            .isoformat(timespec="seconds"),
        }
        self._write(entries)
        return dict(entries[name], name=name)

    def revoke(self, name):
        """Remove a profile. Returns True when it existed."""
        entries = self._read()
        if name not in entries:
            return False
        del entries[name]
        self._write(entries)
        return True

    def get(self, name):
        profile = self._read().get(name)
        if profile is None:
            return None
        return dict(profile, name=name)

    def list(self):
        return {name: dict(profile, name=name)
                for name, profile in self._read().items()}

    # -------------------------------------------------------------- check
    def covered(self, name, target):
        """True only when profile `name` is valid, unexpired, and its
        target list covers `target`."""
        profile = self.get(name)
        if profile is None or self.scoped_out(profile):
            return False
        return target_covered(profile["targets"], target)

    @staticmethod
    def scoped_out(profile):
        expiry = profile.get("expires_at")
        if not expiry:
            return False
        try:
            when = datetime.datetime.fromisoformat(expiry)
        except ValueError:
            return True
        if when.tzinfo is None:
            when = when.replace(tzinfo=datetime.timezone.utc)
        return when < datetime.datetime.now(datetime.timezone.utc)

    def authorized(self, target, profile=None):
        """The single-shot check used by CLI flows: a target is cleared
        when it is local, covered by `profile`, or the user passed an
        explicit assertion upstream. Full scope covers every tool."""
        if profile is None:
            return False
        record = self.get(profile)
        if record is None or self.scoped_out(record):
            return False
        return target_covered(record["targets"], target)

    # -------------------------------------------------------------- io
    def _read(self):
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _write(self, entries):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(entries, fh, indent=2, ensure_ascii=False)
            fh.write("\n")


def target_covered(declared, target):
    """True when `target` matches any declared host, domain suffix, or
    CIDR range. A declared host also covers its resolved addresses."""
    if not target:
        return False
    target = str(target).strip().lower()
    if "://" in target:
        target = target.split("://", 1)[1].split("/", 1)[0]
    host = target.rsplit("@", 1)[-1].split(":", 1)[0]
    for entry in declared:
        entry = entry.strip().lower()
        if not entry:
            continue
        try:
            if ipaddress.ip_network(entry, strict=False):
                for candidate in (host, _resolve(host)):
                    if candidate and ipaddress.ip_address(
                            candidate) in ipaddress.ip_network(
                            entry, strict=False):
                        return True
                continue
        except ValueError:
            pass
        if host == entry or host.endswith("." + entry):
            return True
        resolved = _resolve(host)
        if resolved and _resolve(entry) == resolved:
            return True
    return False


def _resolve(host):
    try:
        return socket.gethostbyname(host)
    except OSError:
        return None
