"""Sentinel: continuous "crack in the cyber wall" detection.

Watches three channels and reports breaches:
- file integrity drift (tamper detection via checksum baselines)
- listening-port drift (new listeners = new attack surface)
- authentication failures (brute-force campaigns in logs)
"""
import time

from .hashing import baseline_build, baseline_verify
from .logscan import brute_force_hits


def listening_ports():
    """Snapshot of local listening TCP ports (Linux best-effort)."""
    ports = set()
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 4:
                continue
            if fields[3] == "0A":  # LISTEN
                ports.add(int(fields[1].split(":")[1], 16))
    return ports


class Sentinel:
    """Detect and report cracks: tampered files, new listeners,
    brute-force surges. Optionally dispatch each breach to a handler."""

    def __init__(self, watch_paths=None, port_baseline=None,
                 auth_log=None, brute_threshold=10):
        self.watch_paths = watch_paths or []
        self._baseline = {}
        for root in self.watch_paths:
            self._baseline[root] = baseline_build([root])
        self.port_baseline = (port_baseline if port_baseline is not None
                              else listening_ports())
        self.auth_log = auth_log
        self.brute_threshold = brute_threshold
        self.log_offset = 0

    def rescan_files(self):
        """Re-baseline watched files (call after legitimate edits)."""
        for root in self.watch_paths:
            self._baseline[root] = baseline_build([root])

    def check(self):
        breach = {"files": [], "new_ports": [], "closed_ports": [],
                  "brute_force": []}
        for root, baseline in self._baseline.items():
            diff = baseline_verify(baseline, root)
            breach["files"].extend(diff["changed"] + diff["missing"]
                                   + diff["added"])
        current = listening_ports()
        breach["new_ports"] = sorted(current - self.port_baseline)
        breach["closed_ports"] = sorted(self.port_baseline - current)
        if self.auth_log:
            tail_entries = self._read_tail()
            if tail_entries:
                hits = brute_force_hits(tail_entries,
                                        threshold=self.brute_threshold)
                breach["brute_force"] = hits["suspects"]
        breach["clean"] = not (breach["files"] or breach["new_ports"]
                               or breach["brute_force"])
        return breach

    def _read_tail(self):
        try:
            with open(self.auth_log, encoding="utf-8",
                      errors="replace") as f:
                f.seek(self.log_offset)
                lines = f.read().splitlines()
                self.log_offset = f.tell()
            return lines
        except OSError:
            return []

    def watch(self, interval=5.0, duration=None, on_event=None):
        """Poll until duration elapses (or forever); dispatch breaches.

        After a breach is reported the affected channel is re-baselined,
        so the next poll only flags the *next* change.
        """
        end = time.monotonic() + duration if duration else None
        events = []
        while end is None or time.monotonic() < end:
            breach = self.check()
            if not breach["clean"]:
                events.append(breach)
                if on_event is not None:
                    on_event(breach)
                if breach["new_ports"]:
                    self.port_baseline |= set(breach["new_ports"])
                if breach["files"]:
                    self.rescan_files()
            time.sleep(interval)
        return events
