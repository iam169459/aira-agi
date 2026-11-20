"""AiraDaemon: keeps Aira always active — supervised, self-healing, awake.

A crash-isolated supervisor loop: registered duties run on their own
intervals, a duty that throws is backed off and (repeatedly failing)
disabled instead of killing the loop, every cycle writes a JSONL
heartbeat, and SIGINT/SIGTERM shut down gracefully. Loopback/private by
construction — the daemon opens no network connections.

    python -m ai_core.daemon --once                 # one supervised cycle
    python -m ai_core.daemon --interval 5           # runs until stopped
    python -m ai_core.daemon --with-agent           # load the model brain
"""
import argparse
import datetime
import json
import os
import signal
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

LOG_DIR = os.path.join(ROOT, ".aira")
DEFAULT_LOG = os.path.join(LOG_DIR, "agent_log.jsonl")

_MAX_FAILURES = 3


class Duty:
    """A named recurring task with its own schedule and health."""

    def __init__(self, name, fn, interval):
        self.name = name
        self.fn = fn
        self.interval = max(0.0, float(interval))
        self.next_due = 0.0
        self.enabled = True
        self.failures = 0
        self.runs = 0
        self.last_error = None


class AiraDaemon:
    """Supervised always-on loop with heartbeat and duty scheduling."""

    def __init__(self, agent=None, log_path=None):
        self.agent = agent
        self.log_path = log_path or DEFAULT_LOG
        self.duties = []
        self.incidents = []
        self.cycles = 0
        self.started_at = None
        self._stop = threading.Event()
        self._register_builtins()

    def register(self, name, fn, interval):
        for duty in self.duties:
            if duty.name == name:
                duty.fn, duty.interval = fn, interval
                duty.enabled = True
                duty.next_due = 0.0
                return duty
        duty = Duty(name, fn, interval)
        self.duties.append(duty)
        return duty

    def _register_builtins(self):
        self.register("heartbeat", self._heartbeat, 60.0)
        self.register("health_check", self._health_check, 120.0)
        if getattr(self.agent, "memory_path", None):
            self.register("memory_persist", self._persist_memory, 30.0)

    def _persist_memory(self, _now):
        if self.agent is not None and hasattr(self.agent, "save_memory"):
            self.agent.save_memory()

    def _heartbeat(self, _now):
        self.log_event("heartbeat", cycles=self.cycles,
                       duties=len(self.duties),
                       incidents=len(self.incidents))

    def _health_check(self, _now):
        problems = []
        for relpath in ("ai_core/model.py", "ai_core/agent.py",
                        "ai_core/daemon.py", "ai_core/main.py"):
            path = os.path.join(ROOT, relpath)
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as fh:
                source = fh.read()
            try:
                compile(source, relpath, "exec")
            except SyntaxError as exc:
                problems.append(f"{relpath}: line {exc.lineno}")
        if problems:
            raise RuntimeError("core files failed to compile: "
                               + ", ".join(problems))

    def run_cycle(self, now=None):
        """Due-duty sweep; every duty failure is isolated and recorded."""
        if self.started_at is None:
            self.started_at = self.started_at or time.time()
        now = time.time() if now is None else now
        executed, failed, skipped = [], [], []
        for duty in self.duties:
            if not duty.enabled:
                skipped.append(duty.name)
                continue
            if now < duty.next_due:
                skipped.append(duty.name)
                continue
            try:
                duty.fn(now)
                duty.runs += 1
                duty.failures = 0
                executed.append(duty.name)
            except Exception as exc:
                duty.failures += 1
                failed.append(duty.name)
                duty.last_error = f"{type(exc).__name__}: {exc}"
                self.incidents.append({
                    "cycle": self.cycles, "duty": duty.name,
                    "error": duty.last_error})
                self.log_event("duty_failed", duty=duty.name,
                               error=duty.last_error,
                               failures=duty.failures)
                if duty.failures >= _MAX_FAILURES:
                    duty.enabled = False
                    self.log_event("duty_disabled", duty=duty.name,
                                   error=duty.last_error)
            finally:
                duty.next_due = now + max(duty.interval, 0.0)
        self.cycles += 1
        return {"cycle": self.cycles, "executed": executed,
                "failed": failed, "skipped": skipped,
                "duties_enabled": sum(d.enabled for d in self.duties),
                "uptime": round(time.time() - self.started_at, 3)
                if self.started_at else 0.0}

    def log_event(self, event, **fields):
        """Append one JSONL line to the activity log."""
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        entry = {"ts": datetime.datetime.now(datetime.timezone.utc)
                       .isoformat(timespec="seconds"),
                 "event": event}
        entry.update(fields)
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
        return entry

    def run(self, interval=1.0, max_cycles=None):
        """Blocking always-on loop; returns the last cycle summary."""
        if self.started_at is None:
            self.started_at = time.time()
        signal.signal(signal.SIGINT, self._graceful)
        signal.signal(signal.SIGTERM, self._graceful)
        last = None
        while not self._stop.is_set():
            last = self.run_cycle()
            if max_cycles is not None and self.cycles >= max_cycles:
                break
            self._stop.wait(max(0.0, float(interval)))
        self.log_event("daemon_stopped", cycles=self.cycles)
        return last

    def run_forever(self, interval=1.0):
        self.log_event("daemon_started", pid=os.getpid(),
                       interval=interval)
        self.run(interval=interval)

    def _graceful(self, _signum, _frame):
        self._stop.set()

    def wake(self):
        """External poke: keep-alive from other processes."""
        self.log_event("wakeup_poke", cycles=self.cycles)
        return self.cycles


def main():
    parser = argparse.ArgumentParser(
        description="Aira always-on supervisor daemon")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--once", action="store_true",
                        help="run a single supervised cycle and exit")
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--with-agent", action="store_true",
                        help="load the model brain at startup")
    parser.add_argument("--log", default=DEFAULT_LOG)
    args = parser.parse_args()

    agent = None
    if args.with_agent:
        from ai_core.agent import AiraAgent
        agent = AiraAgent()
        print(f"brain: {'model-backed' if agent.powered_by_model else 'template voice'}")
    daemon = AiraDaemon(agent=agent, log_path=args.log)
    if args.once:
        summary = daemon.run_cycle()
        print(json.dumps(summary, indent=2))
        print(f"log: {daemon.log_path}")
        return
    print(f"aira daemon active (pid {os.getpid()}, interval "
          f"{args.interval}s, log {daemon.log_path}) — Ctrl+C to stop")
    daemon.run_forever(interval=args.interval)


if __name__ == "__main__":
    main()
