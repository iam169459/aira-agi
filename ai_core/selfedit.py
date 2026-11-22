"""Guarded self-rewrite system for Aira.

Aira can propose and apply changes to her own source, the way real
systems do it: proposals are validated (exact-match replacement,
syntax + compile checks), backed up, applied with automatic rollback,
and logged to an audit trail. By default every change waits for human
approval (`confirm`); `auto_confirm=True` is the user's explicit choice.
"""
import ast
import difflib
import json
import os
import py_compile
import re
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BACKUP_DIR = os.path.join(ROOT, "data", "self_edits", "backup")
DEFAULT_AUDIT_PATH = os.path.join(ROOT, "data", "self_edits",
                                  "audit.jsonl")

_MAX_SOURCE_BYTES = 200 * 1024

_SPLIT_RE = re.compile(r"\s*>>>\s*")


class SelfEditError(Exception):
    """Raised when a proposal would be unsafe or invalid."""


class SelfEditor:
    """Validated, backed-up, approval-gated rewrite of Aira's source."""

    def __init__(self, root=ROOT, backup_dir=None, audit_path=None,
                 auto_confirm=False, protected=("ai_core/selfedit.py",)):
        self.root = os.path.abspath(root)
        self.backup_dir = backup_dir or os.path.join(
            self.root, "data", "self_edits", "backup")
        self.audit_path = audit_path or os.path.join(
            self.root, "data", "self_edits", "audit.jsonl")
        self.auto_confirm = auto_confirm
        self.protected = tuple(protected)
        self.proposals = {}
        self._next_id = 1

    # ---- guard rails -------------------------------------------------

    def _resolve(self, rel_path):
        rel = (rel_path or "").strip().strip("'\"")
        path = os.path.abspath(os.path.join(self.root, rel))
        root = self.root
        if not (path == root or path.startswith(root + os.sep)):
            raise SelfEditError("only files inside this project can be "
                                "rewritten")
        norm = os.path.relpath(path, root).replace(os.sep, "/")
        if norm in self.protected:
            raise SelfEditError(f"{norm!r} is protected from self-rewrite")
        if not norm.endswith(".py"):
            raise SelfEditError("only .py files can be rewritten")
        return path, norm

    def _validate(self, path, new_source):
        try:
            ast.parse(new_source)
        except SyntaxError as exc:
            raise SelfEditError(f"proposal is not valid Python: "
                                f"{exc.msg} (line {exc.lineno})")
        tmp = path + ".selfedit.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(new_source)
            py_compile.compile(tmp, doraise=True)
        except py_compile.PyCompileError as exc:
            raise SelfEditError(f"proposal fails to compile: {exc}")
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    # ---- proposal lifecycle -------------------------------------------

    def propose(self, path, old, new):
        """Stage an exact-match rewrite proposal for one .py file."""
        full, rel = self._resolve(path)
        if not os.path.isfile(full):
            raise SelfEditError(f"no file found at {rel!r}")
        if os.path.getsize(full) > _MAX_SOURCE_BYTES:
            raise SelfEditError(f"{rel!r} is too large to rewrite")
        with open(full, encoding="utf-8") as fh:
            source = fh.read()
        old = old.rstrip("\n")
        new = new.rstrip("\n")
        if not old:
            raise SelfEditError("target text is empty")
        count = source.count(old)
        if count == 0:
            raise SelfEditError("target text not found in "
                                f"{rel!r} — nothing replaced")
        if count > 1:
            raise SelfEditError(f"target text appears {count} times in "
                                f"{rel!r}; it must be unique")
        new_source = source.replace(old, new, 1)
        if new_source == source:
            raise SelfEditError("proposal changes nothing")
        self._validate(full, new_source)
        diff = "\n".join(difflib.unified_diff(
            source.splitlines(), new_source.splitlines(),
            fromfile=f"a/{rel}", tofile=f"b/{rel}", lineterm=""))
        proposal = {
            "id": self._next_id,
            "path": rel,
            "old": old,
            "new": new,
            "diff": diff,
            "status": "staged",
            "ts": time.time(),
        }
        self.proposals[proposal["id"]] = proposal
        self._next_id += 1
        if self.auto_confirm:
            return self.confirm(proposal["id"])
        return (f"Proposal #{proposal['id']} staged for {rel!r}.\n"
                f"{diff}\n"
                f"Say 'apply proposal {proposal['id']}' to commit it.")

    def confirm(self, proposal_id):
        """Apply a staged proposal: backup, write, compile-check,
        rollback on any failure, audit-log the outcome."""
        proposal = self.proposals.get(int(proposal_id))
        if proposal is None:
            raise SelfEditError(f"no proposal #{proposal_id}")
        if proposal["status"] != "staged":
            raise SelfEditError(f"proposal #{proposal_id} is "
                                f"{proposal['status']}")
        full = os.path.join(self.root, proposal["path"])
        with open(full, encoding="utf-8") as fh:
            current = fh.read()
        if current.count(proposal["old"]) != 1:
            proposal["status"] = "conflict"
            raise SelfEditError("file changed since the proposal was "
                                "staged; proposal invalidated")
        new_source = current.replace(proposal["old"], proposal["new"], 1)
        backup = self._backup(full, proposal["path"])
        try:
            with open(full, "w", encoding="utf-8") as fh:
                fh.write(new_source)
            py_compile.compile(full, doraise=True)
        except (py_compile.PyCompileError, OSError):
            self._restore(backup, full)
            proposal["status"] = "failed"
            raise
        proposal["status"] = "applied"
        self._audit(proposal)
        return (f"Proposal #{proposal['id']} applied to "
                f"{proposal['path']!r}. Backup: {backup}")

    def cancel(self, proposal_id):
        proposal = self.proposals.pop(int(proposal_id), None)
        if proposal is None:
            raise SelfEditError(f"no proposal #{proposal_id}")
        return f"Proposal #{proposal_id} cancelled."

    def pending(self):
        staged = [p for p in self.proposals.values()
                  if p["status"] == "staged"]
        if not staged:
            return "No staged rewrite proposals."
        return "\n".join(f"#{p['id']} {p['path']}" for p in staged)

    # ---- backup / audit -------------------------------------------------

    def _backup(self, full, rel):
        os.makedirs(self.backup_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        dest = os.path.join(self.backup_dir,
                            f"{rel.replace(os.sep, '__')}_{stamp}.bak")
        shutil_copy(full, dest)
        return dest

    @staticmethod
    def _restore(backup, full):
        if os.path.exists(backup):
            shutil_copy(backup, full)

    def _audit(self, proposal):
        os.makedirs(os.path.dirname(self.audit_path) or ".", exist_ok=True)
        entry = {k: proposal[k] for k in
                 ("id", "path", "status", "ts")}
        try:
            with open(self.audit_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError:
            pass


def shutil_copy(src, dest):
    with open(src, "rb") as fsrc, open(dest, "wb") as fdst:
        fdst.write(fsrc.read())
