"""Tool skill system for AiraAgent: registry, plugin decorator, and the
built-in offline skills (calculator, clock, notes, safe file read).

Anthropic-style tool loop, adapted to an intent-driven agent: incoming
text is matched against registered tool patterns; each match runs its
handler and returns a grounded answer instead of model guesswork.

Plugins register in one line:

    from ai_core.tools import default_registry

    @default_registry.tool(
        name="weather", patterns=[r"weather in (?P<arg>.+)"])
    def weather(m):
        ...
"""
import ast
import json
import operator
import os
import re
import time

from ai_core.selfedit import SelfEditError, SelfEditor

_SPLIT_RE = re.compile(r"\s*>>>\s*")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_NOTES_PATH = os.path.join(ROOT, "data", "aira_notes.jsonl")

_MAX_FILE_BYTES = 4096

_CALC_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class Tool:
    """One callable skill: regex patterns plus a handler."""

    def __init__(self, name, description, patterns, handler):
        self.name = name
        self.description = description
        self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
        self.handler = handler

    def match(self, text):
        for pattern in self.patterns:
            m = pattern.search(text)
            if m:
                return m
        return None

    def run(self, match):
        return self.handler(match)


class ToolRegistry:
    """Holds Aira's skills and runs the dispatch loop over input text."""

    def __init__(self, notes_path=DEFAULT_NOTES_PATH, root=ROOT,
                 max_tool_steps=2):
        self.notes_path = notes_path
        self.root = os.path.abspath(root)
        self.max_tool_steps = max_tool_steps
        self._tools = []
        self._register_builtins()

    @property
    def tools(self):
        return list(self._tools)

    def names(self):
        return [t.name for t in self._tools]

    def tool(self, name=None, description="", patterns=()):
        """Decorator form: @registry.tool(name=..., patterns=[...])."""
        def decorator(fn):
            self.register(Tool(name or fn.__name__, description,
                               list(patterns), fn))
            return fn
        return decorator

    def register(self, tool):
        self._tools.append(tool)
        return tool

    def unregister(self, name):
        self._tools = [t for t in self._tools if t.name != name]

    def dispatch(self, text):
        """Run up to `max_tool_steps` distinct tool matches in order.

        Returns a list of dicts with tool, arg and output keys.
        """
        used = set()
        results = []
        for tool in self._tools:
            if len(results) >= self.max_tool_steps:
                break
            if tool.name in used:
                continue
            m = tool.match(text)
            if m is None:
                continue
            used.add(tool.name)
            results.append({
                "tool": tool.name,
                "arg": (m.groupdict().get("arg") or "").strip(),
                "output": tool.run(m),
            })
        return results

    def respond(self, text):
        """Dispatch and compose a single agent-facing reply, or None."""
        results = self.dispatch(text)
        if not results:
            return None
        return "\n".join(r["output"] for r in results)

    # ---- built-in skills -------------------------------------------

    def _register_builtins(self):
        self.register(Tool(
            "calculator",
            "exact arithmetic: +, -, *, /, %, **, parentheses",
            [r"(?:what(?:'s| is)\s+)?(?:calc(?:ulate)?\s+)?"
             r"(?P<arg>[\d\s().+\-*/%]+?\*+[\d\s().+\-*/%]+|"
             r"[\d\s().+\-*/%]+[+\-/][\d\s().+\-*/%]+|"
             r"[\d\s().+\-*/%]+\s*%\s*[\d\s().+\-*/%]+|"
             r"calc(?:ulate)?\s+(?P<arg2>[\d\s().+\-*/%]+))"],
            _calculate,
        ))
        self.register(Tool(
            "clock",
            "current date, time and weekday",
            [r"(?:what(?:'s| is)\s+(?:the\s+)?(?:time|date|day)\b"
             r"|what\s+(?:day|time)\s+is\s+it\b"
             r"|today'?s\s+date\b)"],
            _clock,
        ))
        self.register(Tool(
            "notes_add",
            "save a persistent note",
            [r"(?:note that|reminder that|save a note[:,]?|"
             r"add a note[:,]?)\s*(?P<arg>.+)"],
            self._notes_add,
        ))
        self.register(Tool(
            "notes_list",
            "list and read saved notes",
            [r"(?:list|show|read|check)(?:\s+(?:my|the))?\s+notes\b"],
            self._notes_list,
        ))
        self.register(Tool(
            "notes_search",
            "search saved notes",
            [r"search(?:\s+(?:my|the))?\s+notes\s+(?:for|about)\s+"
             r"(?P<arg>.+)"],
            self._notes_search,
        ))
        self.register(Tool(
            "read_file",
            "read a file inside the project, size-capped",
            [r"(?:read|open)(?:\s+(?:the|this))?\s+file\s+(?P<arg>.+)"],
            self._read_file,
        ))
        self.editors = {}
        self.register(Tool(
            "self_edit",
            "stage a rewrite of Aira's own source; "
            "format: rewrite <file> | old code >>> new code",
            [r"rewrite\s+(?:the\s+)?(?:file\s+)?(?P<arg>.+)"],
            self._selfedit_request,
        ))
        self.register(Tool(
            "self_edit_apply",
            "apply a staged rewrite proposal",
            [r"apply(?:\s+proposal)?\s*#?\s*(?P<arg>\d+)"],
            self._selfedit_apply,
        ))
        self.register(Tool(
            "self_edit_cancel",
            "cancel a staged rewrite proposal",
            [r"cancel(?:\s+proposal)?\s*#?\s*(?P<arg>\d+)"],
            self._selfedit_cancel,
        ))
        self.register(Tool(
            "self_edit_pending",
            "list staged rewrite proposals",
            [r"(?:staged|pending|open)\s+(?:rewrite\s+)?(?:proposals|"
             r"edits)|(?:list|show)\s+(?:my\s+)?(?:rewrites|proposals)"],
            self._selfedit_pending,
        ))

    def _editor(self):
        if self.root not in self.editors:
            self.editors[self.root] = SelfEditor(
                root=self.root,
                backup_dir=os.path.join(self.root, "data", "self_edits",
                                        "backup"),
                audit_path=os.path.join(self.root, "data", "self_edits",
                                        "audit.jsonl"),
            )
        return self.editors[self.root]

    def _selfedit_request(self, match):
        arg = (match.groupdict().get("arg") or "").strip()
        if "|" not in arg or ">>>" not in arg.split("|", 1)[1]:
            return ("Give me a rewrite as: rewrite <file> | "
                    "old code >>> new code")
        rel, _, rest = arg.partition("|")
        parts = _SPLIT_RE.split(rest.strip(), maxsplit=1)
        old, new = parts[0], parts[1]
        try:
            return self._editor().propose(rel, old, new)
        except SelfEditError as exc:
            return f"Rewrite refused: {exc}"

    def _selfedit_apply(self, match):
        proposal_id = (match.groupdict().get("arg") or "").strip()
        try:
            return self._editor().confirm(proposal_id)
        except SelfEditError as exc:
            return f"Rewrite not applied: {exc}"

    def _selfedit_cancel(self, match):
        proposal_id = (match.groupdict().get("arg") or "").strip()
        try:
            return self._editor().cancel(proposal_id)
        except SelfEditError as exc:
            return f"Not cancelled: {exc}"

    def _selfedit_pending(self, match=None):
        return self._editor().pending()

    # ---- notes ------------------------------------------------------

    def _notes_load(self):
        if not os.path.exists(self.notes_path):
            return []
        notes = []
        try:
            with open(self.notes_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        notes.append(json.loads(line))
        except (json.JSONDecodeError, OSError):
            return []
        return notes

    def _notes_add(self, match):
        text = (match.groupdict().get("arg") or "").strip()
        if not text:
            return "What should I remember?"
        os.makedirs(os.path.dirname(self.notes_path) or ".", exist_ok=True)
        with open(self.notes_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.time(), "text": text},
                                ensure_ascii=False) + "\n")
        return f"Noted: {text}"

    def _notes_list(self, match=None):
        notes = self._notes_load()
        if not notes:
            return ("Your notes are empty — tell me 'note that ...' "
                    "to add one.")
        lines = [f"{i}. {n['text']}"
                 for i, n in enumerate(notes[-10:], start=1)]
        return f"Here are your last {len(lines)} notes:\n" + "\n".join(lines)

    def _notes_search(self, match):
        query = (match.groupdict().get("arg") or "").strip().lower()
        hits = [n["text"] for n in self._notes_load()
                if query and query in n["text"].lower()]
        if not hits:
            return f"No notes matched {query!r}."
        return f"Found {len(hits)} note(s):\n" + "\n".join(
            f"- {h}" for h in hits[-10:])

    def _read_file(self, match):
        raw = (match.groupdict().get("arg") or "").strip().strip("'\"")
        path = os.path.abspath(os.path.join(self.root, raw))
        root = os.path.abspath(self.root)
        if not (path == root or path.startswith(root + os.sep)):
            return "I can only read files inside this project."
        if not os.path.isfile(path):
            return f"No file found at {raw!r}."
        if os.path.getsize(path) > 10 * _MAX_FILE_BYTES:
            return f"{raw!r} is too large for me to read inline."
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                content = fh.read(_MAX_FILE_BYTES)
        except OSError:
            return f"I couldn't open {raw!r}."
        return f"Contents of {raw!r}:\n{content}"


# ---- pure handlers ---------------------------------------------------

def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value,
                                                     (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _CALC_OPS:
        return _CALC_OPS[type(node.op)](_safe_eval(node.left),
                                        _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _CALC_OPS:
        return _CALC_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def _calculate(match):
    groups = match.groupdict()
    arg = (groups.get("arg") or groups.get("arg2") or "").strip()
    if not arg:
        return None
    try:
        value = _safe_eval(ast.parse(arg, mode="eval"))
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
        return f"Sorry, I couldn't safely evaluate {arg!r}."
    if isinstance(value, float) and value.is_integer():
        return f"That comes out to {int(value)}."
    return f"That comes out to {value}."


def _clock(_match):
    stamp = time.localtime()
    return (f"It's {time.strftime('%A, %d %B %Y, %H:%M', stamp)}. "
            f"Timezone offset UTC{time.strftime('%z', stamp)[:3]}.")


default_registry = ToolRegistry()
