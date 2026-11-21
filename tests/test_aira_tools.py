"""Tool-skill suite: registry, built-in skills, safety guards, plugin
decorator, and AiraAgent tool-loop integration."""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PASSED = 0
FAILED = 0


def check(name, cond, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ok  {name}")
    else:
        FAILED += 1
        print(f" FAIL {name} {detail}")


def test_calculator():
    from ai_core.tools import ToolRegistry
    registry = ToolRegistry(notes_path=os.devnull)
    out = registry.respond("what is 2+2")
    check("addition grounded", out and "4" in out, repr(out))
    out = registry.respond("calculate 12 * 12")
    check("multiplication grounded", out and "144" in out, repr(out))
    out = registry.respond("what is 9/2")
    check("division keeps decimals", out and "4.5" in out, repr(out))
    out = registry.respond("calculate (3+5)*2")
    check("parentheses respected", out and "16" in out, repr(out))
    check("no match for prose", registry.respond("hello there") is None)


def test_clock():
    from ai_core.tools import ToolRegistry
    registry = ToolRegistry(notes_path=os.devnull)
    out = registry.respond("what time is it") or ""
    check("clock answers", "It's" in out, repr(out))
    check("clock gives weekday", any(
        d in out for d in ("Monday", "Tuesday", "Wednesday", "Thursday",
                           "Friday", "Saturday", "Sunday")), repr(out))


def test_notes(tmp):
    from ai_core.tools import ToolRegistry
    registry = ToolRegistry(
        notes_path=os.path.join(tmp, "notes.jsonl"))
    out = registry.respond("note that buy milk tomorrow")
    check("note saved", out and "Noted" in out, repr(out))
    out = registry.respond("list my notes")
    check("note listed", out and "buy milk tomorrow" in out, repr(out))
    out = registry.respond("search my notes for milk")
    check("note search hits", out and "buy milk" in out, repr(out))
    out = registry.respond("search my notes for nothing-matches")
    check("note search miss", out and "No notes" in out, repr(out))
    check("persisted to disk",
          os.path.exists(os.path.join(tmp, "notes.jsonl")))


def test_read_file_guards():
    from ai_core.tools import ToolRegistry
    registry = ToolRegistry(notes_path=os.devnull, root=ROOT)
    sample = os.path.join(ROOT, "data", "data.txt")
    out = registry.respond(f"read file {sample}") or ""
    check("reads inside project", "Contents of" in out, repr(out[:80]))
    out = registry.respond("read file /etc/passwd")
    check("blocks outside paths", out and "only read files inside" in out,
          repr(out))
    out = registry.respond("read file definitely_missing.txt")
    check("blocks missing file", out and "No file found" in out, repr(out))


def test_plugin_decorator():
    from ai_core.tools import ToolRegistry
    registry = ToolRegistry(notes_path=os.devnull)

    @registry.tool(name="echo", patterns=[r"echo\s+(?P<arg>.+)"])
    def echo(match):
        return f"echo: {match.group('arg')}"

    check("plugin registered", "echo" in registry.names())
    out = registry.respond("echo hello world")
    check("plugin runs", out == "echo: hello world", repr(out))
    registry.unregister("echo")
    check("plugin removed", "echo" not in registry.names())


def test_agent_tool_loop():
    from ai_core.agent import AiraAgent
    from ai_core.tools import ToolRegistry
    agent = AiraAgent(
        checkpoint=os.path.join(ROOT, "missing.pt"), memory_path=None)
    out = agent.think("what is 21*2")
    check("agent answers via tool loop", "42" in out, repr(out))
    with tempfile.TemporaryDirectory() as tmp:
        registry = ToolRegistry(
            notes_path=os.path.join(tmp, "notes.jsonl"))
        agent = AiraAgent(
            checkpoint=os.path.join(ROOT, "missing.pt"),
            memory_path=None, registry=registry)
        out = agent.think("note that feed the cat")
        check("agent saves note via tool", "feed the cat" in out, repr(out))
        out = agent.think("list my notes")
        check("agent reads notes back", "feed the cat" in out, repr(out))


def main():
    print("[calculator]")
    test_calculator()
    print("[clock]")
    test_clock()
    with tempfile.TemporaryDirectory() as tmp:
        print("[notes]")
        test_notes(tmp)
    print("[read file guards]")
    test_read_file_guards()
    print("[plugin decorator]")
    test_plugin_decorator()
    print("[agent tool loop]")
    test_agent_tool_loop()
    print(f"\nResults: {PASSED} passed, {FAILED} failed")
    if FAILED:
        sys.exit(1)
    print("All tool-skill tests passed.")


if __name__ == "__main__":
    main()
