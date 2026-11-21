"""Self-rewrite suite: proposals, validation, approval gate, backup,
audit trail, conflict invalidation, and agent-loop integration."""
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


SAMPLE = ('def greet():\n    return "hi"\n\n\ndef wave():\n'
          '    return "wave"\n')


def make_root(tmp):
    import ai_core.selfedit as se
    root = os.path.join(tmp, "proj")
    core = os.path.join(root, "ai_core")
    data = os.path.join(root, "data")
    for d in (core, data):
        os.makedirs(d, exist_ok=True)
    with open(os.path.join(root, "sample.py"), "w",
              encoding="utf-8") as fh:
        fh.write(SAMPLE)
    with open(os.path.join(core, "selfedit.py"), "w",
              encoding="utf-8") as fh:
        fh.write("# guard module\n")
    return root, se


def editor(root, se, **kwargs):
    return se.SelfEditor(
        root=root,
        backup_dir=os.path.join(root, "data", "backup"),
        audit_path=os.path.join(root, "data", "audit.jsonl"),
        **kwargs)


def test_propose_and_confirm(tmp):
    root, se = make_root(tmp)
    ed = editor(root, se)
    reply = ed.propose("sample.py", 'return "hi"', 'return "hello"')
    check("staged reply", "Proposal #1 staged" in reply, reply[:80])
    with open(os.path.join(root, "sample.py"), encoding="utf-8") as fh:
        check("file untouched before approval", fh.read() == SAMPLE)
    check("pending lists it", "#1 sample.py" in ed.pending())
    out = ed.confirm(1)
    check("applied reply", "applied" in out.lower(), out[:80])
    with open(os.path.join(root, "sample.py"), encoding="utf-8") as fh:
        check("file rewritten", 'return "hello"' in fh.read())
    backups = os.listdir(os.path.join(root, "data", "backup"))
    check("backup written", len(backups) == 1, str(backups))
    check("audit written",
          os.path.exists(os.path.join(root, "data", "audit.jsonl")))


def test_validation(tmp):
    root, se = make_root(tmp)
    ed = editor(root, se)
    for name, args in (
        ("syntax error rejected", ("sample.py", 'return "hi"',
                                   'return "hi')),
        ("missing target rejected", ("sample.py", "does-not-exist",
                                     "x")),
        ("ambiguous target rejected", ("sample.py", "return",
                                       "yield")),        ("empty target rejected", ("sample.py", "", "x")),
        ("no-op rejected", ("sample.py", 'return "hi"',
                            'return "hi"')),
    ):
        try:
            ed.propose(*args)
            check(name, False, "no error raised")
        except se.SelfEditError:
            check(name, True)
    with open(os.path.join(root, "sample.py"), encoding="utf-8") as fh:
        check("file intact after rejections", fh.read() == SAMPLE)


def test_space_guards(tmp):
    root, se = make_root(tmp)
    ed = editor(root, se)
    try:
        ed.propose("/outside/evil.py", "a", "b")
        check("outside root refused", False)
    except se.SelfEditError:
        check("outside root refused", True)
    try:
        ed.propose("ai_core/selfedit.py", "guard", "x")
        check("guarded file refused", False)
    except se.SelfEditError:
        check("guarded file refused", True)
    with open(os.path.join(root, "notes.txt"), "w") as fh:
        fh.write("note")
    try:
        ed.propose("notes.txt", "note", "x")
        check("non-python refused", False)
    except se.SelfEditError:
        check("non-python refused", True)


def test_conflict_invalidates(tmp):
    root, se = make_root(tmp)
    ed = editor(root, se)
    ed.propose("sample.py", 'return "hi"', 'return "hello"')
    with open(os.path.join(root, "sample.py"), "w",
              encoding="utf-8") as fh:
        fh.write('# externally changed\nprint("moved on")\n')
    try:
        ed.confirm(1)
        check("conflict detected", False)
    except se.SelfEditError:
        check("conflict detected", True)
    with open(os.path.join(root, "sample.py"), encoding="utf-8") as fh:
        content = fh.read()
    check("conflicted file untouched",
          content.startswith("# externally changed"),
          content[:40])


def test_auto_confirm(tmp):
    root, se = make_root(tmp)
    ed = editor(root, se, auto_confirm=True)
    out = ed.propose("sample.py", 'return "hi"', 'return "hello"')
    check("auto-confirm applies", "applied" in out.lower(), out[:80])
    with open(os.path.join(root, "sample.py"), encoding="utf-8") as fh:
        check("auto-confirm wrote file", 'return "hello"' in fh.read())


def test_tool_dispatch(tmp):
    from ai_core.tools import ToolRegistry
    root, _ = make_root(tmp)
    registry = ToolRegistry(notes_path=os.devnull, root=root)
    out = registry.respond(
        'rewrite sample.py | return "hi" >>> return "hello"')
    check("dispatch stages proposal", "Proposal #1 staged" in out,
          repr(out[:60]))
    out = registry.respond("apply proposal 1")
    check("dispatch applies proposal", "applied" in out.lower(),
          repr(out[:60]))
    registry2 = ToolRegistry(notes_path=os.devnull, root=root)
    out = registry2.respond("rewrite sample.py | no-such-text >>> x")
    check("dispatch refuses bad match",
          out and out.startswith("Rewrite refused"), repr(out[:60]))
    out = registry.respond("cancel proposal 9")
    check("cancel missing proposal", "Not cancelled" in out, repr(out))


def test_agent_loop(tmp):
    from ai_core.agent import AiraAgent
    from ai_core.tools import ToolRegistry
    root, _ = make_root(tmp)
    registry = ToolRegistry(notes_path=os.devnull, root=root)
    agent = AiraAgent(
        checkpoint=os.path.join(ROOT, "missing.pt"), memory_path=None,
        registry=registry)
    out = agent.think('rewrite sample.py | return "hi" >>> return "hey"')
    check("agent stages rewrite", "Proposal" in out, repr(out[:60]))
    with open(os.path.join(root, "sample.py"), encoding="utf-8") as fh:
        check("agent still needs approval", fh.read() == SAMPLE)
    out = agent.think("apply proposal 1")
    check("agent applies rewrite", "applied" in out.lower(),
          repr(out[:60]))
    with open(os.path.join(root, "sample.py"), encoding="utf-8") as fh:
        check("agent rewrote source", 'return "hey"' in fh.read())


def main():
    print("[propose and confirm]")
    with tempfile.TemporaryDirectory() as tmp:
        test_propose_and_confirm(tmp)
    print("[validation]")
    with tempfile.TemporaryDirectory() as tmp:
        test_validation(tmp)
    print("[space guards]")
    with tempfile.TemporaryDirectory() as tmp:
        test_space_guards(tmp)
    print("[conflict]")
    with tempfile.TemporaryDirectory() as tmp:
        test_conflict_invalidates(tmp)
    print("[auto confirm]")
    with tempfile.TemporaryDirectory() as tmp:
        test_auto_confirm(tmp)
    print("[tool dispatch]")
    with tempfile.TemporaryDirectory() as tmp:
        test_tool_dispatch(tmp)
    print("[agent loop]")
    with tempfile.TemporaryDirectory() as tmp:
        test_agent_loop(tmp)
    print(f"\nResults: {PASSED} passed, {FAILED} failed")
    if FAILED:
        sys.exit(1)
    print("All self-rewrite tests passed.")


if __name__ == "__main__":
    main()
