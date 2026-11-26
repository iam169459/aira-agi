"""Weapon-knowledge and offense-awareness skills for AiraAgent.

Registered in one call from a ToolRegistry (style matches more_tools):

    from ai_core import weapon_skills
    weapon_skills.register(registry, Tool)

SAFE BY CONSTRUCTION
- Weapon knowledge is FACTS ONLY (classes, history, treaties, technology
  at a glance) courtesy of modules.weapon_encyclopedia; any how-to intent
  gets the encyclopedia's refusal, never instructions.
- The cyber-offense pack (modules/cyber_tools/offense.py) is NOT callable
  from chat. This module only registers a skill that EXPLAINS the consent
  gate and points to the CLI; running offense actions requires the
  explicit authorized=True / consent-profile path in code or CLI, never a
  chat phrase.
"""
from modules import weapon_encyclopedia as enc


def _facts(match):
    if match is None:
        return enc.summary()
    groups = match.groupdict()
    query = next((v for v in groups.values() if v), None)
    query = (query or match.group(0)).strip()
    if not query:
        return enc.summary()
    return enc.lookup(query)


def _menu(_match=None):
    return enc.summary()


def _safety(match):
    _ = match
    return enc.REFUSAL


def _offense_gate(_match=None):
    return (
        "My cyber-offense skill pack is consent-gated: every action "
        "(recon sweep, attack-surface mapping, web probes, full offense "
        "run) refuses to start without either authorized=True on the "
        "call or a valid consent profile (ConsentStore.grant) covering "
        "the target - even on localhost. It is non-destructive (TCP "
        "connects, GET requests, detection strings only) and every run "
        "ends in defense findings via plan_repairs/build_report. Run it "
        "from Python or the CLI tools, not through chat; ask me for "
        "facts about any of it anytime.")


def register(registry, tool_cls):
    """Add weapon-facts and offense-gate skills to a ToolRegistry."""
    registry.register(tool_cls(
        "weapon_facts",
        "facts about weapon classes, history and arms-control treaties "
        "(never build/usage instructions)",
        [r"(?:tell me about|what\s+(?:is|are)|explain|describe)\s+"
         r"(?:the\s+|a\s+)?(?P<arg>[^?]*?)"
         r"(?:\s+(?:weapon|weapons|treaty|nuclear|missile|artillery|tank|"
         r"submarine|rifle|firearm|bomb|i ?e ?d|chemical|biological|cyber)"
         r"[a-z]*)\??",
         r"(?:history|timeline)\s+of\s+(?:weapons?|arms)\b",
         r"what\s+(?:was|is)\s+(?:the\s+)?"
         r"(?P<arg2>NPT|CWC|BWC|CTBT|ATT|CCW|INF)\s+(?:treaty)?\??",
         r"(?:tell\s+me\s+about|what\s+is)\s+(?:the\s+)?"
         r"(?P<arg3>NPT|CWC|BWC|CTBT|ATT|CCW)\b"],
        _facts,
    ))
    registry.register(tool_cls(
        "weapon_menu",
        "list the weapon topics Aira can discuss (facts only)",
        [r"(?:list|show)\s+(?:the\s+)?weapon\s+topics\b",
         r"what\s+weapons?\s+(?:topics|do\s+you\s+know)\b"],
        _menu,
    ))
    # safety net FIRST in dispatch order registration: any weapon how-to
    # phrasing gets the refusal mid-flight, so it can never leak to the
    # generative model
    registry.register(tool_cls(
        "weapon_safety",
        "refuse weapons how-to questions (facts only by policy)",
        list(enc.HOWTO_PATTERNS) + [
            r"\bhow\s+(?:to|do\s+i|can\s+i)\b.*\b(?:weapon|gun|bomb|"
            r"explosive|missile|i ?e ?d|detonat|grenade|warhead)\w*",
            r"\b(?:build|make|construct|synthesize|manufacture|assemble)"
            r"\b.*\b(?:bomb|explosive|detonat|grenade|warhead|nerve agent|"
            r"sarin| enriched uranium)\w*",
        ],
        _safety,
    ))
    registry.register(tool_cls(
        "offense_consent",
        "explain the consent gate around Aira's cyber-offense pack",
        [r"(?:attack|offense|pentest|penetration)\s+simulation\b",
         r"(?:run|start)\s+(?:an?\s+)?(?:attack|offense)\b",
         r"how\s+(?:do|does)\s+(?:your|the)\s+offense\s+(?:pack|"
         r"tools|skills)\s+work\b",
         r"(?:what\s+is\s+)?consent[- ]gat(?:e|ed|ing)\b"],
        _offense_gate,
    ))


__all__ = ["register"]
