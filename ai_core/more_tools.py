"""Extra built-in tool skills for AiraAgent.

Registered in one call from ``ai_core.tools.ToolRegistry``:

    from ai_core import more_tools
    more_tools.register(registry, Tool)

Everything here is offline and local-first, matching the project's
no-external-API stance. The updater tools wrap
``ai_core.updater.Updater``: fetch + compare only, ``--ff-only`` pull,
and a hard refusal on dirty working trees.
"""
import base64
import hashlib
import json
import os
import platform
import re
import secrets
import shutil
import uuid as uuid_mod

# conversion factors to a canonical unit (metres / kilograms / bytes)
_UNIT_FACTORS = {
    "mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0,
    "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.344,
    "mg": 1e-6, "g": 0.001, "kg": 1.0, "t": 1000.0,
    "oz": 0.0283495, "lb": 0.453592, "st": 6.35029,
    "b": 1, "kb": 1024, "mb": 1024 ** 2, "gb": 1024 ** 3,
    "tb": 1024 ** 4,
}
_TEMP_UNITS = {"c", "f", "k"}
_UNIT_ALIASES = {
    "meters": "m", "meter": "m", "metres": "m", "metre": "m",
    "kilometers": "km", "kilometer": "km", "kilometres": "km",
    "kilometre": "km", "miles": "mi", "mile": "mi",
    "feet": "ft", "foot": "ft", "inches": "in", "inch": "in",
    "yards": "yd", "yard": "yd",
    "centimeters": "cm", "centimeter": "cm", "centimetres": "cm",
    "millimeters": "mm", "millimeter": "mm",
    "grams": "g", "gram": "g", "kilograms": "kg", "kilogram": "kg",
    "tons": "t", "tonne": "t", "tonnes": "t", "pounds": "lb",
    "pound": "lb", "lbs": "lb", "ounces": "oz", "ounce": "oz",
    "stones": "st", "stone": "st",
    "bytes": "b", "kib": "kb", "mib": "mb", "gib": "gb", "tib": "tb",
    "celsius": "c", "fahrenheit": "f", "kelvin": "k",
}
_ALGOS = {"sha-256": "sha256", "sha256": "sha256",
          "sha-1": "sha1", "sha1": "sha1", "md5": "md5"}


def _to_celsius(value, unit):
    if unit == "c":
        return value
    if unit == "f":
        return (value - 32) * 5 / 9
    return value - 273.15


def _from_celsius(value, unit):
    if unit == "c":
        return value
    if unit == "f":
        return value * 9 / 5 + 32
    return value + 273.15


def _pretty_number(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(round(value, 6))


def convert_units(match):
    raw = (match.groupdict().get("arg") or "").strip().lower()
    m = re.match(r"^(-?\d+(?:\.\d+)?)\s*([a-z]+)\s*(?:to|in|into)\s*"
                 r"([a-z]+)$", raw)
    if not m:
        return "Try: convert 5 km to miles, or convert 98 f to c."
    value, src, dst = float(m.group(1)), m.group(2), m.group(3)
    src = _UNIT_ALIASES.get(src, src)
    dst = _UNIT_ALIASES.get(dst, dst)
    if src in _TEMP_UNITS and dst in _TEMP_UNITS:
        out = _from_celsius(_to_celsius(value, src), dst)
    elif src in _UNIT_FACTORS and dst in _UNIT_FACTORS:
        out = value * _UNIT_FACTORS[src] / _UNIT_FACTORS[dst]
    else:
        return f"I don't know how to convert {src} to {dst} yet."
    return f"{value} {src} = {_pretty_number(out)} {dst}."


def base64_tool(match):
    groups = match.groupdict()
    text = (groups.get("arg") or "").strip()
    if groups.get("mode") == "encode":
        return ("Encoded: " + base64.b64encode(
            text.encode("utf-8")).decode("ascii"))
    padded = text + "=" * (-len(text) % 4)
    try:
        return "Decoded: " + base64.b64decode(
            padded, validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return f"{text!r} doesn't look like valid base64 to me."


def hash_text(match):
    groups = match.groupdict()
    text = (groups.get("arg") or "").strip().strip("'\"")
    algo = _ALGOS.get((groups.get("algo") or "").lower(), "sha256")
    digest = hashlib.new(algo, text.encode("utf-8")).hexdigest()
    return f"{algo} of {text!r}:\n{digest}"


def password_gen(match):
    length = int((match.groupdict().get("arg") or "").strip() or 16)
    length = max(8, min(128, length))
    pools = ("abcdefghijkmnopqrstuvwxyz", "ABCDEFGHJKLMNPQRSTUVWXYZ",
             "23456789", "!@#$%^&*()-_=+")
    chars = "".join(pools)
    picked = [secrets.choice(pool) for pool in pools]
    picked += [secrets.choice(chars) for _ in range(length - len(picked))]
    order = [secrets.randbelow(length) for _ in range(length)]
    for i in range(length):
        j = (i + order[i]) % length
        picked[i], picked[j] = picked[j], picked[i]
    return (f"Here's a {length}-char password: {''.join(picked)}\n"
            "Store it in a password manager — I don't keep a copy.")


def uuid_gen(_match):
    return f"Fresh UUID: {uuid_mod.uuid4()}"


def roll(match):
    groups = match.groupdict()
    if "coin" in (match.group(0) or "").lower():
        return f"The coin says: {secrets.choice(['heads', 'tails'])}."
    count = int(groups.get("count") or 1)
    sides = int(groups.get("sides") or 6)
    count = max(1, min(20, count))
    sides = max(2, min(1000, sides))
    rolls = [secrets.randbelow(sides) + 1 for _ in range(count)]
    body = ", ".join(str(r) for r in rolls)
    total = "" if count == 1 else f" — total {sum(rolls)}"
    return f"Rolled {count}d{sides}: {body}{total}."


def word_count(match):
    text = (match.groupdict().get("arg") or "").strip().strip("'\"")
    words = text.split()
    minutes = len(words) / 200
    return (f"{len(words)} words, {len(text)} characters — "
            f"about {max(1, round(minutes))} min to read at 200 wpm.")


def format_json(match):
    text = (match.groupdict().get("arg") or "").strip().strip("`")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return f"That JSON doesn't parse: {exc}"
    return ("Valid JSON, formatted:\n"
            + json.dumps(parsed, indent=2, ensure_ascii=False))


def sys_info(_match):
    lines = [f"Python {platform.python_version()} on "
             f"{platform.system()} {platform.release()}"]
    lines.append(f"CPU cores: {os.cpu_count() or 'unknown'}")
    try:
        usage = shutil.disk_usage(os.getcwd())
        lines.append(f"Disk: {usage.free / 2 ** 30:.1f} GB free of "
                     f"{usage.total / 2 ** 30:.1f} GB")
    except OSError:
        pass
    try:
        import psutil
        vm = psutil.virtual_memory()
        lines.append(f"RAM: {vm.available / 2 ** 30:.1f} GB available "
                     f"of {vm.total / 2 ** 30:.1f} GB")
    except ImportError:
        pass
    return "Machine status:\n" + "\n".join(f"- {line}" for line in lines)


def register(registry, tool_cls):
    """Attach every extra skill to ``registry`` (tools.py imports us)."""
    from ai_core.updater import Updater, UpdaterError

    registry.register(tool_cls(
        "convert_units",
        "unit conversion: convert 5 km to miles, convert 98 f to c",
        [r"(?:convert\s+)?(?P<arg>-?\d+(?:\.\d+)?\s*[a-z]+\s+"
         r"(?:to|in|into)\s+[a-z]+)"],
        convert_units,
    ))
    registry.register(tool_cls(
        "base64",
        "base64 encode or decode short text",
        [r"base64\s+(?P<mode>encode|decode)\s+(?P<arg>\S[\s\S]*?)\s*$"],
        base64_tool,
    ))
    registry.register(tool_cls(
        "hash_text",
        "hash text with sha256 (default), sha1 or md5",
        [r"hash\s+(?P<arg>.+?)\s+(?:with|using)\s+['\"]?"
         r"(?P<algo>sha-?256|sha-?1|md5)['\"]?\s*$",
         r"(?P<algo>sha-?256|sha-?1|md5)\s+hash\s+['\"]?"
         r"(?P<arg>.+?)['\"]?\s*$",
         r"hash\s+['\"]?(?P<arg>.+?)['\"]?\s*$"],
        hash_text,
    ))
    registry.register(tool_cls(
        "password",
        "generate a secure random password (default 16 chars)",
        [r"(?:generate|make|create|new)\s+(?:a\s+)?"
         r"(?:(?:strong|secure|safe|random)\s+)?password"
         r"(?:\s+(?:of|with|length)?\s*)?(?P<arg>\d+)?"],
        password_gen,
    ))
    registry.register(tool_cls(
        "uuid_gen",
        "generate a random UUID4",
        [r"(?:generate|new|make)\s+(?:a\s+)?uuid\b"],
        uuid_gen,
    ))
    registry.register(tool_cls(
        "roll",
        "roll dice (roll a dice, roll 2d20) or flip a coin",
        [r"flip\s+(?:a\s+)?coin\b",
         r"roll\s+(?:(?P<count>\d+)\s*)?d(?P<sides>\d+)\b",
         r"roll\s+(?:a\s+)?(?:die|dice)\b"],
        roll,
    ))
    registry.register(tool_cls(
        "word_count",
        "word/character count and reading time for a phrase",
        [r"(?:how\s+many\s+words(?:\s+are)?\s+in|count\s+"
         r"(?:the\s+)?words\s+in|word\s+count(?:\s+of)?)\s+(?P<arg>.+)"],
        word_count,
    ))
    registry.register(tool_cls(
        "format_json",
        "validate and pretty-print inline JSON",
        [r"(?s)(?:pretty-?print|pretty|format|validate)\s+"
         r"(?:this\s+)?json[:\s]+(?P<arg>.+)"],
        format_json,
    ))
    registry.register(tool_cls(
        "sys_info",
        "operating system, cpu, disk and memory status",
        [r"(?:system|computer|pc|machine)\s+info(?:rmation)?\b",
         r"how(?:'s| is)\s+my\s+(?:computer|pc|machine|system)"
         r"\s+(?:doing|running|performing)\b"],
        sys_info,
    ))

    updater = Updater(root=registry.root)

    def check_updates(_match=None):
        try:
            return updater.check()
        except UpdaterError as exc:
            return f"Update check failed: {exc}"

    def self_update(_match=None):
        try:
            return updater.apply()
        except UpdaterError as exc:
            return f"Update not applied: {exc}"

    registry.register(tool_cls(
        "check_updates",
        "compare this checkout with origin/main (read-only)",
        [r"check\s+(?:for\s+)?updates\b",
         r"am\s+i\s+up\s+to\s+date\?",
         r"what\s+version\s+are\s+you\b",
         r"are\s+you\s+up\s+to\s+date\?"],
        check_updates,
    ))
    registry.register(tool_cls(
        "self_update",
        "fast-forward this project to origin/main (never touches "
        "uncommitted work)",
        [r"update\s+(?:yourself|your\s+code|this\s+project)\b",
         r"self[- ]update\b",
         r"pull\s+(?:the\s+)?latest(?:\s+version)?\b"],
        self_update,
    ))

    def help_skills(_match=None):
        lines = [f"- {tool.name}: {tool.description}"
                 for tool in registry.tools]
        return ("Here's what I can do:\n" + "\n".join(lines)
                + "\n(plugins can add more — see ai_core/tools.py)")

    registry.register(tool_cls(
        "help",
        "list Aira's registered tools and skills",
        [r"what\s+can\s+you\s+do\b",
         r"(?:show|list)\s+(?:me\s+)?(?:your\s+)?(?:tools|skills)\b",
         r"(?:which|what)\s+(?:tools|skills)\s+(?:do\s+you\s+"
         r"have|are\s+available)\b"],
        help_skills,
    ))
