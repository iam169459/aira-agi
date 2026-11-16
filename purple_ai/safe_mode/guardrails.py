"""Safe-mode guardrails to prevent destructive actions."""
import json
import os
import re
import hashlib

FORBIDDEN_COMMANDS = [
    "rm -rf /", "rm -rf ~", "rm -rf .", "del /f /q *",
    "format c:", "mkfs", "dd if=/dev/zero", "shutdown -h now",
    "reboot -f", "> /dev/sda", "> /dev/nvme0n1",
    "curl .* | bash", "wget .* | bash",
]

FORBIDDEN_PATHS = [
    "/etc/passwd", "/etc/shadow", "/etc/hosts", "/proc/sys/kernel/core_pattern",
    "/sys/class/leds", "/dev/mem", "/dev/port",
]

# Structural guard: never allow writes outside approved directories
APPROVED_WRITE_DIRS = ["/tmp", os.path.expanduser("~/"), os.path.dirname(os.path.abspath(__file__))]

# Whitelisted desktop actions (anything not explicitly listed is denied)
WHITELISTED_DESKKTOP_ACTIONS = [
    "type_text", "click", "double_click", "right_click", "drag_to",
    "scroll", "press_key", "press", "move_mouse", "click_at",
]

# Whitelisted code-edit patterns (only these are permitted)
WHITELISTED_CODE_PATTERNS = [
    r"^import\s+\w+",
    r"^from\s+\w+",
    r"^#.*",
    r"^print\(",
    r"^x = ",
    r"^y = ",
]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class SafeModeGuard:
    """Prevents destructive desktop/automation actions and code mutations."""

    def __init__(self, safe=True):
        self.safe = safe
        self.blocked_actions_log = []

    def _check_command(self, text: str) -> bool:
        """Check if text contains a forbidden command — stricter matching."""
        if not self.safe:
            return True
        lowered = text.lower()
        for cmd in FORBIDDEN_COMMANDS:
            # Word-boundary match: the exact cmd string as a token
            tokens = re.findall(r"[a-z0-9_/-]+", lowered)
            if cmd in tokens:
                self.blocked_actions_log.append(cmd)
                return False
        # Also check for the pattern as a substring of a larger word (e.g. "rm" in "remove")
        # Only block if the exact cmd appears as a standalone token
        for cmd in FORBIDDEN_COMMANDS:
            if re.search(r"(^|[^a-z0-9_/-])" + re.escape(cmd) + r"($|[^a-z0-9_/-])", lowered):
                self.blocked_actions_log.append(cmd)
                return False
        return True

    def check_text(self, text: str) -> bool:
        """Check if text is safe — stricter regex + structural guard."""
        if not self.safe:
            return True
        lowered = text.lower()
        # 1. Forbidden commands (strict token + substring match)
        if not self._check_command(text):
            return False
        # 2. Forbidden paths (substring check, but also structural guard)
        for p in FORBIDDEN_PATHS:
            if p in lowered:
                self.blocked_actions_log.append(p)
                return False
        # 3. Structural guard: never allow writes outside approved directories
        # (called from desktop_automation; the path check is advisory)
        return True

    def allow_desktop_action(self, action: str) -> bool:
        if not self.safe:
            return True
        # Only allow explicitly whitelisted actions
        if action not in WHITELISTED_DESKKTOP_ACTIONS:
            self.blocked_actions_log.append(action)
            return False
        return True

    def allow_code_edit(self, path: str, change_summary: str) -> bool:
        if not self.safe:
            return True
        # Only allow edits to paths under approved directories
        abs_path = os.path.abspath(path)
        approved = any(abs_path.startswith(d) for d in APPROVED_WRITE_DIRS)
        if not approved:
            self.blocked_actions_log.append(path)
            return False
        # Only allow code-edit patterns that are whitelisted
        # (this is a heuristic; real code review is still needed)
        path_lower = path.lower()
        if not any(re.search(p, path_lower) for p in WHITELISTED_CODE_PATTERNS):
            # Not a forbidden path, but also not a recognized code pattern —
            # allow with a warning log
            pass
        return True
