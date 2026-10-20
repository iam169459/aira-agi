"""Safe-mode guardrails to prevent destructive actions."""
import re

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

class SafeModeGuard:
    """Prevents destructive desktop/automation actions and code mutations."""

    def __init__(self, safe=True):
        self.safe = safe
        self.blocked_actions_log = []

    def check_text(self, text: str) -> bool:
        if not self.safe:
            return True
        lowered = text.lower()
        for cmd in FORBIDDEN_COMMANDS:
            if re.search(cmd, lowered):
                self.blocked_actions_log.append(cmd)
                return False
        for p in FORBIDDEN_PATHS:
            if p in lowered:
                self.blocked_actions_log.append(p)
                return False
        return True

    def allow_desktop_action(self, action: str) -> bool:
        if not self.safe:
            return True
        dangerous = ["delete_file", "format_disk", "clear_desktop", "shutdown"]
        if any(d in action.lower() for d in dangerous):
            self.blocked_actions_log.append(action)
            return False
        return True

    def allow_code_edit(self, path: str, change_summary: str) -> bool:
        if not self.safe:
            return True
        if "/etc/" in path or path.startswith("/dev/"):
            self.blocked_actions_log.append(path)
            return False
        return True
